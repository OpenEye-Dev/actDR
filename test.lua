-- Author: Tristan Swedish - 2017

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'

local Threads = require 'threads'

-- options to handle:
-- pool size
-- input directory
-- label file path
-- learning rate info
-- validation, testing...relaxing oversampling
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training DR classifier')
cmd:text()
cmd:text('Options:')
-- global:
--cmd:option('-gpu', 1, 'which gpu to run job') -- int param
cmd:option('-plotid', 0, 'id of plot to keep track') -- int param
cmd:option('-test_dir','../kaggle_data/test_tiny_png/','where the test data is')
cmd:option('-labelf','../kaggle_data/totalTrainLabel.csv','file name to label map')
cmd:option('-load_model','','model file name to load')
cmd:option('-model_mode','train','training() or evaluate() dropout mode')
cmd:option('-input_size',128,'input dimension of images')
cmd:option('-resample_weight',1,'amount to resample class distribution')
cmd:option('-batch_size',128,'size of dataloader batch (to fit in memory)')
cmd:option('-pool_size',4,'number of data loader threads')
cmd:option('-log_level',1,'log level if no plotting desired, set to 0')
cmd:option('-multi_gpu',0,'run on 2 GPUS? (scale to more in future)')
cmd:text()
opt = cmd:parse(arg or {})

--local gpunum = opt.gpu
local plotid = opt.plotid
local test_dir = opt.test_dir
local labelf = opt.labelf
local model_name = opt.model_name
local val = opt.val
local input_size = opt.input_size
local batch_size = opt.batch_size
local resample_weight = opt.resample_weight

local plotter
if opt.log_level > 0 then
  require 'thirdparty/jeepers/util_thirdparty/trainplot.lua'
  plotter = TrainPlotter.new('thirdparty/jeepers/output/log/out-'..tostring(plotid)..'.json')
end

cudnn.fastest = true
cudnn.benchmark = true

local model,net

assert (not (opt.load_model == ''), 'Must load a model to test it.')
-- only run on one device

net = torch.load(opt.load_model)

if opt.multi_gpu == 1 then
  net:cuda()
  -- parallelize training over two gpus
  model = nn.DataParallelTable(1):threads(function()
    require 'cudnn'
  end)
  model:add(net, {1,2})
else
  model = net
  model:cuda()
  net = nil
end

-- set parameters to xavier?
-- use cudnn modules where possible
cudnn.convert(model,cudnn)
--cutorch.setDevice(gpunum)
if opt.model_mode == "train" then
  model:training() -- try evaluate in future?
else
  print('setting model to evaluation mode')
  model:evaluate()
end
-- Setup for regression
local criterion = nn.MSECriterion()
-- for class prob
--local criterion = nn.ClassNLLCriterion()
criterion:cuda()

require 'util.dataLoader'

local indir = train_dir
-- if not validation
local indirT
if val == 0 then
  indirT = indir
else
  indirT = test_dir
end

-- test on the true distribution
local paramTestLoader = dataLoader(indirT,labelf,resample_weight,input_size,batch_size)
local numBatchesTest = paramTestLoader.numBatches
local numBatches
print('Num test examples: '..#paramTestLoader.datalist)

-- create a data loader pool
local pool = Threads(
  opt.pool_size,
  function()
    require 'torch'
    require 'cutorch'
    require 'util.dataLoader'
  end,
  function(idx)
    dLT = dataLoader(indirT,labelf,resample_weight,input_size,batch_size)
    Thread_inputs = torch.CudaTensor()
    Thread_labels = torch.CudaTensor()
    t_inputs = torch.Tensor()
    t_labels = torch.Tensor()
    tid = idx
    print('starting thread with id '..tostring(tid))
  end
)

local function getConfusionMatrix(outputs,labels)
  local m
  local cM = torch.zeros(5,5)
  -- if one hot, else is regression
  if outputs:size(2) > 1 then
    _,m = torch.max(outputs,2)
  else
    m = torch.round(outputs):view(-1,1)
  end
  -- add to the matrix accumulator for each
  for n=1,labels:size()[1] do
    local o = m[n][1]
    -- clamp indexing to prevent indexing errors
    if o < 1 then o = 1 end
    if o > 5 then o = 5 end
    cM[labels[n]][o] = cM[labels[n]][o]+1
  end
  return cM
end

local function calcKappa(confusion_matrix)
  local O = torch.Tensor(5,5):copy(confusion_matrix)
  local E = torch.Tensor(5,5):zero()
  -- quadratic distance weight matrix
  local w = torch.Tensor({
    {0.00,0.06,0.25,0.56,1.00},
    {0.06,0.00,0.06,0.25,0.56},
    {0.25,0.06,0.00,0.06,0.25},
    {0.56,0.25,0.06,0.00,0.06},
    {1.00,0.56,0.25,0.06,0.00}
  })
  -- multiple by mask weights
  local sample_sum = O:sum()
  local expert_marginal = O:sum(2):squeeze()
  local estimate_marginal = O:sum(1):squeeze()

  local accuracy = O:trace()/sample_sum

  E:addr(expert_marginal,estimate_marginal):div(sample_sum)

  O:cmul(w)
  E:cmul(w)

  local kappa = 1 - O:sum()/E:sum()

  return kappa,accuracy
end

local loss_epoch,batchNumber

local itnum = 0
local testitnum = 0


local parameters, gradParameters = model:getParameters()

local testConfusionAcc = torch.zeros(5,5)
local err
local num_iter = 20
local batchAccumulator = torch.zeros(opt.batch_size,num_iter)
local currLabels

local function testBatch(inputs, labels, batchid)
   cutorch.synchronize()
   collectgarbage()

   local outputs
   -- do I need a backward pass?
   model:zeroGradParameters()
   outputs = model:forward(inputs)

   currLabels = labels
   batchAccumulator[{{},batchid}] = outputs:squeeze():float()

end

local function test()
  loss_epoch = 0
  batchNumber = 0
  testitnum = 0
  testConfusionAcc = torch.zeros(5,5)
  numBatches = numBatchesTest
  -- make copy so this doesn't hurt batch norm?
  --model:evaluate()
  cutorch.synchronize()

  -- limit so testing doesn't take forever
  if numBatches > 250 then numBatches = 250 end

  for i=1,numBatches do
    -- queue testBatch processes for the same batch
    -- This is equivalent to running the batch through many model runs.
    -- testBatch then updates the result accumulator
    -- queue jobs to data-workers
    batchAccumulator:zero()
    for j=1,num_iter do
      pool:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            t_inputs, t_labels = dLT:getBatch(i,1)

            if ((Thread_inputs:size() ~= t_inputs:size()) or (Thread_labels:size() ~= t_labels:size())) then
              Thread_inputs:resize(t_inputs:size())
              Thread_labels:resize(t_labels:size())
            end

            Thread_inputs:copy(t_inputs)
            Thread_labels:copy(t_labels)

            return Thread_inputs, Thread_labels, j
         end,
         -- the end callback (runs in the main thread)
         testBatch
      )
    end

    pool:synchronize()
    cutorch.synchronize()
    -- save model
    collectgarbage()

    local avg_outputs = batchAccumulator:median(2)

    err = criterion:forward(avg_outputs:cuda(), currLabels)


    local batchCM = getConfusionMatrix(avg_outputs,currLabels)
    testConfusionAcc = testConfusionAcc + batchCM

    -- if end of batch, let's see how we're doing?
    -- generate confusion matrix

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    loss_epoch = loss_epoch + err

    testitnum = testitnum + 1

    local batchOutStd = batchAccumulator:std(2):mean()
    local output_detail = torch.Tensor(opt.batch_size,3)
    output_detail[{{},1}] = batchAccumulator:std(2):squeeze():float()
    output_detail[{{},2}] = avg_outputs:squeeze():float()
    output_detail[{{},3}] = currLabels:squeeze():float()
    print(output_detail)

    _,emsg = pcall(function() plotter:add('Loss','Test',itnum+testitnum,err) end)
    if emsg then print(emsg) end
    local kappa,accuracy = calcKappa(batchCM)
    _,emsg = pcall(function() plotter:add('Accuracy','Test',itnum+testitnum,accuracy) end)
    if emsg then print(emsg) end
    _,emsg = pcall(function() plotter:add('Kappa','Test',itnum+testitnum,kappa) end)
    if emsg then print(emsg) end
    _,emsg = pcall(function() plotter:add('Batch Pred STD','Test',itnum+testitnum,batchOutStd) end)
    if emsg then print(emsg) end

    xlua.progress(i, numBatches)
  end

  print('- Test Confusion -')

  print(testConfusionAcc)
  local kappa,accuracy = calcKappa(testConfusionAcc)
  print('Test Err: '..err..' plot id: '..plotid)
  print('kappa: '..kappa..' accuracy: '..accuracy)

  local avg_loss = loss_epoch/numBatches
  return avg_loss
end
-----------------------------------------------------------------------------

local test_avg_loss = test()
print('finished test. loss: '..test_avg_loss)
