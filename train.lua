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
cmd:option('-lr', 0.001, 'learning rate') -- int param
--cmd:option('-gpu', 1, 'which gpu to run job') -- int param
cmd:option('-plotid', 0, 'id of plot to keep track') -- int param
cmd:option('-val',0,'run validation/test code')
cmd:option('-train_dir','../kaggle_data/train_tiny_png/','where the training data is')
cmd:option('-test_dir','../kaggle_data/test_tiny_png/','where the test data is')
cmd:option('-out_path_base','output/checkpoints/test','where to save models and prefix (new_exper/model1)')
cmd:option('-labelf','../kaggle_data/totalTrainLabel.csv','file name to label map')
cmd:option('-model_name','models/kaggle-diabetic-model-tiny.lua','model file name to load (definition random weights)')
cmd:option('-load_model','','model file name to load')
cmd:option('-input_size',128,'input dimension of images')
cmd:option('-resample_weight',1,'amount to resample class distribution')
cmd:option('-batch_size',128,'size of dataloader batch (to fit in memory)')
cmd:option('-pool_size',4,'number of data loader threads')
cmd:option('-log_level',1,'log level if no plotting desired, set to 0')
cmd:option('-multi_gpu',0,'run on 2 GPUS? (scale to more in future)')
cmd:text()
opt = cmd:parse(arg or {})

local lr = opt.lr
--local gpunum = opt.gpu
local plotid = opt.plotid
local train_dir = opt.train_dir
local test_dir = opt.test_dir
local out_path_base = opt.out_path_base
local labelf = opt.labelf
local model_name = opt.model_name
local val = opt.val
local input_size = opt.input_size
local batch_size = opt.batch_size
local resample_weight = opt.resample_weight
print(lr)

local plotter
if opt.log_level > 0 then
  require 'thirdparty/jeepers/util_thirdparty/trainplot.lua'
  plotter = TrainPlotter.new('thirdparty/jeepers/output/log/out-'..tostring(plotid)..'.json')
end

cudnn.fastest = true
cudnn.benchmark = true

local model,net

if not (opt.load_model == '') then
  print('loading model... '..opt.load_model)
  net = torch.load(opt.load_model)
else
-- only run on one device
  net = require(model_name)
end

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
model:training()
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

local optimState = {
    learningRate = lr,
    -- with batch size of 128...150 epoch is about 75k iterations
    -- to match kaggle solution: 0.003/(1+75000*0.0001) = 0.0003
    learningRateDecay = 0.00001,
    momentum = 0.9,
    weightDecay = 0.0005
}

local paramTrainLoader = dataLoader(indir,labelf,resample_weight,input_size,batch_size)
-- test on the true distribution
local paramTestLoader = dataLoader(indirT,labelf,resample_weight,input_size,batch_size)
local numBatchesTrain = paramTrainLoader.numBatches
local numBatchesTest = paramTestLoader.numBatches
local numBatches
print('Num training examples: '..#paramTrainLoader.datalist)

-- create a data loader pool
local pool = Threads(
  opt.pool_size,
  function()
    require 'torch'
    require 'cutorch'
    require 'util.dataLoader'
  end,
  function(idx)
    dL = dataLoader(indir,labelf,resample_weight,input_size,batch_size)
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
local trainConfusionAcc = torch.zeros(5,5)

local function testBatch(inputs, labels)
   cutorch.synchronize()
   collectgarbage()

   local err = 0
   local outputs
   -- do I need a backward pass?
   model:zeroGradParameters()
   outputs = model:forward(inputs)
   err = criterion:forward(outputs, labels)
   --local gradOutputs = criterion:backward(outputs, labels)
   --model:backward(inputs, gradOutputs)
   local batchCM = getConfusionMatrix(outputs,labels)
   testConfusionAcc = testConfusionAcc + batchCM

   -- if end of batch, let's see how we're doing?
   if batchNumber == numBatches-1 then
     -- generate confusion matrix
     print('- Test Confusion -')
     print(testConfusionAcc)
     local kappa,accuracy = calcKappa(testConfusionAcc)
     print('Test Err: '..err..' plot id: '..plotid)
     print('kappa: '..kappa..' accuracy: '..accuracy)
   end

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err

   testitnum = testitnum + 1

   if itnum > 100 then
     _,emsg = pcall(function() plotter:add('Loss','Test',itnum+testitnum,err) end)
     if emsg then print(emsg) end
     local kappa,accuracy = calcKappa(batchCM)
     _,emsg = pcall(function() plotter:add('Accuracy','Test',itnum+testitnum,accuracy) end)
     if emsg then print(emsg) end
     _,emsg = pcall(function() plotter:add('Kappa','Test',itnum+testitnum,kappa) end)
     if emsg then print(emsg) end
   end

   xlua.progress(batchNumber, numBatches)

end

local function trainBatch(inputs, labels, preproc_time)
   --print(inputsCPU[{{1},{},{1},{1}}])
   --print(labelsCPU[1])
   cutorch.synchronize()
   collectgarbage()

   -- transfer over to GPU
   --inputs:resize(inputsCPU:size()):copy(inputsCPU)
   --labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err = 0
   local outputs
   local batch_timer = torch.Timer()

   local feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      err = criterion:forward(outputs, labels)
      local gradOutputs = criterion:backward(outputs, labels)
      model:backward(inputs, gradOutputs)
      return err, gradParameters
   end
   local start_b = batch_timer:time().real
   optim.sgd(feval, parameters, optimState)
   local batch_time_elapsed = batch_timer:time().real - start_b

   local batchCM = getConfusionMatrix(outputs,labels)
   trainConfusionAcc = trainConfusionAcc + batchCM

   -- if end of batch, let's see how we're doing?
   if batchNumber == numBatches-1 then
     -- generate confusion matrix
     print('- Train Confusion -')
     print(trainConfusionAcc)
     local kappa,accuracy = calcKappa(trainConfusionAcc)
     print('Train Err: '..err..' plot id: '..plotid)
     print('kappa: '..kappa..' accuracy: '..accuracy)
     --torch.save('model_tiny_ab_temp_kaggle_test.t7',model)
   end
   cutorch.synchronize()
   batchNumber = batchNumber + 1
   itnum = itnum + 1
   loss_epoch = loss_epoch + err
   --if (itnum == 1) or (itnum % 1 == 0) then
     -- protected call becasue of weird file writing error
     -- don't print super high errors (to track delta loss on graph easier...)
     --if err < 3 then
   if itnum > 100 then -- remove init out of whack
     _,emsg = pcall(function() plotter:add('Loss','Train',itnum,err) end)
     if emsg then print(emsg) end
     local kappa,accuracy = calcKappa(batchCM)
     _,emsg = pcall(function() plotter:add('Accuracy','Train',itnum,accuracy) end)
     if emsg then print(emsg) end
     _,emsg = pcall(function() plotter:add('Kappa','Train',itnum,kappa) end)
     if emsg then print(emsg) end
     _,emsg = pcall(function() plotter:add('Thread Limited Factor','Train',itnum,preproc_time/(batch_time_elapsed*opt.pool_size)) end)
     if emsg then print(emsg) end
     _,emsg = pcall(function() plotter:add('Pre-Process Time','Train',itnum,preproc_time) end)
     if emsg then print(emsg) end
   end
     --end
   --end
   xlua.progress(batchNumber, numBatches)

end

local function test(epoch)
  loss_epoch = 0
  batchNumber = 0
  testitnum = 0
  testConfusionAcc = torch.zeros(5,5)
  numBatches = numBatchesTest
  -- make copy so this doesn't hurt batch norm?
  --model:evaluate()
  cutorch.synchronize()

  -- limit so testing doesn't take forever
  if numBatches > 100 then numBatches = 100 end

  for i=1,numBatches do
    -- queue jobs to data-workers
    pool:addjob(
       -- the job callback (runs in data-worker thread)
       function()
          t_inputs, t_labels = dLT:getBatch(i,epoch)

          if ((Thread_inputs:size() ~= t_inputs:size()) or (Thread_labels:size() ~= t_labels:size())) then
            Thread_inputs:resize(t_inputs:size())
            Thread_labels:resize(t_labels:size())
          end

          Thread_inputs:copy(t_inputs)
          Thread_labels:copy(t_labels)

          return Thread_inputs, Thread_labels
       end,
       -- the end callback (runs in the main thread)
       testBatch
    )
  end

  pool:synchronize()
  cutorch.synchronize()
  -- save model
  collectgarbage()

  local avg_loss = loss_epoch/numBatches
  return avg_loss
end

local function train(epoch)
  loss_epoch = 0
  batchNumber = 0
  numBatches = numBatchesTrain
  model:training()
  cutorch.synchronize()
  trainConfusionAcc = torch.zeros(5,5)

  for i=1,numBatchesTrain do
    -- queue jobs to data-workers
    pool:addjob(
       -- the job callback (runs in data-worker thread)
       function()
         local thread_timer = torch.Timer()
         local process_start = thread_timer:time().real
         t_inputs, t_labels = dL:getBatch(i,epoch)

         if ((Thread_inputs:size() ~= t_inputs:size()) or (Thread_labels:size() ~= t_labels:size())) then
           Thread_inputs:resize(t_inputs:size())
           Thread_labels:resize(t_labels:size())
         end

         Thread_inputs:copy(t_inputs)
         Thread_labels:copy(t_labels)

         local thread_elapsed = thread_timer:time().real - process_start

         return Thread_inputs, Thread_labels, thread_elapsed
       end,
       -- the end callback (runs in the main thread)
       trainBatch
    )
  end

  pool:synchronize()
  cutorch.synchronize()
  -- save model
  collectgarbage()

  local avg_loss = loss_epoch/numBatches
  return avg_loss
end
-----------------------------------------------------------------------------


local tot_e = 100
for e=1,tot_e do
  local avg_loss = train(e)
  print('finished epoch '..e..' of '..tot_e..' loss: '..avg_loss)
  -- check if we wanna run validation
  if val ~= 0 then
    local test_avg_loss = test(e)
    print('finished test epoch '..e..' of '..tot_e..' loss: '..test_avg_loss)
    --protected call because of weird file writing error
  end
  -- clear the intermediate states in the model before saving to disk
  -- this saves lots of disk space
  model:clearState()
  local temp_save
  if opt.multi_gpu == 1 then
    temp_save = model.modules[1]:clone()
  else
    temp_save = model:clone()
  end
  temp_save:clearState()
  torch.save(out_path_base..'_temp_model_'..tostring(lr)..'_id_'..tostring(plotid)..'.t7',temp_save)
  torch.save(out_path_base..'_temp_optimstate_'..tostring(lr)..'_id_'..tostring(plotid)..'.t7',optimState)
  if e % 10 == 0 then
    print('saving model...')
    torch.save(out_path_base..'_model_'..e..'_'..tostring(lr)..'_id_'..tostring(plotid)..'.t7',temp_save)
    torch.save(out_path_base..'_optimstate_'..e..'_'..tostring(lr)..'_id_'..tostring(plotid)..'.t7',optimState)
  end
end
