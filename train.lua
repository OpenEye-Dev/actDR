-- Author: Tristan Swedish - 2016

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'

local Threads = require 'threads'

require 'TrainPlotter'
local plotter = TrainPlotter.new('out.json')

-- options to handle:
-- pool size
-- input directory
-- label file path
-- learning rate info
-- validation, testing...relaxing oversampling

cudnn.fastest = true
cudnn.benchmark = true

local model
local load_model = false

if not load_model then
  local net = require 'kaggle-diabetic-model-regression.lua'

  net:cuda()

  -- parallelize training over two gpus
  model = nn.DataParallelTable(1):threads(function()
    require 'cudnn'
  end)
  model:add(net, {1,2})

  -- use cudnn modules where possible
  cudnn.convert(model,cudnn)
else
  print('loading model')
  model = torch.load('model_5_10ratio_092116.t7')
end

-- Setup for regression
local criterion = nn.MSECriterion()
-- for class prob
--local criterion = nn.ClassNLLCriterion()
criterion:cuda()

require 'util.dataLoader'

local indir = '/mas/u/tswedish/kaggle_data/test_medium_png/'
local indirT = '/mas/u/tswedish/kaggle_data/train_medium_png/'
local labelf = '/mas/u/tswedish/kaggle_data/totalTrainLabel.csv'

local optimState = {
    learningRate = 0.001,
    -- with batch size of 128...150 epoch is about 75k iterations
    -- to match kaggle solution: 0.003/(1+75000*0.0001) = 0.0003
    learningRateDecay = 0.0001,
    momentum = 0.9,
    weightDecay = 0.0005
}

local paramTrainLoader = dataLoader(indir,labelf,1)
local paramTestLoader = dataLoader(indirT,labelf,1)
local numBatchesTrain = paramTrainLoader.numBatches
local numBatchesTest = paramTestLoader.numBatches
local numBatches
print('Num training examples: '..#paramTrainLoader.datalist)

-- create a data loader pool
local pool = Threads(
  8,
  function()
    require 'torch'
    require 'util.dataLoader'
  end,
  function(idx)
    dL = dataLoader(indir,labelf,1)
    dLT = dataLoader(indirT,labelf,1)
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

local loss_epoch,batchNumber

local itnum = 0

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local parameters, gradParameters = model:getParameters()

local function testBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs
   outputs = model:forward(inputs)
   err = criterion:forward(outputs, labels)

   -- if end of batch, let's see how we're doing?
   if batchNumber == numBatches-1 then
     -- generate confusion matrix
     print('-')
     print(getConfusionMatrix(outputs,labels))
     print('Err: '..err)
   end

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   xlua.progress(batchNumber, numBatches)

end

local function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs
   local feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      err = criterion:forward(outputs, labels)
      local gradOutputs = criterion:backward(outputs, labels)
      model:backward(inputs, gradOutputs)
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)


   -- if end of batch, let's see how we're doing?
   if batchNumber == 1 or batchNumber == numBatches-1 then
     -- generate confusion matrix
     print('-')
     print(getConfusionMatrix(outputs,labels))
     print('Err: '..err)
     torch.save('model_ab_'..'_kaggle_test.t7',model)
   end
   cutorch.synchronize()
   batchNumber = batchNumber + 1
   itnum = itnum + 1
   loss_epoch = loss_epoch + err
   if (itnum == 1) or (itnum % 25 == 0) then
     -- protected call becasue of weird file writing error
     -- don't print super high errors (to track delta loss on graph easier...)
     if err < 3 then
       _,emsg = pcall(function() plotter:add('Loss','Train',itnum,err) end)
       if emsg then print(emsg) end
     end
   end
   xlua.progress(batchNumber, numBatches)

end

local function test()
  loss_epoch = 0
  batchNumber = 0
  numBatches = 50 --numBatchesTest
  model:evaluate()
  cutorch.synchronize()

  for i=1,50 do
    -- queue jobs to data-workers
    pool:addjob(
       -- the job callback (runs in data-worker thread)
       function()
          local inputs, labels = dLT:getBatch(i,1)
          return inputs, labels
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

  for i=1,numBatchesTrain do
    -- queue jobs to data-workers
    pool:addjob(
       -- the job callback (runs in data-worker thread)
       function()
          local inputs, labels = dL:getBatch(i,epoch)
          return inputs, labels
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
   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   --model:clearState()
   --saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   --torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------

-- set the dropouts to training mode

local tot_e = 100
for e=1,tot_e do
  local avg_loss = train(e)
  print('finished epoch '..e..' of '..tot_e..' loss: '..avg_loss)
  local test_avg_loss = test()
  print('finished test epoch '..e..' of '..tot_e..' loss: '..test_avg_loss)

  --protected call because of weird file writing error
  _,emsg = pcall(function() plotter:add('Loss','Test',itnum,test_avg_loss) end)
  if emsg then print(emsg) end
  -- generate confusion matrix on the validation set?
  if e % 5 == 0 then
    print('saving model...')
    torch.save('model_d_'..e..'_kaggle_test.t7',model)
    torch.save('optimstate_d_'..e..'kaggle_test.t7',optimState)
  end
end

print('saving model...')
torch.save('model_kaggle_test.t7',model)
torch.save('optimstate_kaggle_test.t7',optimState)
