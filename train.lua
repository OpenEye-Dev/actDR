-- Author: Tristan Swedish - 2016

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'

local Threads = require 'threads'

-- options to handle:
-- pool size : 8
-- input directory
-- label file path
-- learning rate info

local net = require 'kaggle-diabetic-model.lua'

net:cuda()

-- parallelize training over two gpus
local model = nn.DataParallelTable(1):threads(function()
  require 'cudnn'
end)
model:add(net, {1,2})

-- use cudnn modules where possible
cudnn.convert(model,cudnn)

local criterion = nn.ClassNLLCriterion()
criterion:cuda()

require 'util.dataLoader'

local indir = '/mas/u/tswedish/kaggle_data/train_medium_png/'
local labelf = '/mas/u/tswedish/kaggle_data/totalTrainLabel.csv'

local optimState = {
    learningRate = 0.0001,
    learningRateDecay = 0.0,
    momentum = 0.1,
    dampening = 0.0,
    weightDecay = 0.00001
}

local paramTrainLoader = dataLoader(indir,labelf)
local numBatches = paramTrainLoader.numBatches

-- create a data loader pool
local pool = Threads(
  8,
  function()
    require 'torch'
    require 'util.dataLoader'
  end,
  function(idx)
    dL = dataLoader(indir,labelf)
    tid = idx
    print('starting thread with id '..tostring(tid))
  end
)

local loss_epoch,batchNumber

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local parameters, gradParameters = model:getParameters()

function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs
   feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      err = criterion:forward(outputs, labels)
      local gradOutputs = criterion:backward(outputs, labels)
      model:backward(inputs, gradOutputs)
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)


   -- if first batch, let's see how we're doing?
   if batchNumber == 1 then
     print(labels)
     local m
     _,m = torch.max(outputs,2)
     print(m)
   end
   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   xlua.progress(batchNumber, numBatches)

end

function train(epoch)
  loss_epoch = 0
  batchNumber = 0
  cutorch.synchronize()

  -- set the dropouts to training mode
  model:training()

  for i=1,numBatches do
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

local tot_e = 100
for e=1,100 do
  local avg_loss = train(e)
  print('finished epoch '..e..' of '..tot_e..' loss: '..avg_loss)
end
