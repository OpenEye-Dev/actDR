require 'nn'
require 'image'

local convert_cpu = true

-- if only running CPU models, no need to load CUDA
local gpu = false
if gpu then
  require 'cunn'
  require 'cudnn'
end

local model = torch.load('model_cpu.t7')
model:evaluate()
print('model loaded:')

-- convert GPU model to CPU if is one
-- hack to accept batch size of 1 (fixed in cpu models...)
--
local is_cpu_model
if gpu then
  is_cpu_model = false
else
  is_cpu_model = model:type() == 'torch.FloatTensor'
end

if is_cpu_model then print('CPU model') else print('GPU model') end

if convert_cpu and not is_cpu_model then
  cudnn.convert(model,nn)
  model = model.modules[1]
  -- convert to CPU
  model:float()
  --model:clearState()
  is_cpu_model = true
end
-- 999{4,5,7}, 999, 9 (L/R)
local tmp_input = image.load('../kaggle_data/test_medium_png/10005_right.png')

local inputCPU
if is_cpu_model then
  inputCPU = torch.Tensor(1,3,512,512)
  inputCPU = inputCPU:float()
  tmp_input = tmp_input:float()
else
  inputCPU = torch.Tensor(150,3,512,512)
end

inputCPU[1] = tmp_input


local input
-- check if cpu model
if not is_cpu_model then
  input = torch.CudaTensor()
  input:resize(inputCPU:size()):copy(inputCPU)
  print('loaded into GPU')
else
  input = inputCPU
end
print('image loaded, memory allocated, predicting...')

local output = model:forward(input)

print(model.outputs[1])
print(output[1])
