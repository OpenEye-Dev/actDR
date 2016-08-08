require 'torch'
require 'cutorch'
require 'nn'

local multi = {}
-- from mulitGPU.torch.util
function multi.makeDataParallel(model, nGPU)
   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
   end
   cutorch.setDevice(opt.GPU)

   return model
end

return multi
