require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training DR classifier')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-input_file', '', 'model with weights')
cmd:option('-out_file_def', '', 'model definition with random weights')
cmd:option('-out_name', 'test_merge', 'file name of merged model')
cmd:text()
opt = cmd:parse(arg or {})

--assert(opt.convert_type == 0,'Conversion not supported yet.')

local final_model

local learned_model = torch.load(opt.input_file)
local new_model = require(opt.out_file_def)
local num_mods = #learned_model:get(1).modules
for mod = 1,num_mods do
  if not (learned_model:get(1):get(mod):parameters() == nil) then
    print('Updating module '..mod)
    new_model:get(1):get(mod).weight =  learned_model:get(1):get(mod).weight
    new_model:get(1):get(mod).bias =  learned_model:get(1):get(mod).bias
  end
end
final_model = new_model


torch.save(opt.out_name,final_model)
print('...model saved...')
