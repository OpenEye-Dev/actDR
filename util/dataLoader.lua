-- Author: Tristan Swedish - 2016

require 'torch'
require 'image'
require 'paths'
require 'io'

-- create an class that reads in data and returns an indexed batch
local dataLoader = torch.class('dataLoader')

function dataLoader:__init(dir,labelf)
  -- shuffle seed
  self.gen = torch.Generator()
  self.batch_size = 128
  torch.manualSeed(self.gen, 1337)

  -- input image dimensions
  self.w = 512
  self.h = 512
  self.chan = 3

  -- the directory all the data is in (assumes pre-split)
  self.dataDir = dir
  -- csv of fnames and labels
  self.labelF = labelf

  self.labelList = {}
  self:genLabelList()

  self.datalist = {}
  self:genFileList()

  -- super sample underrepresented labels in datalist
  -- alpha input is range [0,1] to weight balancing
  self:balanceLabels(0.5)

  assert(#self.datalist >= 1,'no images found in folder')

  self.shuffle = torch.randperm(self.gen,#self.datalist)
  self.shuffle_state = 1

  self.numBatches = torch.floor(self.shuffle:size(1) / self.batch_size)

  collectgarbage()

end

function dataLoader:balanceLabels(alpha)
  local labelCounts = {}
  local labelSplits = {}
  for n=1,5 do
    labelCounts[n] = 0
    labelSplits[n] = {}
  end

  for i,fname in pairs(self.datalist) do
    local label = self:getLabel(fname)
    labelCounts[label] = labelCounts[label] + 1
    table.insert(labelSplits[label],fname)
  end

  local maxRep = torch.max(torch.Tensor(labelCounts))

  -- for each label, if below max represented:
  -- add a random sample from label class to bottom of datalist
  for label,count in pairs(labelCounts) do
    for n=1,torch.ceil((maxRep - count)*alpha) do
      local randSel = torch.ceil(torch.rand(1)*count)[1]
      local randPick = labelSplits[label][randSel]
      table.insert(self.datalist,randPick)
    end
  end

end

function dataLoader:genFileList()
  local datalist = paths.dir(self.dataDir)
  for n=3,#datalist do
    if string.sub(datalist[n],-3,-1) == 'png' then
      table.insert(self.datalist,string.sub(datalist[n],1,-5))
    end
  end
end

function dataLoader:genLabelList()
  -- open self.labelF, and add to table self.labelList, indexed by fname
  local labelFp = io.open(self.labelF, 'r')
  local header = labelFp:read() --don't need, can print to debug

  for line in labelFp:lines('*l') do
    local l = line:split(',')
    self.labelList[l[1]] = l[2]
  end
end

function dataLoader:getLabel(fname)
  local label = self.labelList[fname]+1 -- lua 1 indexing
  assert(not (label == nil),'label for fname not found!')
  return label
end

function dataLoader:getBatchList(num, shuff)
  if shuff == nil then
    shuff = self.shuffle_state
  end
  assert (shuff >= self.shuffle_state, 'already shuffled this far!')
  -- either this is true or equal, if equal do nothing
  while (shuff > self.shuffle_state) do
    self.shuffle = torch.randperm(self.gen,#self.datalist)
    self.shuffle_state = self.shuffle_state + 1
  end
  assert (num+self.batch_size-1 < self.shuffle:size(1),
    'batch index out of range')

  local batch_index = self.shuffle[{{num,num+self.batch_size-1}}]

  local batch_list = {}
  for n=1,self.batch_size do
    table.insert(batch_list,self.datalist[batch_index[n]])
  end
  return batch_list
end

function dataLoader:getBatch(num,shuff)
  collectgarbage()
  -- first load in file names
  local batch_list = self:getBatchList(num,shuff)
  -- now load the images
  local input = torch.Tensor(self.batch_size,self.chan,self.w,self.h)
  local labels = torch.Tensor(self.batch_size)
  for i,fname in pairs(batch_list) do
    -- get the label
    labels[i] = self:getLabel(fname)
    -- load in the image
    local tmp_input = image.load(self.dataDir..'/'..fname..'.png')
    -- transform it to fit, center crop, rotate (, then color, etc)
    -- find the smallest side
    local s_size = tmp_input:size(2)
    local s_side = 2
    if tmp_input:size(3) < s_size then
      s_side = 3
      s_size = tmp_input:size(3)
    end

    if (not (s_size == 512)) then
      -- scale to 512
      local sfactor = 512/s_size
      tmp_input = image.scale(tmp_input,'*'..tostring(sfactor))
      -- center crop
      if s_side == 2 then
        local l_y = torch.round(tmp_input:size(3)/2)-256
        input[i] = image.crop(tmp_input,0,l_y,512,l_y+512)
      else
        local l_y = torch.round(tmp_input:size(2)/2)-256
        input[i] = image.crop(tmp_input,l_y,0,l_y+512,512)
      end
    -- load the labels (integer...tranformed to softmax loss)
    else
      input[i] = tmp_input
    end
    -- rand rotate
    local randRad = ((torch.rand(1)-0.5)*0.15)[1] -- 8.5 degrees
    input[i] = image.rotate(input[i],randRad)

    -- add normalization?
  end
  return input,labels
end
