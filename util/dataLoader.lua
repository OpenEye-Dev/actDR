-- Author: Tristan Swedish - 2016

require 'torch'
require 'image'
require 'paths'
require 'io'

-- create an class that reads in data and returns an indexed batch
local dataLoader = torch.class('dataLoader')

function dataLoader:__init(dir,labelf,rebalance)
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

  self.transformer = self:generateTransformer()

  -- super sample underrepresented labels in datalist
  -- alpha input is range [0,1] to weight balancing
  if rebalance == nil then rebalance = 0 end
  self:balanceLabels(rebalance)

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

function dataLoader:generateTransformer()
  -- generate a transform function based on some other stuff
  -- normalize and add color jitter/ geometric transforms
  local T = require('util.transform')
  local transforms = {}
  -- scale, centercrop, scale/zoom, rotate, randcrop, jitter, normalize, pca

  -- to accomodate zooms 1/1.15 to 1.15
  -- also want to have some random crop jitter of 20 pixels
  -- zoom transform
  transforms[1] = T.RandomScale(532,604)
  -- rotate (still centered around circle)
  transforms[2] = T.Rotation(360)
  transforms[3] = T.CenterCrop(532)

  transforms[4] = T.CenterCrop(512)

  -- color jitter
  local jitteropt = {}
  jitteropt.brightness = 0.05
  jitteropt.saturation = 0.05
  jitteropt.contrast = 0.05
  transforms[5] = T.ColorJitter(jitteropt)

  -- normalize
  local meanstd = {}
  meanstd.mean = {108.64628601/255, 75.86886597/255, 54.34005737/255}
  meanstd.std = {70.53946096/255, 51.71475228/255, 43.03428563/255}
  transforms[6] = T.ColorNormalize(meanstd)

  --pca
  local alphastd = 0.05
  local eigval = torch.Tensor({1.655, 0.485, 0.157})
  local eigvec = torch.Tensor({{-0.56543481, 0.71983482, 0.40240142},
    {-0.5989477, -0.02304967, -0.80036049},
    {-0.56694071, -0.6935729, 0.44423429}})
  transforms[7] = T.Lighting(alphastd, eigval, eigvec)

  local totalTransform = T.Compose(transforms)
  return totalTransform
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
    input[i] = self.transformer(tmp_input)
  end
  return input,labels
end
