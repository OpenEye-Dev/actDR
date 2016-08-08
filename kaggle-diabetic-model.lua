require 'torch'
require 'nn'

local features = nn.Sequential()

multi = require('util.multi')

features:add(nn.SpatialConvolution(3,32,4,4,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialConvolution(32,32,4,4,0,0,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialMaxPooling(3,3,2,2))

features:add(nn.SpatialConvolution(32,64,4,4,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialConvolution(64,64,4,4,0,0,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialConvolution(64,64,4,4))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialMaxPooling(3,3,2,2))

features:add(nn.SpatialConvolution(64,128,4,4,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialConvolution(128,128,4,4,0,0,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialConvolution(128,128,4,4))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialMaxPooling(3,3,2,2))

features:add(nn.SpatialConvolution(128,256,4,4,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialConvolution(256,256,4,4,0,0,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialConvolution(256,256,4,4))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialMaxPooling(3,3,2,2))

features:add(nn.SpatialConvolution(256,512,4,4,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialConvolution(512,512,4,4,0,0,2,2))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialConvolution(512,512,4,4))
features:add(nn.LeakyReLU(0.1,true))
features:add(nn.SpatialMaxPooling(3,3,2,2))

features:cuda()
features = mutli.makeDataParallel(features,nGPU)

local classifier = nn.Sequential()
classifier:add(nn.View(512*4))
classifier:add(nn.Linear(512*4,4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, nClasses))
classifier:add(nn.LogSoftMax())
classifier:cuda()

local model = nn.Sequential()

model:add(features)
model:add(classifier)

return model
