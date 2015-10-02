require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'project_utils'
nngraph.setDebug(true)

trees_train, trees_dev, trees_test, wordMap, inv_wordMap = torch.load('trees.t7')


x = torch.rand(2,3)
print(x)





