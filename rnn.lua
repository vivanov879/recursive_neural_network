require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'project_utils'
nngraph.setDebug(true)

trees_train, trees_dev, trees_test, max_num_nodes, inv_wordMap, wordMap = unpack(torch.load('trees.t7'))
