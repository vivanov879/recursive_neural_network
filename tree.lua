require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'project_utils'
require 'table_utils'
nngraph.setDebug(true)


treeStrings = read_words('train1.txt')


openChar = '('
closeChar = ')'

function create_tree(treeString)
  tokens = {}
  for _, toks in pairs(treeString) do
    
    dummy_pass = 1
  end
  
  
  
end

create_tree(treeStrings[1])