--translation of tree.py to lua

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
  local tokens = {}
  for _, toks in pairs(treeString) do
    l = string2table(toks)
    for _, c in pairs(l) do 
      tokens[#tokens + 1] = c
    end

    
    
    dummy_pass = 1
  end
  return parse_tokens(tokens)
end

function create_node(label, word = '')
  local node = {}
  node['label'] = label
  node['word'] = word
  return node
  
end

function parse_tokens(tokens, parent = nil)
  assert(tokens[1] == openChar)
  assert(tokens[#tokens] == closeChar)
  local split = 3
  countOpen = 0
  countClose = 0
  
  if (tokens[split] == openChar) then
    countOpen = countOpen + 1
    split = 1
  end
  
  while countOpen ~= countClose do 
    if tokens[split] == openChar then 
      countOpen = countOpen + 1
    end
    if tokens[split] == closeChar then 
      countClose = countClose+ 1
    end
    split = split + 1
  end
  
  local node = create_node(tonumber(tokens[2]))
  node['parent'] = parent
  
  if countOpen == 0 then
    node['word'] = 
      
    
  end
  
  
end



create_tree(treeStrings[1])