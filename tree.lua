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


treeStrings = read_words('train.txt')


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
  local tree = {}
  tree['root'] = parse_tokens(tokens, nil)
  return tree
end

function create_node(label, word)
  local node = {}
  node['label'] = label + 1
  node['word'] = word
  node['isLeaf'] = false
  return node
  
end

function parse_tokens(tokens, parent)
  assert(tokens[1] == openChar)
  assert(tokens[#tokens] == closeChar)
  local split = 3
  countOpen = 0
  countClose = 0
  
  if (tokens[split] == openChar) then
    countOpen = countOpen + 1
    split = split + 1
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
  
  local node = create_node(tonumber(tokens[2]), nil)
  node['parent'] = parent
  
  if countOpen == 0 then
    node['word'] = table.concat(tokens, ''):lower():sub(3, #tokens - 1)
    node['isLeaf'] = true
    return node
  end
  
  
  local tokens_left = {} 
  local tokens_right = {} 
  for i, token in pairs(tokens) do 
    if i >= split and i < #tokens then
      tokens_right[#tokens_right + 1] = token
    end
    if i > 2 and i < split then
      tokens_left[#tokens_left + 1] = token
    end
  end
  
  node['left'] = parse_tokens(tokens_left, node)
  node['right'] = parse_tokens(tokens_right, node)
  return node
  
end

function leftTraverse(root, nodeFn, args)
  nodeFn(root, args)
  if root['left'] ~= nil then
    leftTraverse(root['left'], nodeFn, args)
  end
  if root['right'] ~= nil then
    leftTraverse(root['right'], nodeFn, args)
  end
end
  
local trees = {}
for i, treeString in pairs(treeStrings) do 
  local tree = create_tree(treeStrings[i])
  trees[#trees + 1] = tree
end

function countWords(node, words)
  if node['isLeaf'] then
    words[node['word']] = words[node['word']] or 0
    words[node['word']] = words[node['word']] + 1
  end
end

inv_wordMap = {}
wordMap = {}
words = {}

function buildWordMap()
  
  for _, tree in pairs(trees) do 
    leftTraverse(tree['root'], countWords, words)
  end
  
  for word, _ in pairs(words) do 
    inv_wordMap[#inv_wordMap + 1] = word
  end
  inv_wordMap[#inv_wordMap + 1] = 'UNK'
  
  for i, word in pairs(inv_wordMap) do 
    wordMap[word] = i
  end
  
end

buildWordMap()

function mapWords(node, wordMap)
  node['word'] = wordMap[node['word']] or wordMap['UNK']
end

dummy_pass = 1

for _, tree in pairs(trees) do 
  leftTraverse(tree['root'], mapWords, words)
end


dummy_pass = 1


--now save test, dev, and train trees using wordMap created with train tree

function gen_trees(fn)
  local treeStrings = read_words(fn)
  local trees = {}
  for i, treeString in pairs(treeStrings) do 
    local tree = create_tree(treeStrings[i])
    trees[#trees + 1] = tree
  end
  for _, tree in pairs(trees) do 
    leftTraverse(tree['root'], mapWords, words)
  end
  return trees
end

trees_train = gen_trees('train.txt')
trees_test = gen_trees('test.txt')
trees_dev = gen_trees('dev.txt')

torch.save('trees.t7', {trees_train, trees_dev, trees_test})



