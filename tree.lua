--based on tree.py

require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'project_utils'
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
  
  if tokens[split] == openChar then
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

num_nodes = 0
function count_nodes(node, args)
  num_nodes = num_nodes + 1
end

max_num_nodes = 0
for _, tree in pairs(trees) do 
  num_nodes = 0
  leftTraverse(tree['root'], count_nodes, nil)
  max_num_nodes = math.max(max_num_nodes, num_nodes)
end

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

trees_train = gen_trees('train1.txt')
trees_test = gen_trees('test1.txt')
trees_dev = gen_trees('dev1.txt')


h_dim = 10
output_dim = 5


h_left = nn.Identity()()
h_right = nn.Identity()()
h = nn.JoinTable(2)({h_left, h_right})
h = nn.Linear(2 * h_dim, h_dim)(h)
h = nn.ReLU()(h)
m = nn.gModule({h_left, h_right}, {h})

h_raw = nn.Identity()()
y = nn.Linear(h_dim, output_dim)(h_raw)
y = nn.LogSoftMax()(y)
lsf = nn.gModule({h_raw}, {y})

embed = Embedding(#inv_wordMap, h_dim)
criterion = nn.ClassNLLCriterion()

local params, grad_params = model_utils.combine_all_parameters(m, embed, lsf)
params:uniform(-0.08, 0.08)

m_clones = model_utils.clone_many_times(m, 100)
embed_clones = model_utils.clone_many_times(embed, 100)
criterion_clones = model_utils.clone_many_times(criterion, 100)
lsf_clones= model_utils.clone_many_times(lsf, 100)

m_counter = 1
embed_counter = 1
criterion_counter = 1
lsf_counter = 1
function fill_clones(node, args)
  node['criterion'] = criterion_clones[criterion_counter]
  criterion_counter = criterion_counter + 1
  node['lsf'] = lsf_clones[lsf_counter]
  lsf_counter = lsf_counter + 1
  if node['isLeaf'] then
    node['embed'] = embed_clones[embed_counter]
    embed_counter = embed_counter + 1
  else
    node['m'] = m_clones[m_counter]
    m_counter = m_counter + 1
  end
end

for _, tree in pairs(trees) do 
  m_counter = 1
  embed_counter = 1
  criterion_counter = 1
  lsf_counter = 1
  leftTraverse(tree['root'], fill_clones, nil)
end


loss = 0
function forwardProp(node)
  if node['isLeaf'] then 
    x = torch.Tensor(1):fill(node['word'])
    h = node['embed']:forward(x)
    node['x'] = x
    
  else
    h_left = forwardProp(node['left'])
    h_right = forwardProp(node['right'])
    h = node['m']:forward({h_left, h_right})
    node['h_left'] = h_left
    node['h_right'] = h_right
    
  end
  y = node['lsf']:forward(h)
  node['loss'] = node['criterion']:forward(y, torch.Tensor(1):fill(node['label']))
  loss = loss + node['loss']
  node['y'] = y
  node['h'] = h
  
  return h
end

function backProp(node, dh)
  y = node['y']
  h = node['h']
    
  dy = node['criterion']:backward(y, torch.Tensor(1):fill(node['label']))
  dh = node['lsf']:backward(h, dy)
  
  if not node['isLeaf'] then
    h_left = node['h_left']
    h_right= node['h_right']
    dh_left, dh_right = unpack(node['m']:backward({h_left, h_right}, dh))
    backProp(node['left'], dh_left)
    backProp(node['right'], dh_right)
    
  else
    dx = node['embed']:backward(x, dh)
  end

end
  
  
tree = trees[1]
forwardProp(tree['root'])
backProp(tree['root'], torch.zeros(1, h_dim))

data_index = 1
n_data = #trees
function gen_batch()
  if data_index > n_data then
    data_index = 1
  end
  local tree = trees[data_index]
  data_index = data_index + 1
  return tree
end

function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    loss = 0
    local tree = gen_batch()
    
    forwardProp(tree['root'])
    backProp(tree['root'], torch.zeros(1, h_dim))
    
    return loss, grad_params
end
        
    
optim_state = {learningRate = 1e-2}


for i = 1, 1000000 do

  local _, loss = optim.adam(feval, params, optim_state)
  if i % 1 == 0 then
    print(string.format( 'loss = %6.8f, grad_params:norm() = %6.4e, params:norm() = %6.4e', loss[1], grad_params:norm(), params:norm()))
  end
  
  if i % 10 == 0 then
    torch.save('model.t7', m)
  end
  
end




dummy_pass = 1





