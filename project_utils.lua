require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)



function strip(str)
  return str:match( "^%s*(.-)%s*$" )
end  
    
  

function read_words(fn)
  fd = io.lines(fn)
  sentences = {}
  line = fd()

  while line do
    sentence = {}
    for _, word in pairs(string.split(line, " ")) do
        sentence[#sentence + 1] = word
    end
    sentences[#sentences + 1] = sentence
    line = fd()
  end
  return sentences
end

function string2table(str)
  local t = {}
  for i = 1, #str do
      t[i] = str:sub(i, i)
  end
  return t
end



function calc_f1(prediction, target)
  local f1_accum = 0
  local precision_accum = 0
  local recall_accum = 0
  for c = 1, 5 do
    local p = torch.eq(prediction, c):double()
    local t = torch.eq(target, c):double()
    local true_positives = torch.mm(t:t(),p)[1][1]
        
    p = torch.eq(prediction, c):double()
    t = torch.ne(target, c):double()
    local false_positives = torch.mm(t:t(),p)[1][1]
    
    p = torch.ne(prediction, c):double()
    t = torch.eq(target, c):double()
    local false_negatives = torch.mm(t:t(),p)[1][1]
    
    local precision = true_positives / (true_positives + false_positives)
    local recall = true_positives / (true_positives + false_negatives)
    
    local f1_score = 2 * precision * recall / (precision + recall)
    f1_accum = f1_accum + f1_score 
    precision_accum = precision_accum + precision
    recall_accum = recall_accum + recall
    
    
  end
  return {f1_accum / 5, precision_accum / 5, recall_accum / 5}
end
