require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'project_utils'
nngraph.setDebug(true)




x = nn.Identity()()
h = nn.Linear(2,4)(x)
h = nn.Dropout()(h)
z = nn.Tanh()(h)

m = nn.gModule({x}, {z})


m:training()
x = torch.rand(10,2)

m:evaluate()
z = m:forward(x)
print(x, z)