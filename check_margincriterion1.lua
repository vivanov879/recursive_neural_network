require 'mobdebug'.start()
require 'nn'


function gradUpdate(mlp, x, y, criterion, learningRate)
  mlp:zeroGradParameters()
  n = 100
  for i = 1, n do 
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:backward(x, gradCriterion)
 end
   mlp:updateParameters(learningRate)
end

mlp = nn.Sequential()
mlp:add(nn.Linear(5, 1))

x1 = torch.rand(5)
x2 = torch.rand(5)
criterion=nn.MarginCriterion(1)

for i = 1, 100 do 
 gradUpdate(mlp, x1, torch.Tensor(1):fill(1), criterion, 0.001)
 gradUpdate(mlp, x2, torch.Tensor(1):fill(-1), criterion, 0.001)
end

print(mlp:forward(x1))
print(mlp:forward(x2))

print(criterion:forward(mlp:forward(x1), torch.Tensor(1):fill(1)))
print(criterion:forward(mlp:forward(x2), torch.Tensor(1):fill(-1)))

