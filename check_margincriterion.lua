require 'mobdebug'.start()
require 'nn'


function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

mlp = nn.Sequential()
mlp:add(nn.Linear(5, 1))

x1 = torch.rand(5)
x2 = torch.rand(5)
criterion=nn.MarginCriterion(1)

for i = 1, 1000 do
   gradUpdate(mlp, x1, torch.Tensor(1):fill(1), criterion, 0.01)
   gradUpdate(mlp, x2, torch.Tensor(1):fill(-1), criterion, 0.01)
end

print(mlp:forward(x1))
print(mlp:forward(x2))

print(criterion:forward(mlp:forward(x1), torch.Tensor(1):fill(1)))
print(criterion:forward(mlp:forward(x2), torch.Tensor(1):fill(-1)))

