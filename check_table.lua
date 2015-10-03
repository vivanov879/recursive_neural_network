
require 'mobdebug'.start()

l1 = {}
l2 = {}

h = 1
l1['1'] = h

h = 2
l2['1'] = h

print(l1, l2)



l1 = {}
l2 = {}
t = torch.rand(1)
l1[1] = t
l2[1] = t

l1[1]:fill(10)

print(l1, l2)

dummy_pass = 1