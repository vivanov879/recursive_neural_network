Torch implementation of Recursive Neural Network based on cs224d assignment 3.

Tree structure assigned to every sentence is assumed given. Every node in a tree has a label. We try to predict the label.
In Terminal.app run `th tree.lua` to train the model and check perfromance on the dev set. Prints confusion matrix for train and dev sets.

Result I got with 
```
h_dim = 30
batch_size = 30
number_of_iteration = 10000
```
train set:
```
ConfusionMatrix:
[[    1356    4922     340    1360     267]   16.446% 	[class: 1]
 [     219   21002    7292    5494     355]   61.120% 	[class: 2]
 [      23    3257  205016   11241     251]   93.279% 	[class: 3]
 [       7     612    5971   34836    2768]   78.825% 	[class: 4]
 [       3      66     101    4917    6906]]  57.584% 	[class: 5]
 + average row correct: 61.450784802437% 
 + average rowUcol correct (VOC measure): 49.672969281673% 
 + global correct: 84.473071297186%
```

dev set:
```
ConfusionMatrix:
[[     127     574     151     192      26]   11.869% 	[class: 1]
 [      17    2229    1531     765      71]   48.320% 	[class: 2]
 [       6     695   25684    1849      71]   90.740% 	[class: 3]
 [       4     161    1210    3975     431]   68.760% 	[class: 4]
 [       1      28      88     778     783]]  46.663% 	[class: 5]
 + average row correct: 53.270339220762% 
 + average rowUcol correct (VOC measure): 41.442299634218% 
 + global correct: 79.132385938669%
```
