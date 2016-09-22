# coding:utf-8
batch = [([3,3],[1,1,1]),([2,2,2],[1])]
"""length = [len(x) for x in batch]
max_len = max(length)
for x in batch:
    x_len = len(x)
    for i in range(max_len - x_len):
        x.append(-1)

print batch
"""
batch.sort(key=lambda x:len(x[0]),reverse=True)
print batch

a = [[1,1],[2,2,2]]
import numpy
b = numpy.array(a)
print type(b)
