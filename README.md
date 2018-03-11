# KP
source activate gluon 

jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'

from mxnet import ndarray as nd

import numpy as np

import mxnet.ndarray as nd

import mxnet.autograd as ag

x = nd.array([[1, 2], [3, 4]])

当进行求导的时候，我们需要一个地方来存x的导数，这个可以通过NDArray的方法attach_grad()来要求系统申请对应的空间。//可以简单理解为x是要被求导的变量

x.attach_grad()

下面定义f。默认条件下，MXNet不会自动记录和构建用于求导的计算图，我们需要使用autograd里的record()函数来显式的要求MXNet记录我们需要求导的程序。

with ag.record():

y = x * 2

z = y * x

接下来我们可以通过z.backward()来进行求导。如果z不是一个标量，那么z.backward()等价于nd.sum(z).backward().

z.backward()

输出导数值：

x.grad

def f(a):   

  b = a * 2
  
    while nd.norm(b).asscalar() < 1000:
    
      b = b * 2
    
    if nd.sum(b).asscalar() > 0:
    
      c = b
      
    else:
    
        c = 100 * b
        
    return c

