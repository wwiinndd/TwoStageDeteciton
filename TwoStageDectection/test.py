import jittor as jt
import jittor.misc as misc
import inspect

if jt.has_cuda:
    jt.flags.use_cuda = 1

a = jt.array([False, False, True]).any()
print(a)
