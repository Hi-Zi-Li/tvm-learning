# 2.5 tensor exercise
import IPython
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import tvm.script


## 2.5.1.1  MyAdd
@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def add(A: T.Buffer((128, 128), "int64"),
            B: T.Buffer((128, 128), "int64"),
            C: T.Buffer((128, 128), "int64")):
        T.func_attr({"global_symbol": "add"})
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = A[vi, vj] + B[vi, vj]

a = np.arange(16384, dtype="int64").reshape(128, 128)
b = np.arange(16384, 0, -1, dtype="int64").reshape(128, 128)

rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((128, 128), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), np.add(a, b), rtol=1e-5)

## 2.5.1.2 Broadcast


