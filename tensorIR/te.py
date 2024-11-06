import IPython
import IPython.display
import tvm
from tvm import te, tir

N:int = 256
A: te.Tensor = te.placeholder((N, N), "float32", name="A")
B: te.Tensor = te.placeholder((N, N), "float32", name="B")
k: tir.IterVar = te.reduce_axis((0, N), "k")
Y: te.Tensor = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C: te.Tensor = te.compute((N, N), lambda i, j: te.max(Y[i, j], 0), name="C")

te_func:tir.PrimFunc = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
MyModuleFromTE:tvm.IRModule = tvm.IRModule({"mm_relu": te_func})
IPython.display.Code(MyModuleFromTE.script(), language="python")
breakpoint()

A = te.placeholder((N, N), "float32", name="A")
B = te.placeholder((N, N), "float32", name="B")
k = te.reduce_axis((0, N), name="k")
Y = te.compute((N, N), fcompute=(lambda i, j: te.sum(A[i, k] * B[k, j], axis=k)))
C = te.compute((N, N), fcompute=(lambda i, j: te.max(Y[i, j], 0)), name="C")
te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
MyModuleFromTE = tvm.IRModule({"mm_relu": te_func})
