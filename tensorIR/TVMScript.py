# tensor ir
# https://mlc.ai/zh/chapter_tensor_program/case_study.html
# 
import IPython
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import tvm.script

dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
# a @ b == np.matmul(a, b)
c_mm_relu = np.maximum(a_np @ b_np, 0)

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm(A: T.Buffer((128, 128), "float32"),
           B: T.Buffer((128, 128), "float32"),
           C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    
    @T.prim_func
    def relu(A: T.Buffer((128, 128), "float32"),
             B: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "relu", "tir.noalias": True})
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], T.float32(0))
    
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"), 
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), "float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

sch = tvm.tir.Schedule(MyModule)
block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)
j0, j1 = sch.split(j, factors=[32, 4])
# sch.reorder(j0, k, j1)
block_C = sch.get_block("C", func_name="mm_relu")
sch.reverse_compute_at(block_C, j0)

rt_lib = tvm.build(MyModule, target="llvm")
a_nd = tvm.nd.array(a_np)
b_nd:tvm.nd.NDArray = tvm.nd.array(b_np)
c_nd:tvm.nd.NDArray = tvm.nd.empty((128, 128), dtype="float32")

# loop transformed before
func_mm_relu = rt_lib["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

# loop transformed after
rt_lib_after = tvm.build(sch.mod, target="llvm")
func_mm_relu_after = rt_lib_after["mm_relu"]
func_mm_relu_after(a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

# runtime compare
f_timer_before = rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of MyModule %g s"%f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed sch.mod %g s"%f_timer_after(a_nd, b_nd, c_nd).mean)

def transform(mod, jfactor):
    sch = tvm.tir.Schedule(mod)
    block_Y = sch.get_block("Y", func_name="mm_relu")
    i, j, k = sch.get_loops(block_Y)
    j0, j1 = sch.split(j, factors=[None, jfactor])
    sch.reorder(j0, k, j1)
    block_C = sch.get_block("C", "mm_relu")
    sch.reverse_compute_at(block_C, j0)
    return sch.mod

for jfactor in (1, 2, 4, 8, 16, 32):
    mod_transformed = transform(MyModule, jfactor=jfactor)
    rt_lib_transformed = tvm.build(mod_transformed, target="llvm")
    f_timer_transformed = rt_lib_transformed.time_evaluator("mm_relu", tvm.cpu())
    print(f_timer_transformed(a_nd, b_nd, c_nd).mean)
