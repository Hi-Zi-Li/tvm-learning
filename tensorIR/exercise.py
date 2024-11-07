# 2.5 tensor exercise
import IPython
import IPython.display
import numpy as np
import torch
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import tvm.script

import tvm.src


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

## 2.5.1.2  Broadcast
a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)
c = a + b
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm:tvm.nd.NDArray = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
@tvm.script.ir_module
class MyBrocastAdd:
    @T.prim_func
    def broadcast_add(A: T.Buffer((4, 4), "int64"),
                    B: T.Buffer((4), "int64"),
                    C: T.Buffer((4, 4), "int64")):
        T.func_attr({"global_symbol": "broadcast_add", "tir.alias": True})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] + B[vj]

rt_lib = tvm.build(MyBrocastAdd, target="llvm")
rt_lib["broadcast_add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c, rtol=1e-5)

## 2.5.1.3  conv2d
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)
### torch test data init 
data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)

@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(A: T.Buffer((1, 1, 8, 8), "int64"),
             B: T.Buffer((2, 1, 3, 3), "int64"),
             C: T.Buffer((1, 2, 6, 6), "int64")):
        T.func_attr({"global_symbol": "conv", "tir.noalias": True})
        strides = 1
        N, CO, OUT_H, OUT_W, K, CI = 1, 2, 6, 6, 3, 1  # Define these constants or make them parameters
        
        # Loop definition
        for b, k, i, j, di, dj, q in T.grid(N, CO, OUT_H, OUT_W, K, K, CI):
            with T.block("C"):
                # Remap the loop iterators inside the block to use them properly
                vb, vk, vi, vj, vdi, vdj, vq = T.axis.remap("SSSSRRR", [b, k, i, j, di, dj, q])
                
                with T.init():
                    # Initialize the output buffer C
                    C[vb, vk, vi, vj] = T.int64(0)
                
                # Perform the convolution operation
                C[vb, vk, vi, vj] = C[vb, vk, vi, vj] + A[vb, vq, strides*vi+vdi, strides*vj+vdj] * B[vk, vq, vdi, vdj]

data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
tvm.build(MyConv, target="llvm")["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)

## 2.5.2 transform TensorIR
sch = tvm.tir.Schedule(MyAdd)
block = sch.get_block("C", func_name="add")
i, j = sch.get_loops(block)
i0, i1 = sch.split(i, factors=[8, 16])
sch.parallel(i0)
sch.unroll(i1)
sch.vectorize(j)
IPython.display.Code(sch.mod.script(), language="python")

@tvm.script.ir_module
class MyBmmRelu:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"),
                 B: T.Buffer((16, 128, 128), "float32"),
                 C: T.Buffer((16, 128, 128), "float32")):
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for n, i, j, k in T.grid(16, 128, 128, 128):
            with T.block("Y"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                with T.init():
                    Y[vn, vi, vj] = T.float32(0)
                Y[vn, vi, vj] = Y[vn, vi, vj] + A[vn, vi, vk] * B[vn, vk, vj]
        for n, i, j in T.grid(16, 128, 128):
            with T.block("C"):
                vn, vi, vj = T.axis.remap("SSS", [n, i, j])
                C[vn, vi, vj] = T.max(Y[vn, vi, vj], T.int64(0))

@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"), C: T.Buffer((16, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.block("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.block("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.block("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))

sch = tvm.tir.Schedule(MyBmmRelu)
block_Y = sch.get_block("Y", func_name="bmm_relu")
n, i, j, k = sch.get_loops(block_Y)
j0, j1 = sch.split(j, factors=[16, 8])
k0, k1 = sch.split(k, factors=[32, 4])
sch.reorder(j0, k0, k1, j1)
sch.parallel(n)
sch.unroll(k1)

block_C = sch.get_block("C", func_name="bmm_relu")
sch.reverse_compute_at(block_C, j0)
block_Y_init = sch.decompose_reduction(block_Y, k0)
_, _, _, init_j1 = sch.get_loops(block_Y_init)
sch.vectorize(init_j1)

_, _, _, c_j1 = sch.get_loops(block_C)
sch.vectorize(c_j1)

tvm.ir.assert_structural_equal(sch.mod, TargetModule)

before_rt_lib = tvm.build(MyBmmRelu, target="llvm")
after_rt_lib = tvm.build(sch.mod, target="llvm")
a_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))

f_timer = after_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))

