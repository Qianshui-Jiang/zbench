import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

# 创建一些随机的测试数据
data = np.random.rand(50000).astype(np.float32)

# 创建一个Context来管理我们的设备和命令队列
os
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# 将我们的数据复制到设备上
device_data = cl_array.to_device(queue, data)

# 创建我们的OpenCL程序
program_source = """
__kernel void multiply_by_scalar(__global float *data)
{
    int i = get_global_id(0);
    data[i] = data[i] * 2;
}
"""
program = cl.Program(context, program_source).build()

# 执行程序
program.multiply_by_scalar(queue, data.shape, None, device_data.data)

# 从设备上获取结果
result = device_data.get()

print(result)  # 输出结果