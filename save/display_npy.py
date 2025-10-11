import numpy as np

# 加载 .npy 文件
data = np.load('save/encode_ids_valid.npy') # 将 'your_file.npy' 替换为你的文件路径

# 查看数组内容
print(data)

# 获取数组的维度信息
print("数组形状：", data.shape)

# 获取数组的数据类型
print("数据类型：", data.dtype)