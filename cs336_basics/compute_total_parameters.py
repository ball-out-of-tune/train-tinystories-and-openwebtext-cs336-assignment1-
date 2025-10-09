def compute_total_parameters(vocab_size: int, d_model: int, d_ffn: int, num_layers: int):
    """
    输入嵌入：一个权重矩阵 W_embed ∈ R^(V × d_model)，其中 V 是词表大小，d_model 是模型维度。
    参数量：V * d_model
    """
    embedding_params = vocab_size * d_model

    """
    位置编码：
    可学习的位置编码：如果使用可学习的位置编码，那么会有一个矩阵 W_pos ∈ R^(max_seq_len × d_model)，其中 max_seq_len 是模型能处理的最大序列长度。
    参数量：max_seq_len * d_model
    正弦/余弦固定位置编码：如果使用原始的 Transformer 论文中的正弦/余弦编码，则没有可训练参数

    课程中使用的是RoPE， 不是可训练的参数所以不需计算
    """
    positional_params = 0

    """
    A. 多头自注意力层
    查询、键、值投影：对于每个注意力头 h，有三个投影矩阵：
    W_q^h ∈ R^(d_model × d_k)
    W_k^h ∈ R^(d_model × d_k)
    W_v^h ∈ R^(d_model × d_v)
    通常，d_k = d_v = d_model / h，其中 h 是头的数量。这些矩阵将输入投影到每个头独立的子空间。
    多头输出投影：在多个头的输出拼接之后，有一个线性投影矩阵 W_o ∈ R^(d_model × d_model) 将其映射回 d_model 维空间。
    参数量：
    对于 h 个头，Q、K、V 投影的总参数量为：h * (d_model * d_k + d_model * d_k + d_model * d_v) = 3 * h * d_model * (d_model / h) = 3 * d_model^2
    输出投影 W_o 的参数量为：d_model * d_model = d_model^2
    所以，一个MSA子层的总参数量为：4 * d_model^2
    """
    MHA_params = 4 * d_model * d_model


    """
    B. 前馈神经网络
    FFN 通常由两个线性层和一个非线性激活函数组成。其内部维度 d_ff 通常比 d_model 大得多（例如，d_ff = 4 * d_model）。
    第一个线性层：W_1 ∈ R^(d_model × d_ff)，将输入从 d_model 维扩展到 d_ff 维。
    第二个线性层：W_2 ∈ R^(d_ff × d_model)，将输出从 d_ff 维投影回 d_model 维。
    偏置项：通常包含两个偏置向量 b_1 ∈ R^(d_ff) 和 b_2 ∈ R^(d_model)。
    参数量：
    W_1：d_model * d_ff
    W_2：d_ff * d_model
    b_1：d_ff
    b_2：d_model
    所以，一个FFN子层的总参数量为：2 * d_model * d_ff + d_ff + d_model
    如果忽略相对较小的偏置项，约为 2 * d_model * d_ff。
    """
    FFN_params = 2 * d_model * d_ffn

    """
    C. 层归一化
    每个子层（MSA 和 FFN）后面通常都有一个层归一化。每个层归一化包含两个可学习的参数向量：
    增益：γ ∈ R^(d_model)
    偏置：β ∈ R^(d_model)
    参数量：每个层归一化有 2 * d_model 个参数。
    在一个标准的 Transformer 块中，通常有 2 个层归一化（一个在 MSA 后，一个在 FFN 后），所以参数量为 4 * d_model。

    课程中用的RMSNorm，没有偏置，为2 * d_model
    """
    Norm_params = 4 * d_model

    """
    final linear and RMSNorm
    """
    Final_params = d_model + d_model * vocab_size

    result = embedding_params + positional_params + (Norm_params + MHA_params + FFN_params) * num_layers 

    print(result)

"""
GPT2 XL total params
"""
print("所需参数量: ")
compute_total_parameters(vocab_size=50257, d_model=1600, d_ffn= 6400, num_layers=48)

"""
answer is 1555278400, which need
1555278400 * 4 =6221113600 字节 = 5.794 GiB
memory
"""