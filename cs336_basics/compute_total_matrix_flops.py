"""
注: softmax、缩放、非线性、LayerNorm 的标量/逐元素运算被视为次要并未计入矩阵乘法 FLOPs 总和
"""
def compute_total_matrix_flops(context_length: int, d_model: int, vocab_size: int, num_layers: int):
    """
    1
    X分别乘以Q K V
    """
    flops_qkv = 6 * context_length * d_model * d_model

    """
    2
    按 head 计算的点积注意力得分：
      每 head FLOPs = 2 ⋅ 𝐿 ⋅ 𝐿 ⋅ 𝑑_𝑘 =2⋅L⋅L⋅d_k ​ 。因为 ∑ ℎ 𝑑 𝑘 = 𝑑 ∑ h ​ d k ​ =d，所以合并为:FLOPsQKT​ = 2 * L * L * d.
    """
    flops_qkT = 2 * context_length * context_length * d_model

    """
    3
    注意力权重乘 V： ( 𝐿 × 𝐿 ) ⋅ ( 𝐿 × 𝑑_𝑘 )（每 head），合并所有 head：
    """
    flops_attenV = 2 * context_length * context_length * d_model

    """
    4
    注意力输出的线性投影（concat heads 后 𝐿 × 𝑑 L×d 乘 𝑑 × 𝑑 ）
    """
    flops_proj = 2 * context_length * d_model * d_model

    """
    5
    前馈网络（FFN）的两个线性层：
    """
    flops_ffn = 4 * context_length * d_model * (4 * d_model)

    """
    6
    最终语言模型头（logits）： 𝐿 × 𝑑 乘 𝑑 × 𝑉（ 𝑉 V 为词表大小）：
    """
    flops_last = 2 * context_length * d_model * vocab_size

    total_flops = num_layers * (flops_qkv + flops_qkT + flops_attenV + flops_proj + flops_ffn) + flops_last
    return total_flops

result = compute_total_matrix_flops(context_length=1024, num_layers=48, d_model=1600, vocab_size=50257)
print("总的FLOPS次数为：")
print(result)

"""
answer is :
总的FLOPS次数为：
3506703564800
"""