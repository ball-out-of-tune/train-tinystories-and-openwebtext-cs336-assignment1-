def compute_percentage_eachpart(context_length: int, d_model: int, vocab_size: int, num_layers: int):
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
    print("total flops: " + (str)(total_flops))
    print("X * QKV flops: " + (str)(flops_qkv * num_layers / total_flops))
    print("flops_qkT:" + (str)(flops_qkT * num_layers / total_flops))
    print("flops_attenV: " + (str)(flops_attenV * num_layers / total_flops))
    print("flops_proj" + (str)(flops_proj * num_layers / total_flops))
    print("flops_ffn: " + (str)(flops_ffn * num_layers / total_flops))
    print("flops_final: " + (str)(flops_last / total_flops))
    return total_flops

compute_percentage_eachpart(context_length=1024, num_layers=48, d_model=1600, vocab_size=50257)

"""
answer is:
total flops: 3506703564800
X * QKV flops: 0.2152947079925357
flops_qkT:0.04592953770507428
flops_attenV: 0.04592953770507428
flops_proj0.07176490266417856
flops_ffn: 0.5741192213134285
flops_final: 0.04696209261970862
对于 GPT-2 XL( L = 1024 ) 最多 FLOPs 来自 FFN(前馈层):
FFN 占约 57.4% 的总 FLOPs, 其次是 Q/K/V 的三个投影(约 21.5%);
注意力计算本身(QK^T 与 att·V)合计只占约 9.2%。
原理上因为 FFN 的矩阵乘法规模是 O(L * d的平方 )(随模型宽度d 的平方增长),
这里的d_model = 1600 >> 1024, 且flops_ffn = 4 * context_length * d_model * (4 * d_model)
即ffn的系数是16, 所以在模型里主导计算量
"""

"""
接下来分别计算小,中,大的GPT-2各个部分的FLOPS
"""

#小
print("小")
compute_percentage_eachpart(context_length=1024, num_layers=12, d_model=768, vocab_size=50257)
print("")

#中
print("中")
compute_percentage_eachpart(context_length=1024, num_layers=24, d_model=1024, vocab_size=50257)
print("")

#大
print("大")
compute_percentage_eachpart(context_length=1024, num_layers=36, d_model=1028, vocab_size=50257)
print("")

# GPT-2 XL增加context length
print("increase context length: ")
compute_percentage_eachpart(context_length=16384, num_layers=48, d_model=1600, vocab_size=50257)
print("")
"""
increase context length: 
total flops: 133416668364800
X * QKV flops: 0.09054037751093341
flops_qkT:0.30904448857065275
flops_attenV: 0.30904448857065275
flops_proj0.030180125836977805
flops_ffn: 0.24144100669582244
flops_final: 0.019749512814960853
当上下文长度极大(16k)时, 注意力的 O(L平方 * d) 项迅速成为计算瓶颈——注意力相关的矩阵乘法占比从原先的小比例跃升，
合计超过 60%，而原本主导的 FFN(随d的平方)反而占比下降
"""