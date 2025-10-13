# 测试代码
import torch

from cs336_basics.my_modules import TransformerLM


def test_transformer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLM(
        vocab_size=1000,
        context_length=128,
        num_layers=2,
        d_model=512,
        num_heads=8,
        d_ff=1024,
        rope_theta=10000.0,
        device=device
    )
    
    # 测试前向传播
    x = torch.randint(0, 1000, (2, 64), device=device)  # (batch_size, seq_len)
    output = model(x)
    print(f"Output shape: {output.shape}")  # 应该是 (2, 64, 1000)

if __name__ == "__main__":
    test_transformer()