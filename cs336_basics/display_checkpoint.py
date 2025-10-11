import torch

from cs336_basics.my_modules import TransformerLM
from cs336_basics.pretrained_config import PretrainedConfig
config = PretrainedConfig(
        project_name="generate_text",
        vocab_path="save/TinyStoriesV2-vocab.pkl",
        merges_path="save/TinyStoriesV2-merges.pkl",
        special_tokens=["<|endoftext|>"],
        train_path="save/TinyStoriesV2-GPT4-train.npy",
        valid_path="save/TinyStoriesV2-GPT4-valid.npy",
        checkpoint_dir="checkpoint"
    )
model = TransformerLM(d_model=config.d_model, num_heads=config.num_heads, d_ff=config.d_ff, context_length=config.context_length, 
                      rope_theta=config.rope_theta, num_layers=config.num_layers, vocab_size=config.vocab_size).to(config.device)
checkpoint = torch.load("checkpoint/checkpoint_40000.pt", map_location=config.device)
model_state_dict = checkpoint['model_state_dict']

print("=== Model keys ===")
for name, param in model.named_parameters():
    print(name)

print("\n=== Checkpoint keys ===")
for key in model_state_dict.keys():
    print(key)

# 然后手动检查哪些键不匹配