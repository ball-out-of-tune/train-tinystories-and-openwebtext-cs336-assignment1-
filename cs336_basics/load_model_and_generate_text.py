import torch
import pickle
from cs336_basics.inference import generate_text, generate_text_with_debug
from cs336_basics.my_modules import TransformerLM
from cs336_basics.tokenizer_two import Tokenizer
from pretrained_config import PretrainedConfig
# 定义模型结构（参数要和训练时一致）
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

# 加载 checkpoint 并处理参数名
checkpoint = torch.load("checkpoint/checkpoint_10000.pt", map_location=config.device)
model_state_dict = checkpoint['model_state_dict']

# 去掉 _orig_mod. 前缀
new_state_dict = {}
for key, value in model_state_dict.items():
    if key.startswith('_orig_mod.'):
        new_key = key.replace('_orig_mod.', '')
    else:
        new_key = key
    new_state_dict[new_key] = value

# 加载处理后的状态字典
model.load_state_dict(new_state_dict)

# # 加载中途的checkpoint文件 先加载整个 checkpoint 文件到一个变量中
# checkpoint = torch.load("checkpoints/model_epoch_59_20250706_221431.pth", map_location=device)
# # 然后从 checkpoint 字典中，根据键 'model_state_dict' 取出真正的模型权重
# model_weights = checkpoint['model_state_dict']
# # 最后将取出的模型权重加载到模型中
# model.load_state_dict(model_weights)

# 从 vocab.pkl 加载词汇表
with open("save/TinyStoriesV2-vocab.pkl", "rb") as f:
    # pickle.load 会自动恢复字典，并且值是 bytes 类型
    vocab = pickle.load(f)

# 从 merges.pkl 加载合并规则
with open("save/TinyStoriesV2-merges.pkl", "rb") as f:
    # pickle.load 会自动恢复列表，并且元组里的元素是 bytes 类型
    merges = pickle.load(f)
special_tokens = ["<|endoftext|>"]  

tokenizer = Tokenizer(vocab, merges, special_tokens)
# input_text = "Once upon a time, there was a pretty girl named Lily.One day, Lily's mom asked her to help cook dinner."
# input_text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
input_text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of darkness, it was the spring of hope, it was the winter of despair."
# input_text = "baby shark"
# input_ids = tokenizer.encode(input_text)
# input_ids = torch.tensor(input_ids, dtype=torch.long).to(config.device)

print("begin the inference")
# 推理
output_text = generate_text(model=model, tokenizer=tokenizer, prompt=input_text, max_length=256,temperature=1)
print("end the inference")


print(output_text)