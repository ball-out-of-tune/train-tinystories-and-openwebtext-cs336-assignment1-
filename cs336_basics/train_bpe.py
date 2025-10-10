# 训练BPE分词器
from cs336_basics.tokenizer import BPETokenizer, BPETokenizerTrainer
import torch
import pickle
import numpy as np
# input_path = "data/TinyStoriesV2-GPT4-valid.txt"
# vocab_size = 10000
special_tokens = ["<|endoftext|>"]
# tokenizer = BPETokenizerTrainer(input_path, vocab_size, special_tokens)
# vocab = tokenizer.get_vocab()
# merges = tokenizer.get_merges()
# print("已经训练好BPE分词器")

# # 保存vocab和merges为.pkl文件
# with open('save/tokenizer_vocab.pkl', 'wb') as f:
#     pickle.dump(vocab, f)

# with open('save/tokenizer_merges.pkl', 'wb') as f:
#     pickle.dump(merges, f)

# print("vocab和merges保存完成")

with open('save/tokenizer_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('save/tokenizer_merges.pkl', 'rb') as f:
    merges = pickle.load(f)
tokenizer = BPETokenizer(vocab, merges, special_tokens)
print("初始化tokenizer完成")
# 加载train_dataset数据
# train_path = "data/TinyStoriesV2-GPT4-train.txt"
# with open(train_path, "r",encoding="utf-8") as f:
    # original_data = f.read()
    # encode_ids = tokenizer.encode(original_data)
    # encode_ids_train = torch.tensor(encode_ids, dtype=torch.long)

# 加载validation_dataset数据
valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
with open(valid_path, "r",encoding="utf-8") as f:
    original_data = f.read()
    print("read文件完成")
    encode_ids = tokenizer.encode(original_data)
    print("encode完成")
    encode_ids_valid = torch.tensor(encode_ids, dtype=torch.long)

# 保存encode_ids为.npy文件
# np.save('encode_ids_train.npy', encode_ids_train.numpy())
np.save('save/encode_ids_valid.npy', encode_ids_valid.numpy())
print("train和valid已保存为.npy文件")

print("数据加载完成")