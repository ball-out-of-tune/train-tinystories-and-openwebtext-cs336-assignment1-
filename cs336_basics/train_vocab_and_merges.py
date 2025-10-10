# 训练BPE分词器
from cs336_basics.tokenizer import BPETokenizer, BPETokenizerTrainer
import torch
import pickle
import numpy as np
from bpe_v3 import BPE_Trainer
input_path = "data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]
tokenizer = BPE_Trainer()
vocab, merges = tokenizer.train(input_path, vocab_size, special_tokens)
print("已经训练好BPE分词器")

# 保存vocab和merges为.pkl文件
with open('save/tokenizer_vocab_train.pkl', 'wb') as f:
    pickle.dump(vocab, f)

with open('save/tokenizer_merges_train.pkl', 'wb') as f:
    pickle.dump(merges, f)

print("vocab和merges保存完成")