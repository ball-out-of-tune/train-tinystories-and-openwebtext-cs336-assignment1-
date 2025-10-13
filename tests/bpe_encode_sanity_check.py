# 选择一些样本文本进行编码-解码测试
from cs336_basics.tokenizer_two import Tokenizer
import pickle

# Round-trip Test
supports_chinese = False
sample_texts = [
       # 基础英文
    "Hello world!",
    "The quick brown fox jumps over the lazy dog.",
    
    # 中文测试（如果支持）
    "测试中文" if supports_chinese else "",
    "你好世界！这是一段中文文本。" if supports_chinese else "",
    
    # 特殊字符和标点
    "Hello, world! How are you?",
    "I'm feeling great!",
    "This costs $100.99.",
    "Email: test@example.com",
    
    # 数字和符号
    "1234567890",
    "1 + 2 = 3",
    "Special chars: !@#$%^&*()",
    
    # 空格和格式
    "Text with     multiple   spaces",
    "Line1\nLine2\nLine3",
    "Tab\tseparated",
    
    # 边界情况
    "",  # 空字符串
    " ",  # 只有空格
    "a",  # 单个字符
    "A",  # 单个大写字符
    
    # 重复字符
    "aaaaaaa",
    "!!!",
    
    # 混合内容
    "Hello 123 世界！",
    "Price: ¥100.50" if supports_chinese else "Price: $100.50",
    
    # 长文本
    "This is a longer text that contains multiple sentences. "
    "It should test how the tokenizer handles extended input. "
    "Let's see if everything works correctly!",
    
    # URL和路径
    "Visit https://example.com/path/to/page",
    "C:\\Users\\Documents\\file.txt",
    
    # 表情符号（如果tokenizer支持）
    "Hello 😊 World 🌍",
    
    # 引号
    '"Quoted text" and \'single quotes\'',
    
    # 括号
    "Text (with parentheses) [and brackets] {and braces}",
]

# === 1. 加载 tokenizer ===
with open('save/TinyStoriesV2-vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('save/TinyStoriesV2-merges.pkl', 'rb') as f:
    merges = pickle.load(f)

special_tokens = "<|endoftext|>"
tokenizer = Tokenizer(vocab, merges, special_tokens)

for text in sample_texts:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert text == decoded, f"Round-trip failed: {text} != {decoded}"

print("Round-trip Test Success")

import numpy as np
from cs336_basics.tokenizer_two import Tokenizer
import pickle

# 使用内存映射方式加载，不实际读入内存
encoded_data = np.load('save/TinyStoriesV2-GPT4-train.npy', mmap_mode='r')

# 加载tokenizer
with open('save/TinyStoriesV2-vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
with open('save/TinyStoriesV2-merges.pkl', 'rb') as f:
    merges = pickle.load(f)

special_tokens = "<|endoftext|>"
tokenizer = Tokenizer(vocab, merges, special_tokens)

# 随机抽取多个样本快速检查
import random

def quick_sample_check(num_samples=5, sample_length=50):
    print("Quick sampling check:")
    print("=" * 50)
    
    # 获取文件大小以确定随机范围
    file_size = len(encoded_data)
    print(f"Total tokens in file: {file_size:,}")
    
    for i in range(num_samples):
        # 随机选择起始位置，避免文件末尾
        start_idx = random.randint(0, file_size - sample_length - 1)
        sample_tokens = encoded_data[start_idx:start_idx + sample_length]
        
        # 转换为Python列表（如果使用内存映射）
        if hasattr(sample_tokens, 'tolist'):
            sample_tokens = sample_tokens.tolist()
        
        decoded_text = tokenizer.decode(sample_tokens)
        
        print(f"Sample {i+1} (position {start_idx:,}):")
        print(f"'{decoded_text}'")
        print("-" * 40)

# 快速检查
quick_sample_check()