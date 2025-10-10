import pickle
import numpy as np
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# ========= 加载 vocab 和 merges =========
with open('save/tokenizer_vocab_train.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('save/tokenizer_merges_train.pkl', 'rb') as f:
    merges = pickle.load(f)

special_tokens = ["<|endoftext|>"]

print(f"加载 vocab({len(vocab)}) 和 merges({len(merges)}) 完成")

# ========= 构建 HuggingFace BPE Tokenizer =========
# vocab: {id: bytes}，而 HF 需要 {token_str: id}
# 因此需要进行一次反转和解码
vocab_str = {v.decode("latin-1"): k for k, v in vocab.items()}
"""
merges里面有这种没法decode的字符
(b'\xe2', b'\x80')
"""
merges_str = []
print("前几个 merges 示例:", merges[440:442])
step = 0
for merge in merges:
    if isinstance(merge, tuple) and len(merge) == 2:
        step = step + 1
        # 将 bytes 解码为字符串
        try:
            token1 = (merge[0].decode("latin-1"))
            token2 = (merge[1].decode("latin-1"))
        except Exception as e:
            print(f"发生了未知错误：{e}")
            print("step: ")
            print(step)
        merges_str.append((token1, token2))
    else:
        print(f"跳过不支持的 merge 格式: {merge}")

print(f"转换后的 merges 数量: {len(merges_str)}")
print("前几个 merges 示例:", merges_str[440:442])



tokenizer = Tokenizer(models.BPE(vocab=vocab_str, merges=merges_str))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
# tokenizer.enable_truncation(None)

# 保存为 HuggingFace 标准格式（可选）
tokenizer.save("save/my_tokenizer.json")
print("初始化 HuggingFace tokenizer 完成 ✅")

# ========= 编码函数 =========
def encode_large_text_file(file_path, output_path, chunk_size=10000):
    """
    流式读取大文件并编码保存为 npy
    """
    all_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        chunk = []
        for i, line in enumerate(f):
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                text_chunk = " ".join(chunk)
                encodings = tokenizer.encode(text_chunk)
                all_ids.extend(encodings.ids)
                chunk = []
                print(f"已处理 {i+1} 行")

        if chunk:
            text_chunk = " ".join(chunk)
            encodings = tokenizer.encode(text_chunk)
            all_ids.extend(encodings.ids)

    arr = np.array(all_ids, dtype=np.int64)
    np.save(output_path, arr)
    print(f"✅ 已保存 {output_path} （{len(all_ids)} tokens）")

# ========= 处理 train 数据 =========
train_path = "data/TinyStoriesV2-GPT4-train.txt"
encode_large_text_file(train_path, "save/encode_ids_train.npy")

# ========= 处理 valid 数据 =========
# valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
# with open(valid_path, "r", encoding="utf-8") as f:
#     text = f.read()
#     encodings = tokenizer.encode(text)
#     ids = np.array(encodings.ids, dtype=np.int64)
#     np.save("save/encode_ids_valid.npy", ids)
#     print(f"✅ valid 数据已保存，共 {len(ids)} tokens")

print("🎉 所有数据处理完成！")
