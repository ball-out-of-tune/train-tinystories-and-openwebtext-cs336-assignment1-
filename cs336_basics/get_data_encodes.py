import pickle
import numpy as np
import torch
import tqdm
import os
import itertools
from cs336_basics.tokenizer_two import Tokenizer
from memory_profiler import profile

def process_data():
    # === 1. 加载 tokenizer ===
    with open('save/TinyStoriesV2-vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    with open('save/TinyStoriesV2-merges.pkl', 'rb') as f:
        merges = pickle.load(f)

    special_tokens = "<|endoftext|>"
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    print("初始化tokenizer完成")
    # === 2. 文件与统计信息 ===
    train_path = "data/TinyStoriesV2-GPT4-train.txt"
    chunk_size = 5000  # 每次读取的行数
    all_encode_ids = []

    # 先统计训练集总行数（方便 tqdm）
    with open(train_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"训练集总行数: {total_lines}")

    # === 3. 先进行一次采样，估算 token 总数（避免二次遍历）===
    # 用前 1 万行估算平均 tokens/行
    sample_lines = 10000
    with open(train_path, "r", encoding="utf-8") as f:
        lines = list(itertools.islice(f, sample_lines))
    sample_text = " ".join(line.strip() for line in lines)
    sample_tokens = len(tokenizer.encode(sample_text))
    avg_tokens_per_line = sample_tokens / sample_lines
    estimated_total_tokens = int(total_lines * avg_tokens_per_line)
    print(f"估算平均每行 {avg_tokens_per_line:.2f} tokens, 预计总 tokens 数: {estimated_total_tokens:,}")   

    # === 4. 使用 memmap 建立一个磁盘映射文件 ===
    save_path = "save/encode_ids_train.dat"
    mmap = np.memmap(save_path, dtype=np.uint16, mode='w+', shape=((int)(estimated_total_tokens * 1.1),))
    write_pos = 0  # 当前写入位置

    # === 5. 流式处理 ===
    with open(train_path, "r", encoding="utf-8") as f:
        progress_bar = tqdm.tqdm(total=total_lines, desc="Encoding train")

        while True:
            lines_to_read = list(itertools.islice(f, chunk_size))
            if not lines_to_read:
                break

            text_chunk = " ".join(line.strip() for line in lines_to_read)
            encode_ids = tokenizer.encode(text_chunk)

            if encode_ids is None or len(encode_ids) == 0:
                progress_bar.update(len(lines_to_read))
                continue

            # 写入 memmap 文件
            end_pos = write_pos + len(encode_ids)
            if end_pos > mmap.shape[0]:
                # 需要扩容
                new_size = mmap.shape[0] * 2
                print(f"扩容 memmap 到 {new_size:,} tokens")
                mmap.flush()
                del mmap
                os.rename(save_path, save_path + ".bak")
                old = np.memmap(save_path + ".bak", dtype=np.uint16, mode='r', shape=(write_pos,))
                mmap = np.memmap(save_path, dtype=np.uint16, mode='w+', shape=(new_size,))
                mmap[:write_pos] = old[:write_pos]
                del old
                os.remove(save_path + ".bak")

            mmap[write_pos:end_pos] = np.array(encode_ids, dtype=np.uint16)
            write_pos = end_pos

            progress_bar.update(len(lines_to_read))

        progress_bar.close()

    # === 6. 截断到真实大小并保存 ===
    mmap.flush()
    del mmap  # 关闭映射

    # 重新打开并截断到实际大小
    mmap_final = np.memmap(save_path, dtype=np.uint16, mode='r+', shape=(estimated_total_tokens,))
    np.save("save/TinyStoriesV2-GPT4-train.npy", mmap_final[:write_pos])
    del mmap_final

    print(f"✅ 处理完成，实际写入 {write_pos:,} 个 tokens。已保存为 encode_ids_train.npy")


    # # 加载validation_dataset数据
    # valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    # with open(valid_path, "r",encoding="utf-8") as f:
    #     original_data = f.read()
    #     print("read valid文件完成")
    #     encode_ids = tokenizer.encode(original_data)
    #     print("encode valid完成")
    #     encode_ids_valid = torch.tensor(encode_ids, dtype=torch.uint16)

    # # 保存encode_ids为.npy文件

    # np.save('save/encode_ids_valid.npy', encode_ids_valid.numpy())
    # print("train和valid已保存为.npy文件")

    # print("数据加载完成")

from pyinstrument import Profiler

def process_data_with_profiler():
    profiler = Profiler()
    profiler.start()
    
    # 你的原始代码
    process_data()
    
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

process_data_with_profiler()