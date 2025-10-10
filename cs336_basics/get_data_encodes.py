import pickle
import numpy as np
import torch
import tqdm
import itertools
from cs336_basics.tokenizer_two import Tokenizer
from memory_profiler import profile

def process_data():
    with open('save/tokenizer_vocab_train.pkl', 'rb') as f:
        vocab = pickle.load(f)

    with open('save/tokenizer_merges_train.pkl', 'rb') as f:
        merges = pickle.load(f)

    special_tokens = "<|endoftext|>"
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    print("初始化tokenizer完成")
    # 加载train_dataset数据
    train_path = "data/TinyStoriesV2-GPT4-train.txt"
    chunk_size = 5000  # 每次读取的行数
    all_encode_ids = []

    # 先统计训练集总行数（方便 tqdm）
    with open(train_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"训练集总行数: {total_lines}")
    max_lines = total_lines
    with open(train_path, "r", encoding="utf-8") as f:
        processed_lines = 0
        progress_bar = tqdm.tqdm(total=min(max_lines, sum(1 for _ in f)), desc="Processing train")
        f.seek(0)  # 重置文件指针
        
        while True:
           # 计算本次读取的行数
            lines_to_read = min(chunk_size, max_lines - processed_lines)
            chunk_lines = list(itertools.islice(f, lines_to_read))
            
            if not chunk_lines:
                break
                
            chunk = [line.strip() for line in chunk_lines]
            text_chunk = " ".join(chunk)
            encode_ids = tokenizer.encode(text_chunk)
            
            if encode_ids is not None:
                all_encode_ids.extend(encode_ids)
            
            processed_lines += len(chunk_lines)
            progress_bar.update(len(chunk_lines))
            print(f"已处理 {processed_lines} 行")

        progress_bar.close()

    print(all_encode_ids[:100])
    
    encode_ids_train = torch.tensor(all_encode_ids, dtype=torch.uint16)
    print(f"处理完成，总共 {len(encode_ids_train)} 个token")

    np.save('save/encode_ids_valid.npy', encode_ids_train.numpy())
    print("train文件已转化为ids, 被保存为npy形式")

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