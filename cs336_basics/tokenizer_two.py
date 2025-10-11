import string
import regex
from collections import defaultdict
from typing import Iterable, Iterator, List, Set, Tuple
import torch
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # 由于需要通过merges字典来排序，所以需要一个字典来存储merges的优先级
        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}
        # 将字节转换为token id，避免直接使用vocab字典
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}


    def _get_bpe_merges(self, piece: bytes) -> List[bytes]:
        """
        对于每一个非特殊符号的字节段word，例如"hello" 进行BPE编码，返回一个字节列表
        """
        # 首先将字节段piece转换为单字节列表
        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            # 记录所有合并对
            pairs = set()
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)
            
            if not pairs:
                break # 如果剩下的合并对都不在merges字典中，就表示没有应该合并的合并对了，直接返回

            # 找到最佳合并对
            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])

            # 应用最佳合并对
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts
    def _get_bpe_merges_optimized(self, piece: bytes) -> List[bytes]:
        """优化版BPE合并"""
        if len(piece) <= 1:
            return [piece] if piece else []
        
        # 初始化为单字节列表
        parts = [bytes([b]) for b in piece]
        n = len(parts)
        
        # 使用优先队列来跟踪可合并的对
        import heapq
        # 初始化优先队列：(-priority, index) 使用负值因为heapq是最小堆
        heap = []
        for i in range(n - 1):
            pair = (parts[i], parts[i+1])
            if pair in self.merges_priority_map:
                priority = self.merges_priority_map[pair]
                heapq.heappush(heap, (priority, i))
        
        # 标记哪些位置已经被合并
        merged = [False] * n
        next_ptr = list(range(1, n + 1))  # 下一个未合并元素的位置
        
        while heap and n > 1:
            # 获取最高优先级的合并对
            priority, i = heapq.heappop(heap)
            
            # 检查这个合并对是否仍然有效（没有被前面的合并影响）
            if merged[i] or merged[i+1]:
                continue
                
            # 执行合并
            merged_piece = parts[i] + parts[i+1]
            parts[i] = merged_piece
            merged[i+1] = True
            n -= 1
            
            # 更新指针
            next_ptr[i] = next_ptr[i+1] if i+1 < len(next_ptr) else len(parts)
            
            # 检查新的合并对（左边）
            if i > 0 and not merged[i-1]:
                left_pair = (parts[i-1], parts[i])
                if left_pair in self.merges_priority_map:
                    left_priority = self.merges_priority_map[left_pair]
                    heapq.heappush(heap, (left_priority, i-1))
            
            # 检查新的合并对（右边）
            next_i = next_ptr[i]
            if next_i < len(parts) and not merged[next_i]:
                right_pair = (parts[i], parts[next_i])
                if right_pair in self.merges_priority_map:
                    right_priority = self.merges_priority_map[right_pair]
                    heapq.heappush(heap, (right_priority, i))
        
        # 收集未合并的结果
        result = []
        i = 0
        while i < len(parts):
            if not merged[i]:
                result.append(parts[i])
            i += 1
        
        return result
    
    def encode(self, text: str) -> List[int]:
        if not text:
            return []

        # 创建一个正则表达式模式来分割特殊符号
        # 按照长度降序排序，确保更长的符号（例如"<|eot|><|eot|>") 在更短的符号（例如"<|eot|>")之前被匹配
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pattern = '|'.join(map(regex.escape, sorted_special_tokens))

        if self.special_tokens:
            # 按照特殊符号分割text，保持特殊符号作为分隔符
            chunks = regex.split(f'({special_token_pattern})', text)
        else:
            chunks = [text]

        final_ids = []
        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                # 如果chunk是特殊符号，直接编码
                final_ids.append(self.bytes_to_id[chunk.encode('utf-8')])
            else:
                # 如果chunk是普通文本，使用BPE算法处理
                # 首先，使用PAT正则表达式将chunk分割为"单词"
                for word in regex.findall(PAT, chunk):
                    if not word:
                        continue
                    
                    # 获取word的合并字节片段
                    merged_pieces = self._get_bpe_merges(word.encode('utf-8'))
                    
                    # 将每个片段转换为token id
                    for piece in merged_pieces:
                        final_ids.append(self.bytes_to_id[piece])
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        all_bytes = b''.join(self.vocab[id] for id in ids)
        return all_bytes.decode("utf-8", errors="replace")