import torch

import torch
import torch.nn.functional as F
import numpy as np

from cs336_basics.tokenizer_two import Tokenizer

def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Top-p (nucleus) 采样
    
    Args:
        logits: 模型输出的logits张量 [vocab_size]
        p: top-p阈值 (0-1之间)
        temperature: 温度参数，控制随机性
    Returns:
        selected_token: 采样得到的token索引
    """
    # 应用温度参数
    logits = logits / temperature
    
    # 转换为概率
    probs = F.softmax(logits, dim=-1)
    
    # 对概率排序（降序）
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过p的第一个位置
    # 我们要移除累积概率超过p的token，所以找到第一个超过p的位置
    sorted_indices_to_remove = cumulative_probs > p
    
    # 确保至少保留一个token（如果所有token概率都很小）
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 获取要移除的token索引
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    
    # 将这些token的概率设为很小的值
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = -float('inf')
    
    # 重新计算概率分布
    filtered_probs = F.softmax(filtered_logits, dim=-1)
    
    # 从过滤后的分布中采样
    selected_token = torch.multinomial(filtered_probs, num_samples=1)
    
    return selected_token.item()

def top_p_sampling_batch(logits, p=0.9, temperature=1.0):
    """
    批量处理的Top-p采样
    
    Args:
        logits: [batch_size, vocab_size]
        p: top-p阈值
        temperature: 温度参数
    Returns:
        selected_tokens: [batch_size] 采样得到的token
    """
    batch_size, vocab_size = logits.shape
    
    # 应用温度参数
    logits = logits / temperature
    
    # 转换为概率
    probs = F.softmax(logits, dim=-1)
    
    # 对每个样本独立处理
    selected_tokens = []
    for i in range(batch_size):
        # 对单个样本的概率排序
        sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到要移除的token
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        
        # 过滤概率
        filtered_probs = probs[i].clone()
        filtered_probs[indices_to_remove] = 0
        
        # 重新归一化
        if filtered_probs.sum() > 0:
            filtered_probs = filtered_probs / filtered_probs.sum()
        else:
            # 如果所有概率都被过滤，回退到原始分布
            filtered_probs = probs[i]
        
        # 采样
        selected_token = torch.multinomial(filtered_probs.unsqueeze(0), num_samples=1)
        selected_tokens.append(selected_token.item())
    
    return torch.tensor(selected_tokens)

# # 测试数据 - 创建一个简单的logits张量
# torch.manual_seed(42)  # 设置随机种子以便复现结果
# vocab_size = 10
# test_logits = torch.randn(vocab_size) * 2  # 随机生成logits

# print("测试logits:", test_logits)
# print("原始概率分布:", F.softmax(test_logits, dim=-1))

# # 测试top-p采样
# selected_token = top_p_sampling(test_logits, p=0.9, temperature=1.0)
# print(f"\n采样结果: token_id = {selected_token}")

# # 多次采样观察分布
# print("\n多次采样结果:")
# for i in range(10):
#     token_id = top_p_sampling(test_logits, p=0.9, temperature=1.0)
#     print(f"第{i+1}次采样: {token_id}")

# # 可视化top-p过滤效果
# print("\n=== Top-p过滤效果分析 ===")
# probs = F.softmax(test_logits, dim=-1)
# sorted_probs, sorted_indices = torch.sort(probs, descending=True)
# cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

# print("排序后的概率:", sorted_probs)
# print("累积概率:", cumulative_probs)
# print("保留的token索引:", sorted_indices[cumulative_probs <= 0.9])

def generate_text(model, tokenizer: Tokenizer, prompt: str, max_length: int=50, p: float=0.9, 
                  temperature: float=1.0, device="cuda" if torch.cuda.is_available() else "cpu", special_tokens = "<endoftext>"):
    """
    文本生成函数
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        prompt: 输入文本
        max_length: 生成的最大长度
        p: top-p阈值
        temperature: 温度参数
        device: 设备
    Returns:
        generated_text: 生成的文本
    """
    model.eval()
    
    # 编码输入文本
    input_ids = torch.tensor(tokenizer.encode(prompt)).to(device)
    prompt_length = len(input_ids)
    if input_ids.dim() == 1:  # 如果是一维的 [seq_len]
        input_ids = input_ids.unsqueeze(0)  # 变成 [1, seq_len]
    generated = input_ids.clone()
    
    print(f"开始生成，初始输入: '{prompt}'")
    print(f"输入token数: {len(input_ids[0])}")
    
    with torch.no_grad():
        for step in range(max_length):
            # 获取模型输出
            outputs = model(input_ids)
            logits = outputs[:, -1, :]  # 取最后一个时间步的logits
            
            # 使用top-p采样选择下一个token
            next_token_id = top_p_sampling(logits[0], p=p, temperature=temperature)
            
            # 将新token添加到序列中
            next_token = torch.tensor([[next_token_id]]).to(device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated = torch.cat([generated, next_token], dim=-1)
            
            # 打印当前生成的token（可选）
            # print(f"步骤 {step+1}: token_id={next_token_id}, text='{tokenizer.decode(next_token_id)}'")
            
            # 如果遇到结束符，提前停止
            special_token_ids = [tokenizer.encode(special_token) for special_token in special_tokens]
            if next_token_id in special_token_ids:
                break
                
            # 限制输入长度（滑动窗口，避免过长）
            if input_ids.shape[-1] > 512:
                input_ids = input_ids[:, -512:]
    
    # 解码生成结果
    new_tokens = generated[0][prompt_length:]  # 去掉prompt对应的token
    # 将CUDA tensor转换为CPU上的普通Python列表
    new_tokens_list = new_tokens.cpu().tolist()
    generated_text = tokenizer.decode(new_tokens_list)
    return generated_text

# 更通用的版本，支持不同的停止条件
def generate_autoregressive(model, tokenizer, prompt, max_length=50, p=0.9, temperature=1.0, 
                          stop_tokens=None, device='cpu'):
    """
    自回归生成函数
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        prompt: 输入文本
        max_length: 生成的最大长度
        p: top-p阈值
        temperature: 温度参数
        stop_tokens: 停止token列表
        device: 设备
    Returns:
        generated_text: 生成的文本
        generated_ids: 生成的token IDs
    """
    model.eval()
    
    if stop_tokens is None:
        stop_tokens = [tokenizer.eos_token_id]
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_ids = input_ids[0].tolist()
    
    print(f"生成开始: '{prompt}'")
    
    with torch.no_grad():
        for step in range(max_length):
            # 获取当前输入（只保留最近的部分以避免过长）
            current_input = input_ids[:, -min(512, input_ids.shape[1]):]
            
            # 模型前向传播
            outputs = model(current_input)
            logits = outputs.logits[:, -1, :]
            
            # top-p采样
            next_token_id = top_p_sampling(logits[0], p=p, temperature=temperature)
            
            # 添加到生成序列
            generated_ids.append(next_token_id)
            
            # 更新输入（用于下一个时间步）
            next_token = torch.tensor([[next_token_id]]).to(device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # 检查停止条件
            if next_token_id in stop_tokens:
                break
    
    # 解码
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids

# 流式生成版本（实时输出）
def generate_streaming(model, tokenizer, prompt, max_length=50, p=0.9, temperature=1.0, device='cpu'):
    """
    流式生成，实时输出结果
    """
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_text = prompt
    
    print(f"输入: {prompt}")
    print("生成: ", end="", flush=True)
    
    with torch.no_grad():
        for step in range(max_length):
            current_input = input_ids[:, -512:]  # 滑动窗口
            
            outputs = model(current_input)
            logits = outputs.logits[:, -1, :]
            
            next_token_id = top_p_sampling(logits[0], p=p, temperature=temperature)
            
            # 解码并打印当前token
            next_token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(next_token_text, end="", flush=True)
            
            generated_text += next_token_text
            
            # 更新输入
            next_token = torch.tensor([[next_token_id]]).to(device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token_id == tokenizer.eos_token_id:
                break
    
    print()  # 换行
    return generated_text

# 使用示例（伪代码）
def example_usage():
    """
    使用示例 - 需要实际的model和tokenizer
    """
    # 假设我们有model和tokenizer
    # from transformers import AutoModel, AutoTokenizer
    # model = AutoModel.from_pretrained("your-model")
    # tokenizer = AutoTokenizer.from_pretrained("your-model")
    
    prompt = "今天天气很好，"
    
    # 基本生成
    # result = generate_text(model, tokenizer, prompt, max_length=50, p=0.9, temperature=0.8)
    
    # 流式生成
    # result = generate_streaming(model, tokenizer, prompt, max_length=50, p=0.9, temperature=0.8)
    
    # 带停止词的生成
    # stop_tokens = [tokenizer.eos_token_id, tokenizer.encode("。")[0]]
    # result, ids = generate_autoregressive(model, tokenizer, prompt, stop_tokens=stop_tokens)
    
    print("生成完成")

# 测试用的简化版本（不需要真实模型）
def test_generation():
    """
    测试生成函数 - 使用模拟的logits
    """
    class MockModel:
        def __init__(self, vocab_size=1000):
            self.vocab_size = vocab_size
            
        def __call__(self, input_ids):
            # 模拟模型输出 - 随机logits
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
            return type('Output', (), {'logits': logits})()
    
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            self.eos_token_id = 999
            
        def encode(self, text, return_tensors=None):
            # 模拟编码 - 返回随机token IDs
            ids = [1, 2, 3, 4, 5]  # 模拟输入序列
            if return_tensors == 'pt':
                return torch.tensor([ids])
            return ids
            
        def decode(self, ids, skip_special_tokens=True):
            # 模拟解码
            return f"[模拟文本: {len(ids)}个token]"
    
    # 测试
    model = MockModel()
    tokenizer = MockTokenizer()
    
    prompt = "测试输入"
    result = generate_text(model, tokenizer, prompt, max_length=10, p=0.9, temperature=1.0)
    print(f"测试生成结果: {result}")

if __name__ == "__main__":
    test_generation()