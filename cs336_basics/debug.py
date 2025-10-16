import pickle
from cs336_basics.tokenizer_two import Tokenizer


def debug_space_issue(tokenizer: Tokenizer):
    text_with_space = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
    text_without_space = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light,"
    print("text_with_space_len:")
    print(len(text_with_space))
    print("=== 有空格的情况 ===")
    tokens_with_space = tokenizer.encode(text_with_space)
    print(f"Tokens: {tokens_with_space}")
    print(f"Token数: {len(tokens_with_space)}")
    print(f"最后一个token: {tokens_with_space[-1]}")
    
    print("\n=== 无空格的情况 ===")
    tokens_without_space = tokenizer.encode(text_without_space)
    print(f"Tokens: {tokens_without_space}")
    print(f"Token数: {len(tokens_without_space)}")
    print(f"最后一个token: {tokens_without_space[-1]}")
    
    # 检查差异
    print("\n=== 差异分析 ===")
    if tokens_with_space[-1] != tokens_without_space[-1]:
        print(f"最后一个token不同: {tokens_with_space[-1]} vs {tokens_without_space[-1]}")
        
        # 检查是否超出词汇表
        vocab_size = 10000
        if tokens_with_space[-1] >= vocab_size:
            print(f"❌ 有空格时的最后一个token {tokens_with_space[-1]} 超出词汇表范围 (0-{vocab_size-1})")
        if tokens_without_space[-1] >= vocab_size:
            print(f"❌ 无空格时的最后一个token {tokens_without_space[-1]} 超出词汇表范围 (0-{vocab_size-1})")

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
debug_space_issue(tokenizer=tokenizer)

def detailed_diagnosis():
    vocab_size = 10000
    print(f"模型词汇表大小: {vocab_size}")
    print(f"Token 32 是否有效: {32 < vocab_size}")
    print(f"Token 44 是否有效: {44 < vocab_size}")
    
    # 检查这些token对应的具体内容
    try:
        token_32_text = tokenizer.decode([32])
        print(f"Token 32 对应: '{token_32_text}'")
    except:
        print("无法解码token 32")
    
    try:
        token_44_text = tokenizer.decode([44])
        print(f"Token 44 对应: '{token_44_text}'")
    except:
        print("无法解码token 44")
    
    # 检查空格相关的token
    space_tokens = tokenizer.encode(" ")
    print(f"空格的tokens: {space_tokens}")
    
    # 检查逗号相关的token  
    comma_tokens = tokenizer.encode(",")
    print(f"逗号的tokens: {comma_tokens}")
    
    # 检查逗号加空格
    comma_space_tokens = tokenizer.encode(", ")
    print(f"', '的tokens: {comma_space_tokens}")

detailed_diagnosis()