import pickle
import pprint

# 以二进制读取模式打开文件
with open('save/TinyStoriesV2-vocab.pkl', 'rb') as f: # 将‘你的文件.pkl’替换为实际路径
    data_vocab = pickle.load(f)

# with open('save/tokenizer_merges.pkl', 'rb') as f: # 将‘你的文件.pkl’替换为实际路径
    # data_merges = pickle.load(f)
print(len(data_vocab))

# 现在，data 变量包含了 .pkl 文件中的全部内容
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(data_vocab)
# pp.pprint(data_merges)


