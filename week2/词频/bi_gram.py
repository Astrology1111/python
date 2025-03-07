import nltk
import jieba
import re 
import matplotlib.pyplot as plt

with open(r'week2\词频\week2.txt', 'r', encoding='utf-8') as f:
    file = f.readlines()

bi_gram = []
# 按行进行分词和二元组提取
for txt in file:
    txt = re.sub(r'[^\w\s]', '', txt[:-1])
    txt = re.sub(r' ', '', txt)
    tokens = jieba.lcut(txt)
    bi_gram.extend(list(nltk.bigrams(tokens)))

# 统计二元组出现的频率
bi_gram_freq = nltk.FreqDist(bi_gram)
print(bi_gram_freq.most_common())

top_10 = bi_gram_freq.most_common(10)

# 分解二元组和频率
bigrams, frequencies = zip(*top_10)
x_list = [i for i in range(10)]
# 绘制条形图
plt.figure(figsize=(10, 6))
plt.bar(x_list, frequencies, color='skyblue')
plt.xlabel('Bigrams')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Bigrams')
plt.xticks(x_list,bigrams, rotation=45,fontsize=7)
plt.savefig(r'week2\词频\img\bi_gram.png')
plt.show()