import jieba
import wordcloud
# 读取和分词
with open(r'week2\词频\week2.txt', 'r', encoding='utf-8') as f:
    file = f.read()
    # file_lis = f.readlines()
    # for i in range(10):
    #     print(file_lis[i])

# 分词
words = jieba.lcut(file)

# 统计词频
counts = {}

for word in words:
    if len(word) == 1:
        continue
    else:
        counts[word] = counts.get(word, 0) + 1

# print(*counts.items(),sep='\n')

# 转化为列表进行后续处理
counts_lis = counts.items()
counts_lis = sorted(counts_lis, key=lambda x: x[1], reverse=True)

# 输出前10个高频词
def show_top10():
    for i in range(10):
        word, count = counts_lis[i][0], counts_lis[i][1]
        print(f'{word} {count}')

# 处理为词云可以处理的格式
words_show_root = [word for word in words if len(word) > 1]
words_show = ' '.join(words_show_root)

def wordcloud_show():
    # 生成词云
    wc = wordcloud.WordCloud(font_path='msyh.ttc', width=800, height=600, background_color='white')
    wc.generate(words_show)
    wc.to_file(r'week2\词频\img\wordcloud.png')


def wordcloud_show_stop():
    # 引入停用词表
    with open(r'week2\词频\cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()

    words_show_stop = ' '.join([word for word in words if word not in stopwords])

    # 再次生成词云
    wc = wordcloud.WordCloud(font_path='msyh.ttc', width=800, height=600, background_color='white')
    wc.generate(words_show_stop)
    wc.to_file(r'week2\词频\img\wordcloud_stop.png')

# 词性分析
def posseg_show():
    import jieba.posseg as pseg
    words = pseg.cut(''.join(words_show_root))
    posseg_count = {}

    # # 统计词性频率
    # for _, flag in words:
    #     posseg_count[flag] = posseg_count.get(flag, 0) + 1
    # print(*posseg_count.items(),sep='\n')

    # 对名词词性进行词云可视化
    words_show_posseg = ' '.join([word for word, flag in words if flag[0] == 'n'])
    wc = wordcloud.WordCloud(font_path='msyh.ttc', width=800, height=600, background_color='white')
    wc.generate(words_show_posseg)
    wc.to_file(r'week2\词频\img\wordcloud_n.png')

# 利用词频筛选特征词

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.special import expit

def feature_words_simplified():
    # 加载停用词表
    with open(r'week2\词频\cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
    
    # 在读取语料时记录原始行号
    original_line_numbers = []
    corpus = []
    with open(r'week2\词频\week2.txt', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):  # 行号从1开始
            text = line.strip()
            if len(jieba.lcut(text)) >= 3:
                corpus.append(text)
                original_line_numbers.append(line_num)  # 存储原始行号


    
    # 优化后的分词函数
    def tokenize(text):
        return [word for word in jieba.lcut(text) if word not in stopwords and len(word) > 1]
    
    # TF-IDF向量化（网页5推荐参数）
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        max_features=800,  # 特征维度优化
        sublinear_tf=True,
        norm='l2',
        min_df=3  # 忽略低频词
    )
    
    # 构建特征矩阵
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    # 混合相似度计算
    def hybrid_similarity(vec1, vec2, text1, text2):
        # 余弦相似度
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Jaccard相似度
        set1 = set(tokenize(text1))
        set2 = set(tokenize(text2))
        jaccard_sim = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
        
        # 动态权重调整
        weight = min(len(set1), len(set2)) / 8  # 调整权重系数
        return (0.75 + weight*0.15)*cosine_sim + (0.25 - weight*0.15)*jaccard_sim
    
    # 近似最近邻搜索
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine')
    nbrs.fit(tfidf_matrix)
    
    # 计算相似度
    distances, indices = nbrs.kneighbors(tfidf_matrix)
    
    # 提取并校准top10
    top_pairs = []
    for i, neighbors in enumerate(indices):
        for j, idx in enumerate(neighbors):
            if i != idx:
                raw_sim = 1 - distances[i][j]
                # 优化后的Sigmoid校准参数
                calibrated_sim = 1 / (1 + np.exp(-12*(raw_sim-0.65)))
                top_pairs.append( (calibrated_sim, i, idx) )
    
    # 去重排序
    top_pairs = sorted(list(set(top_pairs)), reverse=True)[:10]
    


    # 修改结果输出部分
    for sim, i, j in top_pairs:
        # 获取原始行号和文本内容
        orig_i = original_line_numbers[i]
        orig_j = original_line_numbers[j]
        text_i = corpus[i]
        text_j = corpus[j]
        
        # 格式化输出
        print(f"文本{orig_i}: {text_i}")
        print(f"与文本{orig_j}: {text_j}")
        print(f"校准相似度: {sim:.4f}\n{'-'*50}")  # 分隔线
if __name__ == '__main__':
    feature_words_simplified()

