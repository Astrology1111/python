import pandas as pd
from collections import defaultdict
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'week3\情绪分析\week3.csv')
# print(df['cus_comment'].isnull())
# print(df['cus_comment'].isnull().sum())

# 数据清洗，删除空值
df = df.dropna()
cus_comment = list(df['cus_comment'])

# 转化为一维的词语列表
words = []
for i in cus_comment:
    words.append(i.split(' '))
words = [item for sublist in words for item in sublist]

# 用于测试的words
comment_0 = '超级 恶心 打包 了 三份 回家 吃 到 第二份 的 时候 直接 吃 出 不 知道 是 什么 跟 锈 铁丝 一样 的 不明 物体 看到 多赞 才 敢 来 吃 的 排队 等 半天 结果 就 等到 这 玩意 没人 吃 了 拉肚子 的'
test = comment_0.split(' ')
# 引入情绪词典
def emotion_analysis(**path):
    r'''
    惰性加载情绪词典

    path = {
        'anger': r'week3\情绪分析\emotion_lexicon\anger.txt',
        'disgust': r'week3\情绪分析\emotion_lexicon\disgust.txt',
        'fear': r'week3\情绪分析\emotion_lexicon\fear.txt',
        'joy': r'week3\情绪分析\emotion_lexicon\joy.txt',
        'sadness': r'week3\情绪分析\emotion_lexicon\sadness.txt'
        }

    return:
    mixed_analysis: 情绪混合分析
    sigle_analysis: 情绪单一分析
    '''

    # 情绪词典
    emotion_dict = {key: set() for key in path.keys()}
    # 读取情绪词典
    for key, filepath in path.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            emotion_dict[key] = set(f.read().splitlines())
    count = {emotion: 0 for emotion in emotion_dict.keys()}

    # 构建反向映射
    word_to_emotions = defaultdict(list)
    for emotion, emotion_words in emotion_dict.items():
        for word in emotion_words:
            word_to_emotions[word].append(emotion)

    # 情绪混合分析
    def mixed_analysis(words1):
        current_count = {emotion: 0 for emotion in emotion_dict.keys()}
        for word in words1:
            if word in word_to_emotions:
                for emotion in word_to_emotions[word]:
                    current_count[emotion] += 1
        numbers = list(current_count.values())
        numbers_sum = sum(numbers)
        if numbers_sum == 0:
            # print('没有情绪')
            return 0
        weights = [i/numbers_sum for i in numbers]
        result = dict(zip(current_count.keys(), weights))
        # print(f'情绪混合分析：{result}')
        return result

    # 情绪单一分析
    def sigle_analysis(words1):
        for word in words1:
            if word in word_to_emotions:
                for emotion in word_to_emotions[word]:
                    count[emotion] += 1
        max_emotion = max(count.items(), key=lambda x: x[1])
        all_max_emotion = [pair for pair in count.items() if pair[1] == max_emotion[1]]
        print(f'情绪单一分析：{all_max_emotion}')

    
    return mixed_analysis,sigle_analysis


def time_analysis(shopID,time,emotion):
    '''
    shopID: 店铺ID
    time: 时间类型，可选项：'hour','day','weekday','month'
    '''
    
    # 适用于resample类型的函数
    def resample_mixed_analysis(dataframe):
        # 数据清洗，删除空值
        dataframe = dataframe.dropna()
        cus_comment = list(dataframe['cus_comment'])

        # 转化为一维的词语列表
        words = []
        for i in cus_comment:
            words.append(i.split(' '))
        words = [item for sublist in words for item in sublist]
        return mixed_analysis(words)

    # 处理数据的时间格式
    df = pd.read_csv(r'week3\情绪分析\week3.csv')
    df = df.dropna()
    df['datatime'] = pd.to_datetime(df['comment_time'])

    # 提取指定id的数据并把index替换为时间
    object_df = df.loc[df['shopID'] == shopID]
    object_df = object_df.set_index('datatime')

    result = object_df.resample(time).apply(resample_mixed_analysis)
    weights = result.apply(lambda x: x[emotion])
    print(weights)

    # # 结果可视化，绘制折线图
    import plotly.express as px

    # 转换为DataFrame并重置索引
    df = weights.reset_index()
    df.columns = ['Date', 'Value']

    fig = px.line(df, x='Date', y='Value', 
                title='Interactive Time Series',
                labels={'Value': 'Value', 'Date': 'Date'},
                template='plotly_white')

    # 自定义悬浮提示
    fig.update_traces(hovertemplate='Date: %{x}<br>Value: %{y:.4f}<extra></extra>')
    fig.write_html(f"week3\\情绪分析\\img\\{shopID} {time} {emotion}.html")
    fig.show()

# 情绪趋势判断函数
def emotion_trend(weights):
    if weights == 0:
        return 0
    # 阈值
    theta_positive = 0.4
    theta_negative = 0.5
    # 差异容忍度
    D = 0.1
    pscore = weights['joy']
    nscore = weights['sadness'] + weights['anger'] + weights['fear'] + weights['disgust']
    if (pscore > theta_positive) and (pscore > nscore + D):
        return 1
    elif (nscore > theta_negative) and (nscore > pscore + D):
        return -1
    else:
        return 0

def stars_to_trend(stars):
    stars_trend = []
    for star in stars:
        if star >= 4:
            stars_trend.append(1)
        elif star <= 2:
            stars_trend.append(-1)
        else:
            stars_trend.append(0)
    return stars_trend

# 情绪分析的结果和评分进行对比
def emotion_compare():
    # 计算各个评论的情绪趋势
    df = pd.read_csv(r'week3\情绪分析\week3.csv')
    df = df.dropna()
    comments0 = list(df['cus_comment'])
    comments1 = [comment.split(' ') for comment in comments0]

    total_weights = []
    for i in range(len(comments1)):
        total_weights.append(mixed_analysis(comments1[i]))

    trends = [emotion_trend(weight) for weight in total_weights]

    stars = list(df['stars'])
    star_trend = stars_to_trend(stars)

    # 创建DataFrame
    df = pd.DataFrame({"sentiment": trends, "rating": star_trend})
    df["consistent"] = (df["sentiment"] == df["rating"]).astype(int)
    consistency_ratio = df["consistent"].mean()

    # 计算冲突比例（评分与情感完全相反）
    conflict_mask = (
        (df["rating"] == 1) & (df["sentiment"] == -1) |  # 好评但情感消极
        (df["rating"] == -1) & (df["sentiment"] == 1)    # 差评但情感积极
    )
    conflict_ratio = conflict_mask.mean()

    # 堆叠柱状图
    rating_labels = {-1: "1-2星", 0: "3星", 1: "4-5星"}
    sentiment_labels = {-1: "消极", 0: "中性", 1: "积极"}

    # 按评分分组统计情感分布
    stack_data = (
        df.groupby(["rating", "sentiment"])
        .size()
        .unstack()
        .rename(index=rating_labels, columns=sentiment_labels)
    )

    # 绘制堆叠柱状图
    colors = ["#FF6666", "#FFFF99", "#99FF99"]  # 红（消极）、黄（中性）、绿（积极）
    ax = stack_data.plot(
        kind="bar", 
        stacked=True, 
        figsize=(10, 6), 
        color=colors,          # 自定义颜色
        edgecolor="black"      # 添加边框增强可读性
    )

    plt.title("各评分下的情感分布", fontsize=14, pad=20)
    plt.xlabel("评分等级", fontsize=12)
    plt.ylabel("评论数量", fontsize=12)
    plt.legend(
        title="情感分类", 
        bbox_to_anchor=(1.05, 1),  # 将图例移到右侧避免遮挡
        loc="upper left"
    )
    plt.xticks(rotation=0)  # 保持横轴标签水平
    plt.tight_layout()
    plt.savefig(r"week3\情绪分析\img\增加前1.png")
    plt.show()

    # 热力图
    # 计算评分与情感的交叉数量
    heatmap_data = pd.crosstab(df["rating"], df["sentiment"]).rename(
        index=rating_labels, columns=sentiment_labels
    )

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "评论数量"}
    )
    plt.title("评分与情感交叉分布")
    plt.xlabel("情感分类")
    plt.ylabel("评分等级")
    plt.savefig(r"week3\情绪分析\img\增加前2.png")
    plt.show()

    # 情感词典缺陷分析
    # 计算关键冲突比例
    def get_conflict_ratio(rating_val, sentiment_val):
        mask = (df["rating"] == rating_val) & (df["sentiment"] == sentiment_val)
        return mask.mean()

    conflict_metrics = {
        "差评中的积极评论": get_conflict_ratio(-1, 1),
        "差评中的中性评论": get_conflict_ratio(-1, 0),
        "好评中的消极评论": get_conflict_ratio(1, -1),
        "好评中的中性评论": get_conflict_ratio(1, 0),
    }

    print("\n冲突比例:")
    for k, v in conflict_metrics.items():
        print(f"{k}: {v:.1%}")

    print(f"\n总一致性比例: {consistency_ratio:.1%}")
    print(f"完全冲突比例: {conflict_ratio:.1%}")

if __name__ == '__main__':

    path = {
        'anger': r'week3\情绪分析\emotion_lexicon\anger.txt',
        'disgust': r'week3\情绪分析\emotion_lexicon\disgust.txt',
        'fear': r'week3\情绪分析\emotion_lexicon\fear.txt',
        'joy': r'week3\情绪分析\emotion_lexicon\joy.txt',
        'sadness': r'week3\情绪分析\emotion_lexicon\sadness.txt'
        }
    mixed_analysis,single_analysis = emotion_analysis(**path)
    mixed_analysis(words)
    # single_analysis(words)
    # time_analysis(518986,'D','joy')
    # emotion_compare()
