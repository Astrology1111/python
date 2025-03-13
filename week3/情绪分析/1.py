import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
# words = ['发火','自以为是','真吓人','玩笑','愁']
# 引入情绪词典
def emotion_analysis(**path):
    r'''
    惰性加载情绪词典

    path = {
        'anger': r'week3\情绪分析\emotion_lexicon\anger.txt',
        'disgust': r'week3\情绪分析\emotion_lexicon\disgust.txt',
        'fear': r'week3\情绪分析\emotion_lexicon\fear.txt',
        'joy': r'week3\情绪分析\emotion_lexicon\joy.txt',
        'sadness': r'week3\情绪分析\emotion_lexicon\sadness.txt
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
        for word in words1:
            if word in word_to_emotions:
                for emotion in word_to_emotions[word]:
                    count[emotion] += 1
        numbers = list(count.values())
        numbers_sum = sum(numbers)
        if numbers_sum == 0:
            print('没有情绪')
            return
        weights = [i/numbers_sum for i in numbers]
        result = dict(zip(count.keys(), weights))
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
    df = df.rename(columns={'weekday':'day'})
    df.loc[df['day'] == 0,'day'] = 7
    df['datatime'] = pd.to_datetime(df[['year','month','day','hour']])

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

if __name__ == '__main__':

    path = {
        'anger': r'week3\情绪分析\emotion_lexicon\anger.txt',
        'disgust': r'week3\情绪分析\emotion_lexicon\disgust.txt',
        'fear': r'week3\情绪分析\emotion_lexicon\fear.txt',
        'joy': r'week3\情绪分析\emotion_lexicon\joy.txt',
        'sadness': r'week3\情绪分析\emotion_lexicon\sadness.txt'
        }
    mixed_analysis,single_analysis = emotion_analysis(**path)
    # mixed_analysis(words)
    # single_analysis()
    time_analysis(518986,'ME','joy')
