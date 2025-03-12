import random

def my_random(keys,weights):
    total_weights = sum(weights)
    # 归一化
    for i in range(len(weights)):
        weights[i] = weights[i]/total_weights
    # 关联两个列表
    paired = list(zip(keys,weights))
    p = random.random()
    for pair in paired:
        if p < pair[1]:
            return pair[0]
        else:
            p -= pair[1]


if __name__ == '__main__':
    for i in range(10):
        print(my_random(['a','b','c','d'],[1,2,3,4]))    