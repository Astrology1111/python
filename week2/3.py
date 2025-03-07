# 学生数量 课数量
n,m = map(int,input().split())

# 学生名字
names = set(input().split())

# 每个课程的名单
for _ in range(m):
    name_list = set(input().split())
    names -= name_list

# 输出学生数量
print(len(names))