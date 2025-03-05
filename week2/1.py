students = eval(input())
# 一个列表包含多个学生信息，每一个为元组（学号，奖项，得分）

students.sort(key=lambda x: x[2], reverse=True)

for i in students:
    print(*i,sep=' ')