arr = [('0','0',25),
       ('0','0',36),
       ('0','0',14)]

arr.sort(key=lambda x: x[2],reverse=True)

for i in arr:
    print(*i,sep=' ')