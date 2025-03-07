singlist_number = int(input())

all_comments = []
sing_number = 0

for i in range(singlist_number):
    count = int(input())
    sing_number += count
    for _ in range(count):
        info = input().split()
        all_comments.append(info)

all_comments.sort(key=lambda x: int(x[1]),reverse=True)

for item in all_comments:
    print(*item)