arr = list(map(float, input().split(',')))

arr.sort(reverse=True)

print(*arr,sep='\n')
