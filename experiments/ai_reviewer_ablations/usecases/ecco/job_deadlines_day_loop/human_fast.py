n = int(eval(input()))
works = [[]] * n
for i in range(n):
  a, b = list(map(int, input().split(' '))) # a = 時間, b = 締め切り
  works[i] = [a, b]

ans = 'Yes'
sorted_works = sorted(works, key=lambda x: x[1]) # 締切の早い順にソート
total = 0

for work in sorted_works:
  total += work[0]
  if total > work[1]:
    ans = 'No'
    break
  
print(ans)
