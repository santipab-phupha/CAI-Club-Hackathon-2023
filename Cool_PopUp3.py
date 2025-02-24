def p(n, m):
    s = [True] * (m+1)
    s[0:2] = [False, False] 
    for i in range(2, int(m**0.5) + 1):
        if s[i]:
            for j in range(i*i, m+1, i):
                s[j] = False
    return [num for num in range(n, m+1) if s[num]]

a = input(" ").split(" ")
n = int(a[0])
m = int(a[1])
b = ""
for k in p(n,m):
    b+=str(k)
    b+=" "
print(b)