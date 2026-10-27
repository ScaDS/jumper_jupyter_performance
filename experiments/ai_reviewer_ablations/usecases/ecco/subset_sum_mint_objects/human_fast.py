import sys
def main():
    input = sys.stdin.readline
    N,S = list(map(int, input().split()))
    A = list(map(int, input().split()))
    MOD = 998244353

    dp0 = [[0 for _ in range(S+1)] for _ in range(N+1)]
    dp1 = [[0 for _ in range(S+1)] for _ in range(N+1)]
    dp2 = [[0 for _ in range(S+1)] for _ in range(N+1)]
    dp0[0][0] += 1
    for i in range(N):
        a = A[i]
        dp0i = dp0[i]
        dp1i = dp1[i]
        dp2i = dp2[i]
        dp0n = dp0[i+1]
        dp1n = dp1[i+1]
        dp2n = dp2[i+1]
        for j in range(S+1):
            dp0n[j] = (dp0n[j] + dp0i[j]) % MOD
            dp1n[j] = (dp1n[j] + dp1i[j]) % MOD
            dp2n[j] = (dp2n[j] + dp2i[j]) % MOD

            if j+a > S: continue
            dp1n[j+a] = (dp1n[j+a] + dp1i[j] + (dp0i[j] * (i+1))) % MOD
            dp2n[j+a] = (dp2n[j+a] + (dp1i[j] * (N-i)) + (dp0i[j] * (i+1) * (N-i))) % MOD

    print((dp2[N][S]))

if __name__ == '__main__':
    main()
