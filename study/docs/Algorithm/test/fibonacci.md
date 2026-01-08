# Fibonacci 수열 구하기

## 문제 설명
n번째 피보나치 수를 구하는 문제입니다.

## 입력
- n: 1 ~ 100000

## 출력
- n번째 피보나치 수를 출력

## 예시
- 입력: 5
- 출력: 5

## 풀이 전략
1. 동적 프로그래밍을 이용한 풀이
2. 점화식: F(n) = F(n-1) + F(n-2)
3. 시간복잡도: O(n)
4. 공간복잡도: O(n)

## 코드
```python
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```
