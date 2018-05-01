def factorial(n):
    if n ==1 or n ==0:
        return 1
    return n*factorial(n-1)

for i in range(1,1000):
    if ((2**i)*(i**2) < factorial(i-1)):
        print(i)
        break
