from functools import cache
@cache
def fib(n:int):
    if n <= 1:
        return n

    else:
        return(fib(n-1) + fib(n-2))

def fib_lst(n: int, numbers=None):
    if numbers == 'even':
        return [fib(i) for i in range(n) if fib(i) % 2 == 0]
    if numbers == 'odd':
        return [fib(i) for i in range(n) if fib(i) % 2 != 0]
    else:
        return [fib(i) for i in range(n)]
    
print(fib_lst(10))
print(fib_lst(10, 'even'))
print(fib_lst(10, 'odd'))
print(fib.cache_info())
