def fib(n):
    old = 0
    new = 1
    count = 1
    yield new
    while (count < n):
        new, old, count = new+old, new, count+1
        yield new

print(list(fib(10)))
result = list(fib(10))
print([i+10 for i in result if i])
