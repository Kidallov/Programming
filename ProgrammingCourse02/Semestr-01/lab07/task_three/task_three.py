def factorial(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError("n должно быть натуральным числом")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

