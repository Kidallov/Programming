import time
import sys
sys.set_int_max_str_digits(0)

class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time} seconds")

with Timer() as timer:
    def fib(n):
        a, b = 0, 1
        for __ in range(n):
            yield a
            a, b = b, a + b
    print(fib(10000000))
