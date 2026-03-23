import random

class RandomNumberIterator():

    def __init__(self, lst:list):
        self._params = lst
        self._index = 0
        self._numbers = [random.randint(1, 100) for _ in range(10)]

    def __iter__(self):
        self._index = 0
        while self._index < len(self._numbers):
            elem = self._numbers[self._index]
            self._index += 1
            yield elem

    def get_params(self):
        return self._params

def main():
    iterable = list(RandomNumberIterator([]))
    print(iterable)

if __name__ == '__main__':
    main()
