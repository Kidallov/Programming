import itertools
def two_sum(lst: list) -> tuple:
    target = 8
    result = None
    for (i, val1), (j, val2) in itertools.combinations(enumerate(lst), 2):
        if lst[i] + lst[j] == target and (i != j):
            result = (i, j)
    if result is None:
        return None
    return tuple(sorted(result))
print(two_sum([9, 1, 4, 1, 5, 10, 2, 9, 4]))
