def two_sum(lst: list) -> tuple:
    target = 8
    seen = {}

    for i, num in enumerate(lst):
        complement = target - num
        if complement in seen:
            return seen[complement], i
        seen[num] = i

    return None

print(two_sum([9, 1, 4, 3, 5, 10, 2, 9, 0]))
