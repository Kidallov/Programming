def two_sum_hashed_all(lst: list) -> tuple:
    target = 8
    seen = {}
    result = []

    for i, num in enumerate(lst):
        complement = target - num
        if complement in seen:
            for j in seen[complement]:
                result.append([i, j])
        if num in seen:
            print(seen[num].append(i))
        else:
            seen[num] = [i]
    return result

print(two_sum_hashed_all([9, 1, 4, 3, 5, 10, 2, 4, 0]))
