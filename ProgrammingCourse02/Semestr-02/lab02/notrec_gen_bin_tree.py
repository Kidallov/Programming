from collections import deque
import time
from exceptions_for_gen_bin_tree import *
import matplotlib.pyplot as plt

def setup_data(num_tests: int) -> tuple:
    test_params = []
    execution_time = []

    for i in range(num_tests):
        test_params.append((i + 1, i % 5))

    for _ in range(num_tests):
        start_time = time.time()

        for root, height in test_params:
            not_rec_gen_bin_tree(root=root, h=height)

        end_time = time.time()
        execution_time.append(end_time - start_time)

    return test_params, execution_time

def not_rec_gen_bin_tree(h=4, root=7, left_leaf=lambda arg: int(arg) * 3, right_leaf=lambda arg: int(arg) - 4):

    if not isinstance(h, int):
        raise InvalidHeightException(h)
    if h < 0:
        raise InvalidHeightException(h)

    tree = {}
    queue = deque([(root, h)])

    while queue:
        num, lvl = queue.popleft()
        if lvl > 0:
            left_child, right_child = left_leaf(num), right_leaf(num)
            tree[str(num)] = [str(left_child), str(right_child)]
            queue.extend([(left_child, lvl - 1), (right_child, lvl - 1)])
        else:
            tree[str(num)] = []

    return tree

def main():
    max_height = 10
    res = []

    for n in range(max_height + 1):
        start_time = time.time()
        not_rec_gen_bin_tree(h=n)
        end_time = time.time()
        res.append(end_time - start_time)

    plt.figure(figsize=(10, 5))
    plt.plot(range(max_height + 1), res, marker='o', linestyle='-')
    plt.title('Время выполнения gen_bin_tree по высоте дерева')
    plt.xlabel('Высота дерева')
    plt.ylabel('Время выполнения (сек)')
    plt.xticks(range(max_height + 1))
    plt.grid()
    plt.show()

if __name__ == '__main__':
    test_params, execution_time = setup_data(10)

    print("Test Parameters (root, height):")
    print(test_params)

    print(f"Run {execution_time[-1]:.6f} seconds")
    main()
