from exceptions_for_gen_bin_tree import *
import time
import matplotlib.pyplot as plt

def setup_data(num_tests: int) -> tuple:
    test_params = []
    execution_time = []

    for i in range(num_tests):
        test_params.append((i + 1, i % 5))

    for _ in range(num_tests):
        start_time = time.time()

        for root, height in test_params:
            gen_bin_tree(root=root, h=height)

        end_time = time.time()
        execution_time.append(end_time - start_time)

    return test_params, execution_time


def gen_bin_tree(root=5, h=3, left_leaf=lambda arg: int(arg) * 3, right_leaf=lambda arg: int(arg) - 4):

    if not isinstance(h, int):
        raise InvalidHeightException(h)
    if h < 0:
        raise InvalidHeightException(h)

    tree = {str(root): []}

    if h == 0:
        return tree
    else:
        tree[str(root)].append(gen_bin_tree(h=h - 1, root=left_leaf(root)))
        tree[str(root)].append(gen_bin_tree(h=h - 1, root=right_leaf(root)))
        return tree

def main():
    max_height = 10
    res = []

    for n in range(max_height + 1):
        start_time = time.time()
        gen_bin_tree(h=n)
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