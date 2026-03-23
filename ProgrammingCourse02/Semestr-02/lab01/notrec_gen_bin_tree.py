from collections import deque
import json
from exceptions_for_gen_bin_tree import *

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

if __name__ == '__main__':
    print(json.dumps(not_rec_gen_bin_tree(), indent=4, ensure_ascii=False))
