import json
from exceptions_for_gen_bin_tree import *

def gen_bin_tree(h=4, root=7, left_leaf=lambda arg: int(arg) * 3, right_leaf=lambda arg: int(arg) - 4):

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

if __name__ == '__main__':
    print(json.dumps(gen_bin_tree(), indent=4, ensure_ascii=False))
