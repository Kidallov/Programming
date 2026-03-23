""" Код, написанный в аудитории """
def not_rec_gen_bin_tree(root=5, height=3, l_branch=lambda x: x + 3, r_branch=lambda x: x * 2):
    roots = [[root]]
    for leaf in range(height-1):
        if (len(roots) == 1):
            r = roots[0]
        else:
            r = [item for s in roots[-1] for item in s]

        leaves = list(map(lambda root_value: [l_branch(root_value), r_branch(root_value)], r))
        roots.append(leaves)

    roots.reverse()
    roots[-1] = [roots[-1]]
    roots[0] = list(map(lambda x: [{str(x[0]): []}, {str(x[1]): []}], roots[0]))

    for i in range(height-1):
        sublist = roots[i]
        for j in range(len(sublist)):
            x = sublist.pop()
            roots[i+1][j // 2][j % 2] = {str(roots[i+1][j // 2][j % 2]): x}
    tree = roots[-1][0][0]
    return tree

if __name__ == '__main__':
    print(not_rec_gen_bin_tree(root=5, height=3))