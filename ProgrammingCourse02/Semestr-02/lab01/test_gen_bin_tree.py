import unittest

from exceptions_for_gen_bin_tree import InvalidHeightException, BinaryTreeException
from gen_bin_tree import gen_bin_tree
from notrec_gen_bin_tree import not_rec_gen_bin_tree

class TestGenBinTree(unittest.TestCase):

    def test_default_tree(self):
        tree = gen_bin_tree()
        self.assertIsInstance(tree, dict)
        self.assertIn('7', tree)

    def test_zero_height(self):
        tree = gen_bin_tree(h=0, root=10)
        self.assertEqual(tree, {'10': []})

    def test_negative_height(self):
        with self.assertRaises(InvalidHeightException):
            gen_bin_tree(h=-1, root=10)

    def test_non_integer_height(self):
        with self.assertRaises(InvalidHeightException):
            gen_bin_tree(h='string')

    def test_invalid_root_type(self):
        with self.assertRaises(ValueError):
            gen_bin_tree(h=3, root='string')

    def test_custom_leaf_functions(self):
        tree = gen_bin_tree(h=1, root=10, left_leaf=lambda x: x + 2, right_leaf=lambda x: x - 2)
        self.assertEqual(list(tree.keys()), ['10'])
        self.assertEqual(list(tree['10'][0].keys()), ['12'])
        self.assertEqual(list(tree['10'][1].keys()), ['8'])

class TestNotRecGenBinTree(unittest.TestCase):
    def test_default_tree(self):
        expected_root = "7"
        tree = not_rec_gen_bin_tree()
        self.assertIn(expected_root, tree)
        self.assertIsInstance(tree, dict)

    def test_tree_structure(self):
        tree = not_rec_gen_bin_tree(h=2, root=5)
        self.assertIn("5", tree)
        self.assertEqual(tree["5"], ["15", "1"])
        self.assertIn("15", tree)
        self.assertIn("1", tree)
        self.assertEqual(tree["15"], ["45", "11"])
        self.assertEqual(tree["1"], ["3", "-3"])
        self.assertEqual(tree["45"], [])
        self.assertEqual(tree["11"], [])
        self.assertEqual(tree["3"], [])
        self.assertEqual(tree["-3"], [])

    def test_custom_functions(self):
        left_func = lambda x: x + 2
        right_func = lambda x: x * 2
        tree = not_rec_gen_bin_tree(h=2, root=1, left_leaf=left_func, right_leaf=right_func)
        self.assertEqual(tree["1"], ["3", "2"])
        self.assertEqual(tree["3"], ["5", "6"])
        self.assertEqual(tree["2"], ["4", "4"])

    def test_invalid_height(self):
        with self.assertRaises(InvalidHeightException):
            not_rec_gen_bin_tree(h=-1)

    def test_non_integer_height(self):
        with self.assertRaises(InvalidHeightException):
            not_rec_gen_bin_tree(h='string')

    def test_empty_tree(self):
        tree = not_rec_gen_bin_tree(h=0, root=10)
        self.assertEqual(tree, {"10": []})

if __name__ == "__main__":
    unittest.main()
