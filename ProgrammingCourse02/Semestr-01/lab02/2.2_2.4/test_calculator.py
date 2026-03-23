import unittest
from main import addition, subtraction, multiplication, division

class TestCalculator(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(addition(1, 1), 2)
        self.assertEqual(addition(-1, 1), 0)
        self.assertEqual(addition(-1, -1), -2)

    def test_subtraction(self):
        self.assertEqual(subtraction(1, 1), 0)
        self.assertEqual(subtraction(5, 2), 3)
        self.assertEqual(subtraction(-1, 1), -2)

    def test_multiplication(self):
        self.assertEqual(multiplication(2, 3), 6)
        self.assertEqual(multiplication(0, 5), 0)
        self.assertEqual(multiplication(-2, 4), -8)

    def test_division(self):
        self.assertEqual(division(6, 3), 2)
        self.assertEqual(division(0, 5), 0)
        self.assertEqual(division(10, 2), 5)
        self.assertEqual(division(4, 0), "Error!")

if __name__ == "__main__":
    unittest.main()

