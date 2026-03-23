import unittest
from main import write_log

class TestSomeFunc(unittest.TestCase):

    def test_creating_file_exception(self):
        args = [1, 2, 3, 4, 5]
        log_file = 123  # Неверный тип файла
        with self.assertRaises(Exception):
            write_log(*args, action='sum', file=log_file)

    def test_creating_file_exception_with_descr(self):
        args = [1, 2, 3, 4, 5]
        log_file = 'output.invalid'  # Неверное расширение файла
        regex_text = 'Ошибка записи в файл'
        with self.assertRaisesRegex(Exception, regex_text):
            write_log(*args, action='sum', file=log_file)

if __name__ == '__main__':
    unittest.main(verbosity=2)
