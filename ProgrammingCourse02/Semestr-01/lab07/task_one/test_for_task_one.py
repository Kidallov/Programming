import unittest
import tempfile
import os
from task_one import read_settings, write_to_file


class TestSettings(unittest.TestCase):
    def setUp(self):
        """Создает временный файл для тестов."""
        self.tempfile = tempfile.NamedTemporaryFile(delete=False)
        with open(self.tempfile.name, 'w') as file:
            file.write("[database]\nhost = localhost\nport = 3306\nuser = user\npassword = secret")

    def tearDown(self):
        """Удаляет временный файл после тестов."""
        os.remove(self.tempfile.name)

    def test_read_settings(self):
        """Тестирует корректное считывание настроек."""
        settings = read_settings(self.tempfile.name)
        self.assertEqual(settings['host'], 'localhost')
        self.assertEqual(settings['port'], 3306)
        self.assertEqual(settings['user'], 'user')
        self.assertEqual(settings['password'], 'secret')

    def test_file_not_found_error(self):
        """Тестирует обработку ошибки FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            read_settings('nonexistent.ini')

    def test_value_error(self):
        """Тестирует обработку ошибки ValueError при некорректном формате."""
        with open(self.tempfile.name, 'w') as file:
            file.write("invalid content")
        with self.assertRaises(ValueError):
            read_settings(self.tempfile.name)


class TestFileWrite(unittest.TestCase):
    def setUp(self):
        """Создает временный файл для тестов."""
        self.tempfile = tempfile.NamedTemporaryFile(delete=False)
        self.tempfile.close()

    def tearDown(self):
        """Удаляет временный файл после тестов."""
        os.remove(self.tempfile.name)

    def test_write_to_file(self):
        """Тестирует запись строки в файл."""
        write_to_file(self.tempfile.name, "Hello")
        with open(self.tempfile.name, 'r') as file:
            content = file.read().strip()
        self.assertEqual(content, "Hello")

    def test_append_to_file(self):
        """Тестирует добавление строки в существующий файл."""
        write_to_file(self.tempfile.name, "First line")
        write_to_file(self.tempfile.name, "Second line")
        with open(self.tempfile.name, 'r') as file:
            content = file.read().strip().split('\n')
        self.assertEqual(content, ["First line", "Second line"])

    def test_create_file_if_absent(self):
        """Тестирует создание файла, если он отсутствует."""
        os.remove(self.tempfile.name)  # Удаляем файл
        write_to_file(self.tempfile.name, "New content")
        with open(self.tempfile.name, 'r') as file:
            content = file.read().strip()
        self.assertEqual(content, "New content")


if __name__ == '__main__':
    unittest.main()

