class BinaryTreeException(Exception):
    def __init__(self, message="Ошибка в бинарном дереве"):
        super().__init__(message)

class InvalidHeightException(BinaryTreeException):
    def __init__(self, h):
        super().__init__(f"Invalid height: {h}. Height must be a non-negative integer.")