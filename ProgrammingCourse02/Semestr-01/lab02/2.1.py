def foo():
    name = 'Aleksandr'
    first_name = 'Kidalov'
    age = 'Мне 19'
    about_me = 'Мне нравится играть в видеоигры. Гулять на свежем воздухе. Смотреть аниме и сериалы, читать книги и узнавать что-то новое.'

    def boo():
        print(name)
        print(first_name)
        print(age)
        print(about_me)

    return boo

foo()()

