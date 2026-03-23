# Бинарный поиск
import random """импорт нужной библиотеки"""

hidden_number = int(input('Загадайте число (от 0 до 100): '))
num_1 = random.randint(0, hidden_number)
num_2 = random.randint(num_1, 100)
number_list = list(range(num_1, num_2 + 1))
print(f"Диапазон чисел: {number_list}")
"""Даем пользователю ввести загаданное число и создаем список чисел от него до 100 и выводим его на экран."""

count = 0
"""Создаем счетчик"""

while len(number_list) > 0:
    mid_index = len(number_list) // 2
    mid_number = number_list[mid_index]
    count += 1

    if mid_number == hidden_number:
        print(f"Угадано число {hidden_number} за {count} попыток!")
        break
    elif mid_number > hidden_number:
        number_list = number_list[:mid_index] 
    else:
        number_list = number_list[mid_index + 1:]
"""Главная функция, которая выполняет поиск числа пока ее длина не станет = 0. После каждой попытки мы увелииваем счетчик на 1, чтобы в конце вывести число попыток. Также с помощью условий мы можем обрезать один или другой конец списка, смотря где находится наше загаданное число. """



# Медленный перебор
import random
"""импорт нужной библиотеки"""

hidden_number = int(input('Загадайте число (от 0 до 100): '))
num_1 = random.randint(0, hidden_number)
num_2 = random.randint(num_1, 100)
number_list = list(range(num_1, num_2 + 1))
print(f"Диапазон чисел: {number_list}")
count = 1
for i in number_list:
  if i == hidden_number:
    print(f"Угадано число {hidden_number} за {count} попыток!")
    break
  else: 
    count += 1