import numpy as np
import matplotlib.pyplot as plt

# Сгенерировать данные для значений x
x = np.linspace(0, 2*np.pi, 100)
# Вычислить значения y для синусной функции
y = np.sin(x) # График синусной функции
z = np.cos(x) # График косинусной функции
plt.plot(x, y)
plt.plot(x, z)
# Добавить заголовок и метки осей
plt.title("Sine and cosine Function")
plt.xlabel("x")
plt.ylabel("y")
# Вывести график
plt.grid()
plt.show()

# Второе окно графика тангенса

x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.tan(x)
# This operation inserts a NaN where the difference between successive points is negative
# NaN means "Not a Number" and NaNs are not plotted or connected
# I found this by doing a search for "How to plot tan(x) in matplotlib without the connecting lines between asymtotes"
y[:-1][np.diff(y) < 0] = np.nan
# show grid
plt.grid()
plt.xlabel("x")
plt.ylabel("$tan(x)$")

plt.ylim(-10,10)
plt.xlim(-2 * np.pi, 2 * np.pi)

radian_multiples = [-2, -3/2, -1, -1/2, 0, 1/2, 1, 3/2, 2]
radians = [n * np.pi for n in radian_multiples]
radian_labels = ['$-2\pi$', '$-3\pi/2$', '$\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
plt.xticks(radians, radian_labels)
plt.title("y = tan(x)")
plt.plot(x, y)
plt.show()
