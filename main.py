import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Константы для уравнения
a = 0.158
b = 0.8
c = 1.164

# Начальное условие
x0 = 0.1
y0 = 0.25

# Функция для производной y'
def f(x, y):
    return a * (x**2 + np.cos(b * x)) + c * y

# Метод Адамса-Башфорта (2-го порядка) с шагом h/2
def adams_method_half_step(x0, y0, h, x_end):
    h_half = h / 2
    # Используем метод Эйлера для первого шага
    x_values, y_values = [x0], [y0]
    x1, y1 = euler_method(x0, y0, h_half, x0 + h_half)
    x_values.append(x1[1])
    y_values.append(y1[1])
    
    # Шаги метода Адамса-Башфорта на всем интервале с шагом h/2
    while x_values[-1] < x_end:
        f0 = f(x_values[-2], y_values[-2])
        f1 = f(x_values[-1], y_values[-1])
        y_next = y_values[-1] + h_half / 2 * (3 * f1 - f0)
        x_next = x_values[-1] + h_half
        x_values.append(x_next)
        y_values.append(y_next)
    
    return np.array(x_values), np.array(y_values)

# Метод Эйлера (необходим для первого шага метода Адамса-Башфорта)
def euler_method(x0, y0, h, x_end):
    n_steps = int((x_end - x0) / h) + 1
    x = np.linspace(x0, x_end, n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    for i in range(1, n_steps):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
    return x, y

# Параметры для расчета
h = 0.1
x_end = 1.2

# Решение методом Адамса-Башфорта с полушагом (h/2)
x_adams_half, y_adams_half = adams_method_half_step(x0, y0, h, x_end)

# Вычисление относительной разности (до точности 10^-5) между последовательными значениями y
relative_diff = [0]  # Для начального значения устанавливаем разницу в 0
for i in range(1, len(y_adams_half)):
    diff = abs((y_adams_half[i] - y_adams_half[i - 1]) / y_adams_half[i - 1])
    relative_diff.append(round(diff, 5))  # Округление до 5 знаков после запятой

# Создаем таблицу с x-значениями, y-значениями (Адамс h/2)
df = pd.DataFrame({
    'x': x_adams_half,
    'Адамс-Башфорт (h=0.05)': y_adams_half,
    'Относительное различие': relative_diff
})

# Отображение таблицы
print("Таблица результатов (метод Адамса-Башфорта с h/2 и относительной разностью):")
print(df)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x_adams_half, y_adams_half, label="Адамс-Башфорт (h=0.05)", marker='s', linestyle='-.')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Решение задачи Коши методом Адамса-Башфорта (h=0.05)")
plt.legend()
plt.grid(True)
plt.show()
