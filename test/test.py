import matplotlib.pyplot as plt
import numpy as np


def laplace_pdf(x, b, μ):
    return (1 / (2 * b)) * np.e ** (-1 * (np.abs(x - μ) / b))


x = np.linspace(-10, 10, 10000)
y1 = [laplace_pdf(x_, 1, 0) for x_ in x]
y2 = [laplace_pdf(x_, 2, 0) for x_ in x]
y3 = [laplace_pdf(x_, 4, 0) for x_ in x]
y4 = [laplace_pdf(x_, 4, -5) for x_ in x]

plt.plot(x, y1, label="μ=0,b=1")
plt.plot(x, y2, label="μ=0,b=2")
plt.plot(x, y3, label="μ=0,b=4")
plt.plot(x, y4, label="μ=-5,b=4")

plt.title("Laplace pdf curves")
plt.legend()
plt.show()
