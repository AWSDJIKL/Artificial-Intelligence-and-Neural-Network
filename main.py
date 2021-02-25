# 1-3
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

a = np.asarray([4, 5, 6])
print(a.shape)
print(a[0])

b = np.asarray([[4, 5, 6], [1, 2, 3]])
print(b.shape)
print(b[0, 0])
print(b[0, 1])
print(b[1, 1])

# 4-5
a = np.matlib.zeros((3, 3), int)
b = np.matlib.ones((4, 5))
c = np.matlib.identity(4)
d = np.matlib.rand((3, 2))

a = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[2, 3])
print(a[0, 0])

# 6-7
b = np.asarray(a[0:2, 2:])
print(b)
print(b[0, 0])

c = np.asarray(a[1:, :])
print(c)
print(c[0, -1])

# 8-10
a = np.asarray([[1, 2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])

a = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])
print(a[[0, 1, 2, 3], [0, 2, 0, 1]])

a[np.arange(4), b] += 10
print(a)

# 11-12
x = np.array([1, 2])
print(x.dtype)

x = np.array([1.0, 1.0])
print(x.dtype)

# 13-18
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))
print(np.dot(x, y))

print(np.divide(x, y))

print(np.sqrt(x))

print(x.dot(y))
print(np.dot(x, y))

# 19-20
print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

print(np.mean(x))
print(np.mean(x, axis=0))
print(np.mean(x, axis=1))

# 21-23
print(x.T)

print(np.exp(x))

print(np.nanargmax(x))
print(np.nanargmax(x, axis=0))
print(np.nanargmax(x, axis=1))

# 24-25
x = np.arange(0, 100, 0.1)
y = x * x
plt.plot(y)
plt.show()

x = np.arange(0, 3 * np.pi, 0.1)
plt.plot(np.sin(x))
plt.plot(np.cos(x))
plt.show()
