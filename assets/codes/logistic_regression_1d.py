import numpy as np
from matplotlib import pyplot as plt
from utils.gradient_scalar import *

np.set_printoptions(precision=2)
alpha=1e-7
alpha_norm=1e-6
iterations=3000000
x_train = np.array([0., 1, 2, 3, 4, 5,6,7,8])
y_train = np.array([0,  0, 0, 1, 1, 1,0,0,1])

optim_w,optim_b=run_gradient_descent(x_train,y_train,alpha,iterations)

fig=plt.figure(figsize=(4,3))

plt.scatter(x_train, y_train,c='b')
plt.plot(x_train, optim_w*x_train+optim_b,c='k')
print(f"optim_w: {optim_w}, optim_b: {optim_b}")


# x=np.arange(0,5,0.1)
# y=1/(1+np.exp(-x))
# plt.plot(x, y)

plt.show()