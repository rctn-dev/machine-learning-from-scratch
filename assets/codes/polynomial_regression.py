import numpy as np
from matplotlib import pyplot as plt
from utils.utility_functions import *

np.set_printoptions(precision=2)
x = np.arange(0, 20, 1)
y = x**2
x_matrix= np.c_[x, x**2, x**3]   #engineered new features
x_norm,_,_=zscore_normalize(x_matrix)
alpha=1e-7
alpha_norm=1e-1
iterations=100000

optim_w,optim_b=run_gradient_descent(x_matrix,y,alpha,iterations)
fig=plt.figure(figsize=(4,3))
print(optim_w,optim_b)
plt.scatter(x, y, marker='o', c='r', label="target"); 
plt.plot(x, x_matrix@optim_w+optim_b,c='b', label="predicted "); 
plt.title(f"multi-features: x, x**2, x**3",fontsize=10)
plt.annotate(f"{optim_w[0]:.1e}*x+{optim_w[1]:.1e}*x^{2}+{optim_w[2]:.1e}*x^{3}",xy=(0,350), xytext=(0, 350))
plt.annotate(r"$\alpha={alpha}$".format(alpha=alpha),xy=(0,320), xytext=(0, 320))
plt.xlabel("x"); 
plt.ylabel("y=x**2");
plt.legend(loc='lower right'); 
plt.show()
fig.savefig('assets/images/polynomial_regression.svg', format='svg', dpi=1200)

optim_w,optim_b=run_gradient_descent(x_norm,y,alpha_norm,iterations)
fig=plt.figure(figsize=(4,3))
plt.scatter(x, y, marker='o', c='r', label="target"); 
plt.title(f"multi-features: x, x**2, x**3, normalized",fontsize=10)
plt.plot(x, x_norm@optim_w+optim_b,c='b', label="predicted "); 
plt.annotate(f"w:[{optim_w[0]:.1e}, {optim_w[1]:.1e}, {optim_w[2]:.1e}]",xy=(0,340), xytext=(0, 340))
plt.annotate(r"$\alpha={alpha}$".format(alpha=alpha_norm),xy=(0,310), xytext=(0, 310))
plt.xlabel("x"); 
plt.ylabel("y=x**2");
plt.legend(loc='lower right'); 
plt.show()
fig.savefig('assets/images/polynomial_regression_norm.svg', format='svg', dpi=1200)


x = np.arange(0,20,1)
y = np.cos(x/3)
x_matrix= np.c_[x, x**2,x**3,x**4,x**5,x**6,x**7,x**8,x**9,x**10,x**11,x**12,x**13]
x_norm,_,_= zscore_normalize(x_matrix) 
iterations=1000000
alpha = 1e-1
model_w,model_b = run_gradient_descent(x_norm, y, alpha, iterations)
print(model_w)
fig=plt.figure(figsize=(4,3))
plt.scatter(x, y, marker='o', c='r', label="target"); 
plt.title("multi-features: x x**2,..., x**13, normalized",fontsize=10)
plt.plot(x,x_norm@model_w + model_b, c='b', label="predicted");
plt.xlabel("x");
plt.ylabel("y=cos(x/3)"); 

plt.annotate(r"$\alpha={alpha}$".format(alpha=alpha),xy=(10,0.7), xytext=(10, 0.7))
plt.legend(loc='lower right'); 
plt.show()
fig.savefig('assets/images/sinx_regression.svg', format='svg', dpi=1200)