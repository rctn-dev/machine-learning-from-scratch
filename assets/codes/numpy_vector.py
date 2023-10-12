import numpy as np
import time

def my_dot(a, b): 
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

 # very large arrays
a = np.random.rand(10000000) 
b = np.random.rand(10000000)

tic = time.time() 
c = np.dot(a, b)
toc = time.time()  
print(f"np.dot(a, b) =  {c:.2f},", f"  time duration: {1000*(toc-tic):.2f} ms ")

tic = time.time()  
c = my_dot(a,b)
toc = time.time()  
print(f"my_dot(a, b) =  {c:.2f},", f"  time duration: {1000*(toc-tic):.2f} ms ")
del(a);del(b)  #remove these big arrays from memory