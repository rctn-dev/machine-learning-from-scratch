import numpy as np 
from matplotlib import pyplot as plt

z=np.arange(-10,10,0.1)

y=1/(1+np.exp(-z))

fig=plt.figure(figsize=(4,3))
plt.plot(z,y,c='b')
plt.xlabel('z')
plt.ylabel('sig(z)')
plt.xticks([])
plt.show()
fig.savefig('assets/images/sigmoid_func.svg', format='svg', dpi=1200)

