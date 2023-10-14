import numpy as np
from matplotlib import pyplot as plt


data = np.loadtxt("assets/data/houses.txt",delimiter=',')  
x_train,y_train=data[:,0:4], data[:,4]
x_features = ['size(sqft)','bedrooms','floors','age']
print(x_train)
print(y_train)

fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()


