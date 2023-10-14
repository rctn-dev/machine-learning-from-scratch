import numpy as np
from matplotlib import pyplot as plt

def zscore_normalize(x):
    """
    Args:
      x (np.array (m,n))     : training dataset inputs, m examples, n features
    Returns:
      x_norm (np.array (m,n)): input normalized by column,corresponding to each feature.
      mu (np.array (n,))     : mean of each feature, i.e., of each column
      sigma (np.array (n,))  : standard deviation of each feature,i.e., of each column
    """
    # axis=0 means, take mean of each column (size of m) and create a row vector of means (size of n).
    mu=np.mean(x, axis=0) # dim: (n,)
    sigma=np.std(x, axis=0)# dim: (n,)               
    x_norm =(x-mu)/sigma      
    return (x_norm, mu, sigma)

data = np.loadtxt("assets/data/houses.txt",delimiter=',')  
x_train,y_train=data[:,0:4], data[:,4]
x_features = ['size(sqft)','bedrooms','floors','age']
print(x_train)
print(y_train)

fig,ax=plt.subplots(1, 4, figsize=(11, 2.5), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("Price (1000's)")
fig.suptitle('Original Input Features', fontsize=12)
plt.show()
fig.savefig('assets/images/original_feature_scaling.svg', format='svg', dpi=1200)

x_norm,_,_=zscore_normalize(x_train)
y_norm,_,_=zscore_normalize(y_train)

fig,ax=plt.subplots(1, 4, figsize=(11, 2.5), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_norm[:,i],y_norm)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("Price (1000's)")
fig.suptitle('Normalized Input Features', fontsize=12)
plt.show()
fig.savefig('assets/images/norm_feature_scaling.svg', format='svg', dpi=1200)


