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

def compute_cost(x, y, w, b): 
    """
    Args:
      X (np.darray (m,n)): training datasets, m examples with n features
      y (np.array (m,)) : target values
      w (np.array (n,)) : model parameters, n features  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]
    cost = 0.0
    for i in range(m):                                
        y_hat = np.dot(x[i], w) + b           #(n,).(n,) -> scalar 
        cost = cost + (y_hat - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

def compute_partial_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (np.array (m,n)): Data, m examples with n features
      y (np.array (m,)) : target values
      w (np.array (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (np.array (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = x.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):                             
        diff = (np.dot(x[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] += diff * x[i, j]    
        dj_db+= diff                        
    dj_dw=dj_dw/m   
    dj_db=dj_db/m                                                    
    return dj_db, dj_dw

def compute_gradient_descent(X, y, w_init, b_init,alpha, max_iters,compute_cost, compute_partial_gradient): 
    '''
    Args:
      X (np.array (m,n))   : Data, m examples with n features
      y (np.array (m,))    : target values
      w_init (ndarray (n,)) : initial model parameters  
      b_init (scalar)       : initial model parameter
      compute_cost       : compute cost
      compute_partial_gradient   : compute the partial gradient
      alpha (float)       : Learning rate
      max_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (np.array (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      '''

    J_history = []
    w = w_init
    b = b_init
    for i in range(max_iters):
        dj_db,dj_dw = compute_partial_gradient(X, y, w, b)
        w = w - alpha * dj_dw        
        b = b - alpha * dj_db              
        J_history.append( compute_cost(X, y, w, b))
    return w, b, J_history

data = np.loadtxt("assets/data/houses.txt",delimiter=',')  
x_train,y_train=data[:,0:4], data[:,4]

w_init = np.zeros(x_train.shape[1])
b_init = 0.
max_iters = 1000
alpha = 1.0e-1

x_norm,x_mu,x_sigma=zscore_normalize(x_train)
w_norm, b_norm,cost_history =compute_gradient_descent(x_norm, y_train, w_init, b_init,alpha, max_iters,compute_cost, compute_partial_gradient)
print(f'w_norm,b_norm:{w_norm}, {b_norm:.2f}')
plt.plot(cost_history)
plt.show()


x_house = np.array([1200, 3, 1, 40])
# Normalize the input feature
x_house_norm = (x_house - x_mu) / x_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
print(cost_history[-1])