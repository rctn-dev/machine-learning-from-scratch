import numpy as np

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
    mu=np.mean(x,axis=0) # dim: (n,)
    sigma=np.std(x,axis=0)# dim: (n,)               
    x_norm=(x-mu)/sigma      
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
    y_hat=x@w+b # for matrix-vector multiplication, use np.dot or @ or np.matmul
    #y_hat=np.dot(x,w)+b # use np.dot or @ or np.matmul
    #y_hat=np.matmul(x,w)+b# use np.dot or @ or np.matmul
    cost=(1/(2*m))*np.sum((y_hat-y)**2)
    return cost

def compute_partial_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (np.array (m,n)): Data, m examples with n features
      y (np.array (m,)) : target values
      w (np.array (n,)) : model parameters  
      b (scalar)        : model parameter
      
    Returns:
      dj_dw (np.array (n,)): The gradient vector of the cost w.r.t. the parameters w. 
      dj_db (scalar)       : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n=x.shape           #(number of examples, number of features)
    dj_dw=np.zeros((n,))
    dj_db=0.
    y_hat=x@w+b
    err=y_hat-y 
    # matrix operation makes the error sum autimatically for the dj_dw
    dj_dw = (1/m)*(x.T@err)    
    dj_db = (1/m)*np.sum(err)                                                                           
    return dj_db, dj_dw

def compute_gradient_descent(x, y, w_init, b_init,alpha, max_iters,compute_cost, compute_partial_gradient): 
    '''
    Args:
      x (np.array (m,n))   : Data, m examples with n features
      y (np.array (m,))    : target values
      w_init (ndarray (n,)) : initial model parameters  
      b_init (scalar)       : initial model parameter
      compute_cost       : compute cost
      compute_partial_gradient   : compute the partial gradient
      alpha (float)       : Learning rate
      max_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      '''
    cost_history = []
    w = w_init
    b = b_init
    for i in range(max_iters):
        dj_db,dj_dw = compute_partial_gradient(x, y, w, b)
        w = w - alpha * dj_dw        
        b = b - alpha * dj_db              
        cost_history.append( compute_cost(x, y, w, b))
    return w, b, cost_history

def run_gradient_descent(x,y,alpha,iterations):
    w_init=np.zeros(x.shape[1])
    b_init=0.
    w_final, b_final,_=compute_gradient_descent(x, y, w_init, b_init,alpha, iterations,compute_cost, compute_partial_gradient)
    return w_final,b_final


