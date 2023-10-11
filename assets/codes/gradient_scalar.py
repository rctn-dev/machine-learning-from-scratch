import math
import numpy as np
from matplotlib import pyplot as plt 

def compute_cost(x, y, w, b):
    '''
    Args:
      x (np.array (m,)): training dataset, input features.
      y (np.array (m,)): training dataset, target values.
      w,b (scalar)     : model parameters for single feature linear regression, w is NOT vector.
   
    Return:
      cost (scalar): total cost,i.e., square-error between the target and the estimated output. 
    '''
    m = x.shape[0] 
    cost = 0
    for i in range(m):
        y_hat = w * x[i] + b
        cost = cost + (y_hat - y[i])**2
    cost = 1/(2 * m) * cost
    return cost

def compute_partial_gradient(x, y, w, b): 
    """
    Computes the partial gradient for linear regression with single feature.
    Args:
      x (np.array (m,)): training dataset, input features.
      y (np.array (m,)): training dataset, target values.
      w,b (scalar)     : model parameters for single feature linear regression, w is NOT vector.
    Return:
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    for i in range(m):  
        y_hat=w*x[i]+b 
        dj_dw+= (y_hat - y[i]) * x[i] 
        dj_db+= y_hat - y[i] 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
    return dj_dw, dj_db

def compute_gradient_descent(x, y, w_init, b_init, alpha, max_iters, compute_cost, compute_partial_gradient): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (np.array (m,))  :Training dataset, input features
      y (np.array (m,))  :Training dataset, target values
      w_init,b_init (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      compute_cost:     function to call to produce cost
      compute_partial_gradient: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      cost_history (List): History of cost values
      wb_history (list): History of parameters [w,b] 
      """
    cost_history = []
    wb_history = []
    b = b_init
    w = w_init
    
    for i in range(max_iters):
        dj_dw, dj_db = compute_partial_gradient(x, y, w , b)     
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            
        cost_history.append( compute_cost(x, y, w , b))
        wb_history.append([w,b]) 
    return w, b, cost_history, wb_history 

# initialize parameters
w_init = 0.1
b_init = 10
max_iters = 10000
alpha = 1e-2
x_train=np.array([1,2])
y_train=np.array([300,500])

# Second dataset is problematic, why???
# x_train=np.array([480,600,750,900,1000,1200,1500,1600,1800,2200])
# y_train=np.array([85,120,200,150,250,230,280,260,290,310])

# run gradient descent
w_final, b_final, J_hist, p_hist = compute_gradient_descent(x_train ,y_train, w_init, b_init, alpha, 
                                                    max_iters, compute_cost, compute_partial_gradient)
print(f"Optimum (w,b): ({w_final:9.4f},{b_final:9.4f})")
plt.plot(J_hist[1000:])
plt.plot(J_hist[1000:])
plt.xlabel("# of iterations")
plt.ylabel("Cost")
plt.show()
