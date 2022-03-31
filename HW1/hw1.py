import torch
import hw1_utils as utils
import matplotlib.pyplot as plt
import numpy
from itertools import combinations
'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 4 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 3
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    x = torch.ones(X.shape[0]).reshape(-1,1)
    X = torch.cat((x, X), 1)
    n_samples, n_features = X.shape
    print(n_samples, n_features)

    def forward(x, w):
        return torch.matmul(x,w)
    
    def loss(y,y_predicted):
        return (1/2*(y_predicted-y)**2).mean()

    
    w = torch.tensor([[0] for i in range (n_features)], dtype=torch.float32, requires_grad=True)
    
    for epoch in range(num_iter):
        y_pred = forward(X, w)

        l=loss(Y,y_pred)
    
        l.backward() 
        
        with torch.no_grad():
            w-= lrate*w.grad

        if epoch%100 == 0:
            print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
        
        w.grad.zero_()
        
    return w

def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    x = torch.ones(X.shape[0]).reshape(-1,1)
    X = torch.cat((x, X), 1)
    n_samples, n_features = X.shape
    print(n_samples, n_features)
    w = torch.pinverse(X)@Y
    
    return w

# def plot_linear():
#     '''
#         Returns:
#             Figure: the figure plotted with matplotlib
#     '''
#     X, Y = utils.load_reg_data()
#     n_samples, n_features = X.shape
#     w = linear_normal(X,Y).clone().detach().requires_grad_(True)
#     print(w)
#     x = torch.ones(X.shape[0]).reshape(-1,1)
#     X = torch.cat((x, X), 1)
#     y_pred = (X@w).detach().numpy()
#     X = X[:,1:]
#     X_np = X.detach().numpy()
#     Y_np = Y.detach().numpy()
#     w_np = w.detach().numpy()
# #     plt.plot(X_np, Y_np, 'bo')
# #     plt.plot(X_np, y_pred, 'y')
# #     plt.show()

# plot_linear()

# Problem 4
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    x = torch.ones(X.shape[0]).reshape(-1,1) 
    
    X = torch.cat((x, X), 1)

    d = [i for i in range(1,X.shape[1])] 

#     empty_dict = dict()
#     empty_list =[]

#     for i in d:
#         for j in range(i+1):
#             if i in empty_dict:
#                 empty_dict[i].append(j)
#             else:
#                 empty_dict[i]=[]
#     for i in empty_dict.keys():
#         #print(i)
#         for j in empty_dict[i]:
#             x = (j,i)
#             empty_list.append(x) 
    
#     for tuples in empty_list:
#         X = torch.cat((X, X[:,tuples[0]].reshape(-1,1)*X[:,tuples[1]].reshape(-1,1)), 1)
    
    combx = set()
    if (len(d) >= 2):
        for (i,j) in combinations(d,2):
            combx.update([(i,i),(j,j),(i,j)])
        combx = list(combx)

    else:
        combx = [(i,i) for i in d]
    
    
    for idx in combx:
        X = torch.cat((X, X[:,idx[0]].reshape(-1,1)*X[:,idx[1]].reshape(-1,1)), 1)
    
    def forward(x, w):
        return torch.matmul(x, w)
    
    def loss(y,y_predicted):
        return(1/2*(y_predicted-y)**2).mean()
    
    
    learning_rate = lrate
    n_iters = num_iter
    
    w = torch.tensor([[0] for i in range (X.shape[1])], dtype=torch.float32, requires_grad=True)
    
    for epoch in range(n_iters):
        y_pred = forward(X,w)

        l=loss(Y,y_pred)
    
        l.backward() 
        
        with torch.no_grad():
            w-= lrate*w.grad
        
        w.grad.zero_()
        
#         if epoch%100 == 0:
#             print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
    
    return w

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    x = torch.ones(X.shape[0]).reshape(-1,1) 
    
    X = torch.cat((x, X), 1)

    d = [i for i in range(1,X.shape[1])] 
    
    combx = set()
    if (len(d) >= 2):
        for (i,j) in combinations(d,2):
            combx.update([(i,i),(j,j),(i,j)])
        combx = list(combx)
        combx.sort()

    else:
        combx = [(i,i) for i in d]
    
    for idx in combx:
        X = torch.cat((X, X[:,idx[0]].reshape(-1,1)*X[:,idx[1]].reshape(-1,1)), 1)
    
    w = torch.pinverse(X)@Y
    
    return w

# def plot_poly():
#     x = torch.ones(X.shape[0]).reshape(-1,1) 
    
#     X = torch.cat((x, X), 1)

#     d = [i for i in range(1,X.shape[1])] 
    
#     combx = set()
#     if (len(d) >= 2):
#         for (i,j) in combinations(d,2):
#             combx.update([(i,i),(j,j),(i,j)])
#         combx = list(combx)
#         combx.sort()

#     else:
#         combx = [(i,i) for i in d]
    
#     for idx in combx:
#         X = torch.cat((X, X[:,idx[0]].reshape(-1,1)*X[:,idx[1]].reshape(-1,1)), 1)
    
#     w = torch.pinverse(X)@Y
#     '''
#     Returns:
#         Figure: the figure plotted with matplotlib
#     '''
#     n_samples, n_features = X.shape
#     temp = torch.ones(n_samples, 1)
#     y_pred = (X@w).detach().numpy()
#     X = X[:,1:2]
#     X_np = X.detach().numpy()
#     Y_np = Y.detach().numpy()
#     w_np = w.detach().numpy()
#     plt.plot(X_np, Y_np, 'bo')
#     plt.plot(X_np, y_pred, 'y')
#     plt.show()
#     pass

# def poly_xor():
#     '''
#     Returns:
#         n x 1 FloatTensor: the linear model's predictions on the XOR dataset
#         n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
#     '''
#     X,Y = utils.load_xor_data()
    
#     lin_w = linear_normal(X, Y)
#     poly_w = poly_normal(X,Y)
    
#     def pred_fxn1(X):
#         x = torch.ones(X.shape[0]).reshape(-1,1)
#         X = torch.cat((x, X), 1)
#         n_samples, n_features = X.shape
#         print(n_samples, n_features)
#         w = X@lin_w
          
#         return w

    
#     def pred_fxn2(X):

#         x = torch.ones(X.shape[0]).reshape(-1,1) 
      
#         X = torch.cat((x, X), 1)

#         d = [i for i in range(1,X.shape[1])] 
      
#         combx = set()
#         if (len(d) >= 2):
#             for (i,j) in combinations(d,2):
#                 combx.update([(i,i),(j,j),(i,j)])
#             combx = list(combx)
#             combx.sort()

#         else:
#             combx = [(i,i) for i in d]
      
#         for idx in combx:
#             X = torch.cat((X, X[:,idx[0]].reshape(-1,1)*X[:,idx[1]].reshape(-1,1)), 1)
      
#         w = X@poly_w
      
#         return w
    
#     xmin = -5 
#     xmax = 5
#     ymin = -5
#     ymax = 5
    
#     linear_plot = utils.contour_plot(xmin, xmax, ymin, ymax, pred_fxn1)
#     poly_plot = utils.contour_plot(xmin, xmax, ymin, ymax, pred_fxn2)
    
#     return pred_fxn1(X), pred_fxn2(X)

# Problem 5
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    x = torch.ones(X.shape[0]).reshape(-1,1)
    X = torch.cat((x, X), 1)
    n_samples, n_features = X.shape
    print(n_samples, n_features)

    def forward(x, w):
        return torch.matmul(x,w)
    
    def loss(y,y_predicted):
        return (torch.log(1+torch.exp(-1*y*y_pred))).mean()

    
    w = torch.tensor([[0] for i in range (n_features)], dtype=torch.float32, requires_grad=True)
    
    for epoch in range(num_iter):
        y_pred = forward(X, w)

        l=loss(Y,y_pred)
    
        l.backward() 
        
        with torch.no_grad():
            w-= lrate*w.grad

#         if epoch%100 == 0:
#             print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
        
        w.grad.zero_()
    
    w=w.detach()
    return w

# def logistic_vs_ols():
#     '''
#     Returns:
#         Figure: the figure plotted with matplotlib
#     '''
#     X, Y = utils.load_logistic_data()
    
#     wlg = log_reg(X, Y, 0.01, 1000)
    
#     wlin = linear_gd(X, Y,0.01, 1000)
    
#     x = torch.ones(X.shape[0]).reshape(-1,1)   
#     X = torch.cat((x, X), 1)

#     n_samples, n_features = X.shape
    
#     lin =(X@wlin).detach().numpy()
#     log =(X@wlg).detach().numpy()
    
#     lin1 = np.random.choice(X[:,1], size = 70)
#     log1 = np.random.choice(X[:,1], size = 70)
    
#     log2 = -(wlg[0,:].item() + wlg[1,:].item() * log1)/wlg[2,:].item()
    
#     lin2 = -(wlin[0,:].item() + wlin[1,:].item() * lin1)/wlin[2,:].item()
    
#     X = X[:, 1:].numpy()
#     plt.scatter(X[:,0], X[:,1], c= Y)
#     plt.plot(log1, log2, label = "Log")
#     plt.legend()
#     plt.show()
#     plt.scatter(X[:,0], X[:,1],c= Y)
#     plt.plot(lin1, lin2, 'g', label = "Linear")
#     plt.legend()
#     plt.show()
#     pass
