import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    a = torch.zeros([x_train.shape[0]])
    a.requires_grad_()
    def loss(alf):
        l1 = 0
        rows=x_train.shape[0]
        for i in range(rows):
            for j in range(rows):
                l1 += 0.5*(alf[i]*alf[j]*y_train[i]*y_train[j]*kernel(x_train[i],x_train[j]))
        l1 = l1-torch.sum(alf)
        return l1
    
    for epoch in range(num_iters):
        
        l=loss(a)

        l.backward() 

        with torch.no_grad():
            a-= lr*a.grad
            
            a.grad.zero_()
#             a = a.clone().detach().requires_grad_(True)
            if c==None:
                a.clamp_(0, float('inf'))
            else:
                a.clamp_(0, c)
            
            a.requires_grad_()
    return a.view(-1,)
  

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    pass

class CAFENet(nn.Module):
    def __init__(self):
        '''
            Initialize the CAFENet by calling the superclass' constructor
            and initializing a linear layer to use in forward().

            Arguments:
                self: This object.
        '''
        super(CAFENet,self).__init__()
        input_size = 91200
        classes = 6
        
        self.linear1 = nn.Linear(input_size, classes)

    def forward(self, x):
        '''
            Computes the network's forward pass on the input tensor.
            Does not apply a softmax or other activation functions.

            Arguments:
                self: This object.
                x: The tensor to compute the forward pass on.
        '''
        out = self.linear1(x)
        return out

def fit(net, X, y, n_epochs=201):
    '''
    Trains the neural network with CrossEntropyLoss and an Adam optimizer on
    the training set X with training labels Y for n_epochs epochs.

    Arguments:
        net: The neural network to train
        X: n x d tensor
        y: n x 1 tensor
        n_epochs: The number of epochs to train with batch gradient descent.

    Returns:
        List of losses at every epoch, including before training
        (for use in plot_cafe_loss).
    '''
    input_size = X.shape[1]
    classes = torch.unique(y).shape[0]
    l_rate = 0.001
    #model = net(input_size, classes)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = l_rate)
    losses = []
    for epochs in range(n_epochs):
        y_pred = net(X)

        l=loss(y_pred,y)
    
        l.backward() 
        
        optimizer.step()
    
        optimizer.zero_grad()
        
        losses.append(l.item())
        
        w,b = net.parameters()
    #print(losses)
    
    return w,b,losses

def plot_cafe_loss():
    '''
    Trains a CAFENet on the CAFE dataset and plots the zero'th through 200'th
    epoch's losses after training. Saves the trained network for use in
    visualize_weights.
    '''
    x, y = utils.get_cafe_data()
    print(x.shape[1])
    # x.shape[0]
    # print(torch.unique(y).shape[0])
    net1 = CAFENet(x.shape[1],torch.unique(y).shape[0])
    w, b, l = fit(net1,x,y) 
    print (w,b)
    plt.plot(l)
    plt.show
    torch.save(net1, 'HW2Q3')
    
    
def print_confusion_matrix():
    '''
    Loads the CAFENet trained in plot_cafe_loss, loads training and testing data
    from the CAFE dataset, computes the confusion matrices for both the
    training and testing data, and prints them out.
    '''
    model = torch.load('HW2Q3')
    dataset_train, labels_train = utils.get_cafe_data(set='train')
    dataset_test, labels_test = utils.get_cafe_data(set='test')
    train_output = model(train_dataset).detach()
    test_output = model(test_dataset).detach()
    train_output_list = []
    test_output_list = []
    for i in train_output: 
        train_output_list.append(torch.argmax(torch.exp(i/100)/torch.sum(torch.exp(i/100))).item())
    for i in test_output:
        test_output_list.append(torch.argmax(torch.exp(i/100)/torch.sum(torch.exp(i/100))).item())
    train_mat = confusion_matrix(labels_train, train_output_list)
    test_mat = confusion_matrix(labels_test, test_output_list)

def visualize_weights():
    '''
    Loads the CAFENet trained in plot_cafe_loss, maps the weights to the grayscale
    range, reshapes the weights into the original CAFE image dimensions, and
    plots the weights, displaying the six weight tensors corresponding to the six
    labels.
    '''
    model = torch.load('HW2Q3')
    w, b = model.parameters()
    min_weight = torch.min(w)
    max_weight = torch.max(w)
    w = (w - min_weight)*255/(max_weight-min_weight)
    w = w.view(6,380,240)
    label = ['anger', 'disgusted', 'happy', 'maudlin', 'fear', 'surprise']
    plt.figure(figsize=(8, 8))
    for i in range(6):
        plt.figure()
        plt.imshow(w[i].detach().numpy(), cmap = 'gray')
        plt.title(label[i])

