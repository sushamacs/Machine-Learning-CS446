import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import random
import sklearn.datasets



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block. It is the
                          parameter C in the handout hw3.pdf.
        """
        super(Block, self).__init__()
        """
        Write your code here.
        """
        self.num_channels = num_channels
        self.conv2d = nn.Conv2d(self.num_channels, self.num_channels, kernel_size = 3, stride=1, padding=1,bias = False)
        self.batchnorm2d = nn.BatchNorm2d (num_features = num_channels)
        self.relu = nn.ReLU()
        self.conv2d2 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size = 3, stride=1, padding=1,bias = False) 
        self.batchnorm2d2 = nn.BatchNorm2d(num_features = num_channels) 
        

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        out = self.conv2d(x)
        out = self.batchnorm2d(out)
        out = self.relu(out)
        out = self.conv2d2(out)
        out = self.batchnorm2d2(out)
        out = out + x
        out = self.relu(out)
        return out  
        
        # block returns sigma(x+f(x))
        pass


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.resconv2d = nn.Conv2d(1, self.num_channels, kernel_size = 3, stride=2, padding=1,bias = False)
        self.resbatchnorm2d = torch.nn.BatchNorm2d (num_features = num_channels)
        self.resrelu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.block1 = Block(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.num_channels, self.num_classes)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
       
        out1 = self.resconv2d(x)
        out1 = self.resbatchnorm2d(out1)
        out1 = self.resrelu(out1)
        out1 = self.maxpool(out1)
        out1 = self.block1(out1)
        out1 = self.avgpool(out1)
        out1 = out1.view(out1.shape[0], -1)
        out1 = self.linear(out1)
        return out1
        
        
def plot_resnet_loss_1():
    """
    Train ResNet with different parameters C on digits data and draw the training
    error vs the test error curve. To make life easier, we provide you with the
    starter code to load the digits data and draw the figures with different
    parameters C. You do not need to modify the starter code and you only need
    to implement the training part. Train your algorithms for 4000 epochs with mini
    batch size = 128 and step size 0.1.
    """
    
    sk_digits = sklearn.datasets.load_digits()
    (X, Y) = (torch.tensor(sk_digits.data).type(torch.float), torch.tensor(sk_digits.target))
    Y = Y.type(torch.LongTensor)
    print(X.shape, Y.shape, X.max(), X.min())
    X /= X.max()
    n = X.shape[0]
    perm = list(range(n))
    random.shuffle(perm)
    (X, Y) = ({'tr': X[perm[:n//2], ...], 'te': X[perm[n//2:], ...]}, {'tr':Y[perm[:n//2]], 'te':Y[perm[n//2:]]})
    mb_sz = 128
    stepsize = 0.1
    

    for (_, (net_s, num_channels)) in enumerate([
        ('ResNet_1', 1),
        ('ResNet_2', 2),
        ('ResNet_4', 4),
    ]):
        losses = { 'tr' : [], 'te' : [] }
        net =  ResNet(num_channels)
        l = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr = stepsize)
        for i in range(4000):
            idxs = random.sample(range(X['tr'].shape[0]), mb_sz)
            (x, y) = (X['tr'][idxs, ...], Y['tr'][idxs])
            x = x.view(x.shape[0], 1, 8, 8)
            
            """
            Write your code here.
            """

            y_pred = net.forward(x)
            loss = l(y_pred,y)
                
            with torch.no_grad():
                if (i + 1) % 25 == 0:
                    x = X['te']
                    x = x.view(x.shape[0], 1, 8, 8)
                    yhat2 = net(x)
                    loss2 = torch.nn.CrossEntropyLoss()(yhat2, Y['te'])
                    print(f"{i} {loss:.3f} {loss2:.3f}")
                    losses['tr'].append(loss.detach())
                    losses['te'].append(loss2.detach())
                loss.backward() 
                optimizer.step()
                optimizer.zero_grad()
  
        for s in ['tr', 'te']:
            plt.figure(1)
            plt.plot(range(len(losses[s])), losses[s],
                     label = f"{net_s} {s}")
                     
    plt.figure(1)
    plt.title("risk curves")
    plt.legend()
    plt.savefig('f1.pdf')
    plt.show()        


def plot_resnet_loss_2():
    """
    Train ResNet with parameter C = 64 on digits data and draw the training
    error vs the test error curve. To make life easier, we provide you with the
    starter code to load the digits data and draw the figures with C = 64.
    You do not need to modify the starter code and you only need
    to implement the training part. Train your algorithms for 4000 epochs with mini
    batch size = 128 and step size 0.1.
    """
    
    sk_digits = sklearn.datasets.load_digits()
    (X, Y) = (torch.tensor(sk_digits.data).type(torch.float), torch.tensor(sk_digits.target))
    Y = Y.type(torch.LongTensor)
    print(X.shape, Y.shape, X.max(), X.min())
    X /= X.max()
    n = X.shape[0]
    perm = list(range(n))
    random.shuffle(perm)
    (X, Y) = ({'tr': X[perm[:n//2], ...], 'te': X[perm[n//2:], ...]}, {'tr':Y[perm[:n//2]], 'te':Y[perm[n//2:]]})
    mb_sz = 128
    stepsize = 0.1

    for (_, (net_s, num_channels)) in enumerate([
        ('ResNet_64', 64),
    ]):
        losses = { 'tr' : [], 'te' : [] }
        net =  ResNet(num_channels)
        l = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr = stepsize)

        for i in range(4000):
            idxs = random.sample(range(X['tr'].shape[0]), mb_sz)
            (x, y) = (X['tr'][idxs, ...], Y['tr'][idxs])
            x = x.view(x.shape[0], 1, 8, 8)
            
            """
            Write your code here.
            """
            y_pred = net.forward(x)
            loss = l(y_pred,y)
            
            with torch.no_grad():
                if (i + 1) % 25 == 0:
                    x = X['te']
                    x = x.view(x.shape[0], 1, 8, 8)
                    yhat2 = net(x)
                    loss2 = torch.nn.CrossEntropyLoss()(yhat2, Y['te'])
                    print(f"{i} {loss:.3f} {loss2:.3f}")
                    losses['tr'].append(loss.detach())
                    losses['te'].append(loss2.detach())
                loss.backward() 
                optimizer.step()
                optimizer.zero_grad()
        for s in ['tr', 'te']:
            plt.figure(1)
            plt.plot(range(len(losses[s])), losses[s],
                     label = f"{net_s} {s}")
                     
    plt.figure(1)
    plt.title("risk curves")
    plt.legend()
    plt.savefig('f2.pdf')
    plt.show()