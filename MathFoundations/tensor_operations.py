import numpy as np
import torch
import matplotlib.pyplot as plt
'''
Tensor transposition:
    1. Transpoe of a scalar is itslf x^T=x
    2. Transpoe of a vector is the conversion from column to row and vice versa
    3. Scalar and vector transposition are special cases of matrix transposition
        - during this operation we flip the axis over main diagonal
        - (X^T)_ij = X_ji
'''
x=np.array([1,2,3])
x_pt=torch.tensor([1,2,3])
X = np.array([[1,2],[3,4],[5,6]])
X_pt = torch.tensor([[1,2],[3,4],[5,6]])

def tensor_transposition():
    print(X,'\n','----','\n',X.T)
    print('----------')
    print(X_pt,'\n','----','\n',X_pt.T)

def tensor_arithmetic():
    print(X+2)
    print(X*2)
    print(X*2+2)

    ##In reality behind scene we run torch.mul() and torch.add()
    print(X_pt*2+2)
    print(X_pt*2+2)
    print(X_pt*2+2)
    print(torch.mul(X_pt,2))

def hadamard_product():
    '''
    If tensors have the same size, operations are often by default applied elemen-wise
    It is worth noting that this is not matrix multiplication and 
    is called Hadamard product or 'element-wise product'
    '''
    A=X+2
    print(A)
    print(A+X)
    ##Hadamar product
    print(A*X)
    
    A_pt = X_pt+2
    print(A_pt)
    print(X_pt)
    print(A_pt+X_pt)
    print(A_pt*X_pt)

def tensor_reduction():
    '''
    Action of summing all of the elements in the tensor
    '''
    print(X)
    print(X.sum())
    print(X_pt)
    print(torch.sum(X_pt))
    # we can perfom reduction on specific axis
    print(X.sum(axis=0))
    print(X.sum(axis=1))
    print(torch.sum(X_pt,0))
    print(torch.sum(X_pt,1))
    '''
    Other operations can be applied with reduction:
        1. maximum
        2. minimum
        3. mean
        4. product
    '''
def dot_product():
    '''
    When we have two vectors with the same length we can calculate the
    'dot product' between them (x^Ty). x^Ty = Sum from i=1 to n of (x_i*y_i)
    dot product is really important in deep learning where it is performed at every neuron

    '''
    y = np.array([0,1,2])
    y_pt = torch.tensor([0,1,2])

    print(x,y)
    print(np.dot(x,y))
    print('------------')
    print(x_pt,y_pt)
    print(torch.dot(x_pt,y_pt))
dot_product()

def linear_systems_with_substiution():
    '''
        1. Substitution used whenever we have a varaiable with coeficient of 1
            eg.  y=3x
            -5x+2y=2
            we can substitue y with 3x in the second equation
    '''
    fig, ax = plt.subplots()
    x= np.linspace(-10,10,1000)
    
    equation_1 = 3*x
    equation_2 = (5*x/2)+1

    ax.set_xlim([0,3])
    ax.set_ylim([0,8])
    ax.plot(x,equation_1,c='blue')
    ax.plot(x,equation_2,c='red')
    plt.axvline(x=2,color='purple',linestyle='--')
    plt.axhline(y=6,color='purple',linestyle='--')
    plt.show()

def linear_systems_with_elimination():
    '''
        1. Substitution used whenever there is no coeficient 1 variables
            eg.  2x-3y=15
                4x+10y=14
            by multipying the first equation by -2 we will be able sum and solve
    '''
    fig, ax = plt.subplots()
    x= np.linspace(-10,10,1000)
    
    equation_1 = (2*x/3)-5
    equation_2 = (-2*x/5)+(7/5)

    ax.set_xlim([-2,10])
    ax.set_ylim([-6,4])
    ax.plot(x,equation_1,c='blue')
    ax.plot(x,equation_2,c='red')
    plt.axvline(x=6,color='purple',linestyle='--')
    plt.axhline(y=-1,color='purple',linestyle='--')
    plt.show()

linear_systems_with_elimination()