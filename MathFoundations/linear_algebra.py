import numpy as np
import matplotlib.pyplot as plt
import torch 
def intro_function():
    
    ##define the time space [start,stop, amount of points]
    t= np.linspace(0,50,1000)
    
    equation_1 = 2.5*t
    equation_2 = 3*(t-5)

    fig, ax = plt.subplots()
    plt.xlabel('time[m]')
    plt.ylabel('distance[km]')
    ax.set_xlim([0,40])
    ax.set_ylim([0,100])
    ax.plot(t,equation_1,c='blue')
    ax.plot(t,equation_2,c='red')
    plt.axvline(x=30,color='purple',linestyle='--')
    plt.axhline(y=75,color='purple',linestyle='--')
    plt.show()

##vectors and vectors transposition
def vectors_aka_rank_1_tensor():
    '''Vectors can represet either a point in dimensions
    or a 'norm' which is a function that quantifies vector magnitude 
    
    L^2 Norm = We square each element of the vector add them to each other and take the root.
    ||x||_2 - Measures simple (Euclidean) distance from origin of a vector
    '''
    x = np.array([12,24,4])
    print(x)
    print(len(x))
    print(x.shape)
    ##Vector transposition
    #no difference beause its initialized as one dimension
    x_T = x.transpose()
    print(x_T.shape)
    ##When we initialize with more dimensions now we can work with something
    y=np.array([[12,24,4]])
    print(y)
    print(y.T)
    ##Zero vectors - vectors with only zeros
    z = np.zeros(3)
    ##Calculating the L2norm
    print((12**2+24**2+4**2)**(1/2))
    np.linalg.norm(x)
    ## Unit vectors are special vectors whose L2 norm is equal to 1
    ##L1 and L2 norms are used to regularize objective functions
    '''
    - Basis vectors can be scaled to represent any vector in a given vector space
        eg. i[0,1],j[1,0] -> v[1.5i,2j]
    - Orthogonal Vectors, if we take the 'dot product' x^T*y = 0 Transposed vector x times y = 0
        assuming their length(their norm) is non-zero they are at 90degree angle to each other
        n-dimensional spaces has maximum of n mutually orthogonal vectors
        Orthonormal vectors are orthogonal and all have unit norm (L2=1)
    '''
    i = np.array([1,0])
    j = np.array([0,1])
    print(np.dot(i,j))

def matrices_aka_rank_2_tensors():
    '''
    Matrices are simply two dimensional array of numbers
    - are denoted with uppercase italics bolds
    - represented as (n_row,n_col)
    '''
    X = np.array([[12,24],[2,5],[21,33]])
    print(X.shape)
    print(X.size)
    print(X)
    print('----------')
    print(X[:,0]) ##Left column
    print(X[1,0]) ##second row , left column
    print(X[1,:]) ##second(middle) row
    print(X[0:2,0:2])

    X_pt = torch.tensor([[12,24],[2,5],[21,33]])
    print(X_pt)
    print(X_pt.shape)
    '''
    Generic tensor notation Generic tensors are notated as matrices but they use
    sans serif font.

    4 Rank tendors are common for images where each dimension:
        1. number of images in traning batch
        2. Image height in pixels
        3. imgae width in pixeld
        4. Number of color channels 
    '''
    # images = torch.zeros([16,22,19,1])
    # print(images)

matrices_aka_rank_2_tensors()