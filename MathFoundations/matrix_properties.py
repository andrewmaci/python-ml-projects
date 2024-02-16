import numpy as np
import torch
import matplotlib
def frobenius_norm():
    '''
    ||X||_F  = sqrt of sum of each (element of matrix )^2
    Analogous to L2 norm there fore ||X||F measures
    the size of matrix in terms of Euclidean distance
    '''
    X = np.array([[1,2],[3,4]])
    # (1**2+2**2+3**2+4**2)**(1/2)
    print(np.linalg.norm(X))

    ##this operation with tensor and pytorch requires floats
    X_pt = torch.tensor([[1,2],[3,4.]])
    print(torch.norm(X_pt))

frobenius_norm()

def matrix_multiplication():
    '''
        C_i,k = Sum A_i,j*B_j,k. To perform the multiplication
        the number of columns in the first matrix must be euqal 
        to the number of rows in the second matrix
    '''
    A = np.array([[3,4],[5,6],[7,8]])
    b = np.array([1,2])
    print(np.dot(A,b))
    A_pt = torch.tensor([[3,4],[5,6],[7,8]])
    b_pt = torch.tensor([1,2])
    print(torch.matmul(A_pt,b_pt))
    C= np.array([[1,9],[2,0]])
    C_pt= torch.tensor([[1,9],[2,0]])
    C_pt_v2 = torch.tensor([[1,2],[9,0]]).T
    print(np.dot(A,C))
    print(torch.matmul(A_pt,C_pt))
    print(torch.matmul(A_pt,C_pt_v2))
matrix_multiplication()

def symetric_matrices():
    '''
    Special case of matrix in which:
     1. Matrix is square
     2. X^T = X (mirrored over diagonal)
    '''
    X = np.array([[0,1,2],[1,7,8],[2,8,9]])
    print(X)
    print(X==X.T)

    '''
    Identity matrix, diagonal is 1 and everything is 0. Multiplying
    vector by any a identity matrix of the same length returns the original vector.
    '''
    I_pt = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
    x_pt=torch.tensor([1,2,3])
    print(torch.matmul(I_pt,x_pt))
symetric_matrices()

def matrix_inversion():
    '''
    Used for solving linear equations
    Matrix inverse of X is denoted as X^-1
    If we multiply inverse by non inversed matrix we get Identity matrix
    - X^-1*X = I

    Can be only calculated if matrix isn't singular
     - all matrix must be linearly independent (we cant solve for expressions
     that don't have a solution)
      - also matrix must be square
    '''
    X = np.array([[4,2],[-5,3]])
    y = np.array([4,-7])
    X_inv = np.linalg.inv(X)
    print(X_inv)
    w = np.dot(X_inv,y)
    print(w)
    print(np.dot(X,w))

    X_inv_pt = torch.inverse(torch.tensor([[4,2],[-5,-3.]]))
    print(X_inv_pt)
matrix_inversion()

def diagonal_matrices():
    '''
    Diagonal matrices have nonzero elements along main diagonal and zeros everywhere lese
    diag(x) where x is vector of main diagonal elements
    really computationally efficient:
        - diag(x)y = equal to hadamard product
        - inversion of diag(x)^-1 = diag[1/x,...,1/n]^T
            -  if has zeros inversion is not possible
        - we can have them non square bu just adding or removing zeros
    '''

def orthogonal_matrices():
    '''
    An orthogonal matrix, orthonormal vectors make up all rows and columns
    This means that A^TA == AA^T = I
    and also A^T = A^-1I= A^-1
    therefore AT and A-1 calculatio is cheap
    '''
    I=torch.eye(3)
    print(torch.dot(I[:,0],I[:,1]))
    print(torch.dot(I[:,0],I[:,2]))
    print(torch.norm(I[:,0]))


orthogonal_matrices()