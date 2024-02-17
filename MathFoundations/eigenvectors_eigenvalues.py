import numpy as np
import matplotlib.pyplot as plt
import torch 

def plot_vectors(vectors,colors):
    plt.figure()
    plt.axvline(x=0,color='lightgray')
    plt.axhline(y=0,color='lightgray')

    for i in range (len(vectors)):
        x = np.concatenate([[0,0],vectors[i]])
        plt.quiver([x[0]],[x[1]],[x[2]],[x[3]],
                   angles='xy',scale_units='xy',scale=1,color=colors[i])
        
v = np.array([1,4])
def affine_transformations():
    '''
    Flipping a matrix is an example  of an affine transformation
    a change in geometry that may adjust distances or angles between
    vectors but preserves parallelism between them.

    '''
    # plot_vectors([v],['lightgreen'])
    # plt.xlim(-1,5)
    # plt.ylim(-1,5)
    # plt.show()

    I = np.array([[1,0],[0,1]])
    Iv = np.dot(I,v)
    print(v==Iv)

    E=np.array([[1,0],[0,-1]]) ##Flips vectors across X axis
    Ev=np.dot(E,v)

    # plot_vectors([v,Ev],['lightgreen','green'])
    # plt.xlim(-1,5)
    # plt.ylim(-3,3)
    # plt.show()

    F=np.array([[-1,0],[0,1]]) ##Flips vectors across y axis
    Fv=np.dot(F,v)

    # plot_vectors([v,Fv],['lightgreen','green'])
    # plt.xlim(-4,4)
    # plt.ylim(-1,5)
    # plt.show()


def multiple_afine_transformations():
    '''
    We can concatenate several vectors together into a martix where each column
    is a separate vector. Then whaever linear transofmrations are
    applied to the matrix will be indepedently applied to each column/vector
    '''
    v1= np.array([-3,1])
    v2= np.array([-1,1])
    I = np.array([[1,0],[0,1]])

    V = np.concatenate((np.matrix(v).T,
                        np.matrix(v1).T,
                        np.matrix(v2).T,),
                        axis=1)
    print(V)
    print(np.dot(I,V)) 
