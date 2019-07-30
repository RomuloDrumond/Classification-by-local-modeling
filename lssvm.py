import numpy as np
from numpy import dot, exp
from functools import partial 
from scipy.spatial.distance import cdist

class LSSVM:
    'Class the implements the Least-Squares Support Vector machine.'
    
    def __init__(self, gamma=1, kernel='rbf', **kernel_params): 
        self.gamma = gamma
        
        self.x        = None
        self.y        = None
        self.y_labels = None
        
        self.alpha = None
        self.b     = None
        
        self.kernel = LSSVM.get_kernel(kernel, **kernel_params) # saving kernel function
        
           
    @staticmethod
    def get_kernel(name, **params):
        
        def linear(x_i, x_j):                           
            return dot(x_i, x_j.T)
        
        def poly(x_i, x_j, d=params.get('d',3)):        
            return ( dot(x_i, x_j.T) + 1 )**d
        
        def rbf(x_i, x_j, sigma=params.get('sigma',1)):
            if x_i.ndim==x_i.ndim and x_i.ndim==2: # both matrices
                return exp( -cdist(x_i,x_j)**2 / sigma**2 )
#             return exp( -( dot(x_i,x_i.T) + dot(x_j,x_j.T)- 2*dot(x_i,x_j) ) / sigma**2 )
#             temp = x_i.T - X
#             return exp( -dot(temp.temp) / sigma**2 )
                
        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
                
        if kernels.get(name) is None: 
            raise KeyError("Kernel '{}' is not defined, try one of the list: {}.".format(
                name, list(kernels.keys())))
        else: return kernels[name]
        
    
    def opt_params(self, X, y_values):
        sigma = np.multiply( y_values*y_values.T, self.kernel(X,X) )

        A_cross = np.linalg.pinv(np.block([
            [0,                      y_values.T                   ],
            [y_values,   sigma + self.gamma**-1 * np.eye(len(y_values))]
        ]))

        B = np.array([0]+[1]*len(y_values))

        solution = dot(A_cross, B)
        b     = solution[0]
        alpha = solution[1:]
        
        return (b, alpha)
            
    
    def fit(self, X, Y, verboses=0):
        self.x = X
        self.y = Y
        self.y_labels = np.unique(Y, axis=0)
        
        if len(self.y_labels)==2: # binary classification
            # converting to -1/+1
            y_values = np.where(
                (Y == self.y_labels[0]).all(axis=1)
                ,-1,+1)[:,np.newaxis] # making it a column vector
            
            self.b, self.alpha = self.opt_params(X, y_values)
        
        else: # multiclass classification
              # ONE-VS-ALL APPROACH
            n_classes = len(self.y_labels)
            self.b      = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(Y)))
            for i in range(len(self.y_labels)):
                # converting to +1 for the desired class and -1 for all other classes
                y_values = np.where(
                    (Y == self.y_labels[i]).all(axis=1)
                    ,+1,-1)[:,np.newaxis] # making it a column vector
  
                self.b[i], self.alpha[i] = self.opt_params(X, y_values)

        
    def predict(self, X):
        k = self.kernel(self.x, X)
        
        if len(self.y_labels)==2: # binary classification
            y_values = np.where(self.y==self.y_labels[0],-1,1)
            Y = np.sign( dot( np.multiply(self.alpha, y_values.flatten()), k ) + self.b)
            
            y_pred_labels = np.where(Y==-1,        self.y_labels[0],
                                     np.where(Y==1,self.y_labels[1],Y))
        
        else: # multiclass classification, ONE-VS-ALL APPROACH
            Y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where((self.y == self.y_labels[i]).all(axis=1),
                                         +1, -1)[:,np.newaxis] # making it a column vector
                Y[i] = np.sign( dot( np.multiply(self.alpha[i], y_values.flatten()), k ) + self.b[i])
            
            predictions = np.argmax(Y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
            
        return y_pred_labels
        