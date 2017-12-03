
# coding: utf-8

# In[36]:


import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
import sklearn as sk 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[37]:


class LinReg:
    
    
    def __init__(self):
        self.W = [] # model's weights
        self.fscaling = False # is feature scaling used
        
    def cost(self, y_real, y_pred): 
        # cost function for gradient descent algorithm
        return np.sum((y_pred-y_real)**2)/(len(y_real))
    
    def gradient_descent_step(self, learning_rate, dy, m, n, X_tr):
        # one gradient descent step
        s = (np.dot(dy.T, X_tr)).reshape(n, 1)
        dW = 2*(learning_rate*s/m).reshape(n, 1)
        return self.W - dW
    
    def normalize(self, X):
        # normalize X table
        for j in range(X.shape[1]):
            X[:,j] = X[:,j]/np.max(X[:,j])
        return X
    
    def fit(self, X, y, learning_rate = 0.99, nsteps = 3000, e = 0.000000001,
            weight_low = 0, weight_high = 1,
            fscaling = False, kweigths = 1, random_state = 0):
        # train our Linear Regression model
        
        np.random.seed(random_state)
        X = X.astype(float)
        
        # Normalize process
        if fscaling == True:
            X = self.normalize(X)
            self.fscaling = True
        m = X.shape[0]
        # add one's column to X
        X = np.hstack( (np.ones(m).reshape(m, 1), X) )
        n = X.shape[1]
        
        # Weights: random initialization
        self.W = np.random.randint(low = weight_low, high = weight_high, size=(n, 1))
            
        y_pred = np.dot(X, self.W)
        cost0 = self.cost(y, y_pred)
        y = y.reshape(m, 1)
        k = 0
        
        ########## Gradient descent's steps #########
        while True:
            dy = y_pred - y
            W_tmp = self.W
            self.W = self.gradient_descent_step(learning_rate, dy, m, n, X)
            y_pred = np.dot(X, self.W)
            cost1 = self.cost(y, y_pred)
            k += 1
            if (cost1 > cost0):
                self.W = W_tmp
                break    
                
            if ((cost0 - cost1) < e) or (k == nsteps):
                break
                
            cost0 = cost1
        #############################################
        return self.W # return model's weights
    
    def predict(self, X):
        m = X.shape[0]
        return np.dot( np.hstack( (np.ones(m).reshape(m, 1),
                                       self.normalize(X.astype(float))) ),
                          self.W)


# In[38]:


#train = pd.read_csv('train_DR.csv')
#test = pd.read_csv('test_DR.csv')


# In[39]:


#train.drop('Unnamed: 0', axis = 1, inplace = True)
#test.drop('Unnamed: 0', axis = 1, inplace = True)


# In[40]:


#labels = train['Survived']
#train.drop('Survived', axis = 1, inplace = True)


# In[41]:


from sklearn import datasets
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(X, y)


# In[52]:


#Custom model
lr = LinReg()
lr.fit(x_train, y_train, learning_rate = 0.997, random_state = 0, weight_low = -500, weight_high = 500, nsteps=3000)
xx = [i for i in range(x_train.shape[0])]
y1 = lr.predict(x_test)
print ('My LR model:')
f=0
t=40
plt.plot(xx[f:t], y_test[f:t], color='r', linewidth=4, label='y')
plt.plot(xx[f:t], y1[f:t], color='b', linewidth=2, label='predicted y')
plt.ylabel('Target label')
plt.xlabel('Line number in dataset')
plt.legend(loc=4)
plt.show()


# In[50]:


lr2 = LinearRegression()
lr2.fit(x_train,y_train)
y2 = lr2.predict(x_test)
plt.figure()
plt.plot(xx[f:t], y_test[f:t], color='r', linewidth=4, label='y')
plt.plot(xx[f:t], y2[f:t], color='b', linewidth=2, label='predicted y')
plt.ylabel('Target label')
plt.xlabel('Line number in dataset')
plt.legend(loc=4)
plt.show()

