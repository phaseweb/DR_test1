
# coding: utf-8

# In[10]:


from sklearn import datasets
import numpy as np


# In[17]:


x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], 300)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], 300)

data = np.vstack((x1, x2)).astype(np.float32)
labels = np.hstack((np.zeros(300),
                              np.ones(300)))


# In[1]:


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


# In[2]:


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll


# In[25]:


def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
       
        
    return weights


# In[26]:


weights = logistic_regression(data, labels,
                     num_steps = 300000, learning_rate = 5e-5, add_intercept=True)


# In[27]:


weights


# In[24]:


#Comparison with sklearn
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=True, C = 1e15)
clf.fit(data, labels)

print (weights)

