
# coding: utf-8

# # K-means

# In[41]:


# necessary imports
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import secrets


# ### Generate data

# In[42]:


# Generate Data
points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
              (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
              (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))


# ### Main functions

# In[71]:


def initialize_centroids(points, k):
    '''
        Selects k random points as initial
        points from dataset
    '''
    #centroids=[]
    #for i in range(0,3):
    #centroids.append(secrets.choice(points))

    #return centroids
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


# In[90]:


def closest_centroid(points, centroids):
    '''
        Returns an array containing the index to the nearest centroid for each point
    '''
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis = 2))
    return np.argmin(distances, axis = 0)


# In[91]:


def move_centroids(points, closest, centroids):
    '''
        Returns the new centroids assigned from the points closest to them
    '''
    return np.array([points[closest == k].mean(axis = 0) for k in range(centroids.shape[0])])


# In[92]:


def main(points):
    num_iterations = 100
    k = 3
        
    # Initialize centroids
    centroids = initialize_centroids(points, 3)
    
    # Run iterative process
    for i in range(num_iterations):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)
    
    return centroids


# In[93]:


centroids = initialize_centroids(points, 3)
centroids


# In[94]:


centroids = main(points)


# In[95]:


centroids = initialize_centroids(points, 3)

plt.scatter(points[:, 0], points[:, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
ax = plt.gca()


# In[99]:


closest = closest_centroid(points, centroids)
centroids = move_centroids(points, closest, centroids)

plt.scatter(points[:, 0], points[:, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
ax = plt.gca()


# In[1]:


centroids

