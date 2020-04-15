#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


# ## READ CSV FILE

# In[3]:


data = pd.read_csv('C:\\Users\\Jerome Kene\\Downloads\\ElectronicsProductData.csv')


# In[18]:


data.head(10)


# ## SELECT FEATURES

# In[22]:


data.columns


# In[38]:


data['categories']


# In[63]:


data1 = data[['reviews.username', 'id','reviews.rating']]


# In[64]:


data1.head()


# In[65]:


data1.shape


# In[66]:


data1 = data1.dropna()


# In[67]:


data1.describe()


# In[68]:


data1['rating'] = data1['reviews.rating']
data1['user_id'] = data1['reviews.username']
data1['product_id'] = data1['id']
data1.drop(['id','reviews.username','reviews.rating'], axis=1, inplace=True)


# In[72]:


data1 = data1[['user_id', 'product_id', 'rating']]
data1.head()


# In[73]:


print('Minimum rating is: %d' %(data1.rating.min()))
print('Maximum rating is: %d' %(data1.rating.max()))


# In[75]:


popular_products = pd.DataFrame(data1.groupby('product_id')['rating'].count())
most_popular = popular_products.sort_values('rating', ascending=False)
most_popular.head(10)


# In[77]:


most_popular.head(20).plot(kind='bar')
sns.set()


# In[82]:


#CHECK DISTRIBUTION OF RATINGS

with sns.axes_style('darkgrid'):
    g = sns.catplot('rating',data=data1, aspect=2.0, kind='count')


# In[170]:


data2 = data1.copy()
data2.head()


# In[ ]:





# In[164]:


no_of_rated_products_per_user = data1.groupby(by='user_id')['rating'].count().sort_values(ascending=False)

no_of_rated_products_per_user.head()


# In[ ]:





# In[176]:


#AVERAGE RATING OF PRODUCTS

data2.groupby('product_id')['rating'].mean().head()


# ## Product popularity based recommendation - Part I

# In[178]:


#TOTAL NUMBER OF RATINGS PER PRODUCT

data2.groupby('product_id')['rating'].count().sort_values(ascending=False).head()


# In[183]:


popular_products = pd.DataFrame(data2.groupby('product_id')['rating'].count())
most_popular = popular_products.sort_values('rating', ascending=False)

most_popular.head(30).plot(kind = "bar",figsize=(10,6))


# ## Model-based collaborative filtering system - Part II

# In[187]:


utility_matrix = data2.pivot_table(values='rating', index='user_id', columns='product_id',fill_value=0)

utility_matrix.head()


# In[188]:


utility_matrix.shape


# In[189]:


X = utility_matrix.T
X.head()


# In[191]:


X.shape


# In[192]:


X1 = X


# In[193]:


#DIMENSIONALITY REDUCTION

from sklearn.decomposition import TruncatedSVD
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape


# In[194]:


#CORRELATION MATRIX

correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape


# In[203]:


#PICKING A RANDOM PRODUCT

X.index[23]


# In[ ]:





# In[204]:


i = 'AVpg59zyilAPnD_xyv3y'

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID


# In[ ]:





# In[205]:


correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID


# In[ ]:





# In[206]:


Recommend = list(X.index[correlation_product_ID > 0.50])
Recommend.remove(i)

Recommend[0:24]


# In[ ]:




