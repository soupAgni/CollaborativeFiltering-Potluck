
# coding: utf-8

# In[1]:

from __future__ import division
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:

df = pd.read_csv('formatted_data_updated.csv')

df = df.replace('?', np.nan)
#print df
newdf = df.drop('UserID', axis = 1)
newdf


# In[3]:

training_set = newdf[0:31]

test_set = newdf[31:37]


# Adding all of the ratings corresponding to each recipe in a list of lists to make computation easier.

# In[4]:

def makeArrays(dataframe): 
    recipeRatings = []
    #print recipeRatings
    for col in dataframe:
        lists = []
        for val in dataframe[col]:
            lists.append(val) 
        recipeRatings.append(lists)
    #print recipeRatings
    return recipeRatings


# def averages(array):
#     i = 0
#     for ratings in array:
#         sums = 0
#         count = 1
#         for rating in ratings:
# 
#             if(not math.isnan(rating)):
#                 sums  = sums + rating
#                 count = count +1
#             average = float(sums/count)
#             #print average
#             array[i] = [average if math.isnan(x) else float(x) for x in array[i]]
#         i = i+1
# 
#     #returns a numpy array
#     return np.array(array)

# In[5]:

#compute the averages of the row
def averages(row):
    #instantiate our variables
    avg = float(0)
    counter = float(1)

    #iterate over the cols in the row
    for col in row:
        if not np.isnan(float(col)):
            avg += float(col)
            counter += 1
    #compute the average
    return (avg/counter)


# In[6]:

#initlize list of lists for spectral
training_data = []

for index, row in newdf.iterrows():
    #preprocessing for every row
    avg = averages(row)
    #initlize rows
    rows = []
    #for each col in row
    for col in row:
        #cast to float for nan behavior
        col = float(col)
        #if missing data
        if np.isnan(col):
            rows.append(avg)
        #not missing data
        else:
            rows.append(col)

    training_data.append(rows)

#cast to np array for fun times
training_data = np.array(training_data)


# # K-means

# #run PCA to reduce the data dimensionality so that it is easier to visualize
# from sklearn.decomposition import PCA
# reduced_data = PCA(n_components=2).fit_transform(training_data)

# In[7]:

#training_data = averages(makeArrays(training_set.T))
#print training_set.shape
kmeans = KMeans(n_clusters=4, random_state=0).fit(training_data)


# In[8]:

labels_kmeans = kmeans.labels_
set_lk = set(labels_kmeans)
print set_lk
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.show()


# test_data = averages(makeArrays(test_set.T))
# kmeans.predict(test_data)
# kmeans.score(test_data)
# 

# # Spectral clustering

# In[9]:

spectral = SpectralClustering()
spectral.fit(training_data)
print "lables from clustering"
print spectral.labels_


# # Hierarchial/Agglomerative Clustering

# In[10]:

# Define the structure A of the data. Here a 10 nearest neighbors
from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(training_data, n_neighbors=10, include_self=False)


# In[11]:

ward = AgglomerativeClustering(n_clusters=8, connectivity=connectivity,
                               linkage='ward').fit(training_data)


# In[12]:

label = ward.labels_
print label


# In[13]:

# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(training_data[label == l, 0], training_data[label == l, 1], training_data[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')

plt.show()


# In[14]:

#testing
labels_dict = {}
ind = 0

for i in range(0, len(set_lk)):
    labels_dict[i] = []
#print len(set_lk)
#print labels_dict
for i in labels_kmeans:
    #if i not in labels_dict.values:
    #print i
    labels_dict[i].append(ind)
    ind = ind + 1
   
print labels_dict
    


# In[15]:

colNames =  list(newdf.columns.values)
#print colNames
def findEle(index):
    #get all the colums corresponding to the particular row
    #colVals = df.loc[[index]]
    toRet = []
    likes = []
    dislikes = []
    for col in colNames:
        if not np.isnan(float(newdf.loc[[index]][col])):
            #print np.array(newdf.loc[[index]][col].astype(list))[1]
            for i in np.array(newdf.loc[[index]][col].astype(list)):
                #print type(i)
                if i == "1":
                    likes.append(col)
                else:
                    dislikes.append(col)
            
            #toRet.append(col)
    #print "likes: ", likes
    #print "dislikes: ", dislikes
    return (likes, dislikes)
    


# TF-IDF

# In[16]:

#compute tf-idf for the given array
#in Scikit-Learn
def tfIDF(array):
    sklearn_tfidf = TfidfVectorizer()#(stop_words = 'english')
    vec_representation = sklearn_tfidf.fit_transform(array)
    print vec_representation
    return vec_representation


# Cosine similarity

# In[17]:

#compute cosine similarity value for the given 
def cosine_sim(tf_idf_matrix):
    cosine_similarity()
    


# In[21]:

#users = []
for key in labels_dict:
    for vals in labels_dict[key]:
        likes, dislikes  = findEle(vals)
        #print "key is ", key
        for like in likes:
            if not len(likes) == 0:
                tfIDF(likes)
                #print likes
        #print likes
        
        
        #print likes
        #print dislikes


# In[ ]:




# In[ ]:



