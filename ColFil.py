
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
import time
from scipy import spatial
import tfidf_similarity_computations as tsc


# In[2]:

#df = pd.read_csv('formatted_data_updated.csv')
df = pd.read_csv('formatted_data_ing.csv')
df = df.replace('?', np.nan)
#print df
newdf = df.drop('UserID', axis = 1)
#newdf


# In[3]:

#storing the recipe ratings in a list of lists
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


# In[4]:

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


# In[5]:


#initlize list of lists for clusering
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


# # K means

# In[6]:

#training_data = averages(makeArrays(training_set.T))
#print training_set.shape
kmeans = KMeans(n_clusters=6, random_state=0).fit(training_data)


# In[7]:

labels_kmeans = kmeans.labels_
set_lk = set(labels_kmeans)
print labels_kmeans
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
#plt.show()


# # Spectral Clustering

# In[8]:

spectral = SpectralClustering()
spectral.fit(training_data)
print "lables from clustering"
spectral_labels =  spectral.labels_
set_ls = set(spectral_labels)
print spectral_labels


# # Hierarchial/Agglomerative Clustering

# In[9]:

# Define the structure A of the data. Here a 10 nearest neighbors
from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(training_data, n_neighbors=10, include_self=False)


# In[10]:

ward = AgglomerativeClustering(n_clusters=8, connectivity=connectivity,
                               linkage='ward').fit(training_data)


# In[11]:

h_labels = ward.labels_
print h_labels
set_lh = set(h_labels)


# In[12]:


# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(h_labels):
    ax.scatter(training_data[h_labels == l, 0], training_data[h_labels == l, 1], training_data[h_labels == l, 2],
               color=plt.cm.jet(float(l) / np.max(h_labels + 1)),
               s=20, edgecolor='k')

plt.show()


# In[13]:

colNames =  list(newdf.columns.values)
#print colNames


# #get the labels dictionary
# labels_dict = tsc.getLabelsDict(set_lk, labels_kmeans)
# #pass into cosine similarity computations
# userdata = tsc.cosine_computations(labels_dict, set_lk, labels_kmeans, colNames, newdf)
# # print userdata
# print tsc.average_sim_cluster(userdata)

# In[17]:


#get the labels dictionary
labels_dict_kc = tsc.getLabelsDict(set_lk, labels_kmeans)
#pass into cosine similarity computations
userdata = tsc.cosine_computations(labels_dict_kc, set_lk, labels_kmeans, colNames, newdf)
# print userdata
print tsc.average_sim_cluster(userdata)


# In[18]:

#get the labels dictionary
labels_dict_sc = tsc.getLabelsDict(set_ls, spectral_labels)
#pass into cosine similarity computations
userdata = tsc.cosine_computations(labels_dict_sc, set_ls,spectral_labels, colNames, newdf )
# print userdata
print tsc.average_sim_cluster(userdata)

#print 


# In[19]:

#get the labels dictionary
labels_dict_hc = tsc.getLabelsDict(set_lh, h_labels)
#pass into cosine similarity computations
userdata = tsc.cosine_computations(labels_dict_hc, set_lh, h_labels, colNames, newdf)
# print userdata
print tsc.average_sim_cluster(userdata)


# In[37]:

def analysis(labels_dictionary, name):
    
    f= open(name,"w+") 
    
    for label_vals in labels_dictionary:
        
        for indices in labels_dictionary[label_vals]:
            
            likes, dislikes = tsc.findEle(indices, colNames, newdf)
            #print "for label: ", label_vals
            f.write("For label: %d \n" %(label_vals))
            #print "People liked: "
            f.write("People liked:\n")
            for like_index in likes:
                
                #if label_vals == 1:
                 
                #print colNames[like_index]
                f.write(colNames[like_index] + "\n" )
                #print " ------"
                
            #print "People disliked: "
            f.write("People disliked:\n")
            for dislike_index in dislikes:
                
                #if label_vals == 1:
                
                #print colNames[dislike_index]
                #print colNames[dislike_index]
                f.write(colNames[dislike_index] + "\n" )
                #print " ------ "
    f.close()


# In[38]:

analysis(labels_dict_hc, "hirerchial_analysis.txt")
analysis(labels_dict_sc, "spectral_analysis.txt")
analysis(labels_dict_kc, "kmeans_analysis.txt")


# In[ ]:



