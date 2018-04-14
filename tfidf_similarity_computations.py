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

# In[10]:

#testing
#
def getLabelsDict(set_labels, label_vals):
    labels_dict = {}
    ind = 0

    for i in range(0, len(set_labels)):
        labels_dict[i] = []
    #print len(set_lk)
    #print labels_dict
    for i in label_vals:
        #if i not in labels_dict.values:
        #print i
        labels_dict[i].append(ind)
        ind = ind + 1

    return labels_dict


# In[11]:

#finding the index of the recipe corresponding to a like and a particular dislike of a particular user. 
#returns a tuple of lists where tuple(0) contains the index of the likes and tuple(1) contains the index of the dislikes. 
def findEle(index, colNames, newdf):
    #get all the colums corresponding to the particular row
    #colVals = df.loc[[index]]
    toRet = []
    likes = []
    dislikes = []
    counter = int(0)
    for col in colNames:
        if not np.isnan(float(newdf.loc[[index]][col])):
            #print np.array(newdf.loc[[index]][col].astype(list))[1]
            for i in np.array(newdf.loc[[index]][col].astype(list)):
                #print type(i)
                if i == "1":
                    likes.append(counter)
                else:
                    dislikes.append(counter)
        counter += 1
            #toRet.append(col)
    #print "likes: ", likes
    #print "dislikes: ", dislikes
    return (likes, dislikes)


# In[12]:

#compute tf-idf for the given array
#in Scikit-Learn
def tfIDF(colNames):
    
    names_to_vec = {}
    sklearn_tfidf = TfidfVectorizer(stop_words = 'english')
    vec_representation = sklearn_tfidf.fit(colNames)
    feature_names = sklearn_tfidf.get_feature_names()
    #print vec_representation
    idx = 0
    for names in feature_names:
        
        names_to_vec[names] = (sklearn_tfidf.idf_[idx], idx)
        idx = idx +1
    
    zerosLists = np.zeros([len(colNames), len(feature_names)])
    
    i = 0
    for rec in colNames:
       
        for word in rec.strip().split(' '):
            if word in names_to_vec:
                zerosLists[i][names_to_vec[word][1]] = names_to_vec[word][0]
                
        i = i + 1
     
    #print zerosLists
    return zerosLists

#print len(tfIDF(colNames))


# In[13]:

#cosine similarity computations. 


# In[14]:

def cosine_computations(labels_dict, set_lk, label, colNames, newdf):
    userdata = {}
    tfidf_vecs = tfIDF(colNames)
    for i in range(0, len(set_lk)):
        userdata[i] = {}
        for j in range(0, len(label)):
            userdata[i][j] = {}
            for k in range(0, len(label)):
                userdata[i][j][k] = ([],[])

    user_ignored = int(0)

    #1-------
    #         for like in likes:
    #             eachuserlike = []
    #             if not len(likes) == 0:
    #                 eachuserlike = like
    #             userdata.append(eachuserlike)


    #for each cluster
        #for each user in the cluster
            #get the liked recipes
            #get the disliked recipes
            #for each user not /= original 
                #get the liked recipes
                #get the disliked recipes
                #compare the recipes


    #for each cluster

    for key in labels_dict:
        print "key: " , key   
        #print "Key: ", key
        #for each user in the cluster
        for user1 in labels_dict[key]:
            likes1, dislikes1  = findEle(user1, colNames, newdf)
            #print "User1: ", user1
            #-----------1
            #go through every other user
            if not len(likes1) == 0 and not len(dislikes1) == 0:
                for user2 in labels_dict[key]:
                    #print "User2", user2
                    #get the likes and dislikes
                    likes2, dislikes2 = findEle(user2, colNames, newdf)

                    if not len(likes2) == 0 and not len(dislikes2) == 0:
                        
                        #comparison of recipes
                        #for each recipe in first user's 
                        for like1 in likes1:
#                             #for each recipe in second user's
                             for like2 in likes2:
                                #get the liked similarity and place in array
                                #cluster -> user1 -> user2 -> tuple of likes/dislikes -> liked comparison
                                #print "like1: ", np.array(like1), " like2: ", np.array(like2)
                                #userdata[key][user1][user2][0].append(cosine_similarity(like1, like2)) #likes
                                userdata[key][user1][user2][0].append(spatial.distance.cosine(tfidf_vecs[like1], tfidf_vecs[like2]))

                        for dislike1 in dislikes1:
#                           #for each recipe in second user's
                             for dislike2 in dislikes2:
#                                 #get the liked similarity and place in array
#                                 #cluster -> user1 -> user2 -> tuple of likes/dislikes -> disliked comparison
#                                 userdata[key][user1][user2][1].append(cosine_similarity(dislike1, dislike2)) #likes
                                userdata[key][user1][user2][1].append(spatial.distance.cosine(tfidf_vecs[dislike1], tfidf_vecs[dislike2])) #likes

                    else:
                        user_ignored += 1
                    
                    #print "likes: ", userdata[key][user1][user2][0]
                    #print "dislikes: ", userdata[key][user1][user2][0]
            else:
                user_ignored += 1
            
            #print user_ignored
   
        
    return userdata
        
                    #tfIDF(likes)

                    #print likes
            #print likes


            #print likes
            #print dislikes


# In[15]:

def average_list_nan(data):
    average = float(0)
    counter = float(1)
    for el in data:
        if not np.isnan(float(el)):
            average += float(el)
            counter += 1
    return average/counter


# In[16]:

def average_sim_cluster(userdata):
    averages = []#[() for _ in range(len(userdata))]
    for cluster in userdata:
        averagelikes = float(0)
        averagedislikes = float(0)
        counter = float(1)
        
        for user1 in userdata[cluster]:
            for user2 in userdata[cluster]:
                averagelikes += average_list_nan(userdata[cluster][user1][user2][0])
                averagedislikes += average_list_nan(userdata[cluster][user1][user2][1])
                counter += 1
        averages.append((float(1) - averagelikes/counter, float(1) - averagedislikes/counter))
    return averages


# In[ ]:



