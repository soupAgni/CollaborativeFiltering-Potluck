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
from sklearn.preprocessing import Imputer
from scipy import spatial
from scipy.sparse.linalg import svds
import proj_utils as tsc
from sklearn.neighbors import kneighbors_graph
from fancyimpute import  KNN,IterativeSVD , SoftImpute





def doiteration(new_data, dataframe):
    #kmeans1
    kmeans = KMeans(n_clusters=6, random_state=0).fit(new_data)
    labels_kmeans = kmeans.labels_
    set_lk = set(labels_kmeans)
    #spectral
    spectral = SpectralClustering()
    spectral.fit(new_data)
    spectral_labels =  spectral.labels_
    set_ls = set(spectral_labels)
    #Hierarchial
    connectivity = kneighbors_graph(new_data, n_neighbors=10, include_self=False)
    ward = AgglomerativeClustering(n_clusters=8, connectivity=connectivity,
                               linkage='ward').fit(new_data)
    h_labels = ward.labels_
    set_lh = set(h_labels)

    colNames =  list(dataframe.columns.values)

    labels_dict_kc = tsc.getLabelsDict(set_lk, labels_kmeans)
    #pass into cosine similarity computations
    print "\nkmeans\n"
    userdata = tsc.cosine_computations(labels_dict_kc, set_lk, labels_kmeans, colNames, dataframe)
    print tsc.average_sim_cluster(userdata)
    print "\nSpectral\n"
    labels_dict_sc = tsc.getLabelsDict(set_ls, spectral_labels)
    #pass into cosine similarity computations
    userdata = tsc.cosine_computations(labels_dict_sc, set_ls,spectral_labels, colNames, dataframe )
    # print userdata
    print tsc.average_sim_cluster(userdata)
    print "\nHeirarchial\n"
    labels_dict_hc = tsc.getLabelsDict(set_lh, h_labels)
    #pass into cosine similarity computations
    userdata = tsc.cosine_computations(labels_dict_hc, set_lh, h_labels, colNames, dataframe)
    # print userdata
    print tsc.average_sim_cluster(userdata)



def iterateImputer(data, type_str, dataframe):
    data = np.array(data, np.float)
    impute = Imputer(strategy=type_str)
    impute.fit(data)
    new_data = impute.transform(data)
    doiteration(new_data, dataframe)



def fancyImputeAttempts(data, dataframe):
    data = np.array(data, np.float)
    #use fancy impute package
    filled_knn = KNN(k=3, verbose=False).complete(data)
    filled_softimpute = SoftImpute(verbose=False).complete(data)
    filled_svd = IterativeSVD(verbose=False).complete(data)

    print "\nKNN computations\n"
    doiteration(filled_knn, dataframe)
    print "\n SOFTIMPUTE computations\n"
    doiteration(filled_softimpute, dataframe)
    print "\n SVD computations\n"
    doiteration(filled_svd, dataframe)







df = pd.read_csv('formatted_data_ing.csv')
df = df.replace('?', np.nan)
#print df
newdf = df.drop('UserID', axis = 1)




training = []


for index, row in newdf.iterrows():
    #preprocessing for every row
    #initlize rows
    rows = []
    #for each col in row
    for col in row:
        #cast to float for nan behavior
        col = float(col)
        rows.append(col)

    training.append(rows)

#cast to np array for fun times
training = np.array(training)


print "\nMEAN computations\n"
iterateImputer(training, "mean", newdf)
print "\nMEDIAN computations\n"
iterateImputer(training, "median", newdf)
print "\nMOST_FREQUENT computations\n"
iterateImputer(training, "most_frequent",newdf)

#fancy impute package tests
fancyImputeAttempts(training, newdf)
