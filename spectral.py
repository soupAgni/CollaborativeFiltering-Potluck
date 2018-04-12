from __future__ import division
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import math



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

#-------------------------------------#
#basic processing
#-------------------------------------#
#read in the csv file
df = pd.read_csv('formatted_data.csv')

#not needed right now
# means = df.mean(axis = 1, numeric_only  = True)

#replace with nans
df = df.replace('?', np.nan)


#drop the userID beacuse it's not needed
newdf = df.drop('UserID', axis = 1)


#-----------------------------------#
#fill in missing data with averages
#-----------------------------------#

#initlize list of lists for spectral
training = []

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

    training.append(rows)

#cast to np array for fun times
training = np.array(training)

#------------------------------#
#clustering phase begin
#------------------------------#
#begin spectral clustering phase
#use default state
spectral = SpectralClustering()
spectral.fit(training)

print "lables from clustering"
print spectral.labels_
