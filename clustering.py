# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:27:35 2022

@author: robin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.utils import resample

plt.close('all')

load_data='blobs'  # Options: - blobs, circles, moons

n_clusters=3       # choose from the dendogram

eps=0.15           # parameter needed for db-scan

def main():
    raw_data=importData(load_data)
    
    scaled_data=scaleData(raw_data)
    
    plotData(scaled_data)
    
    plotDendogram(scaled_data)
    
    compute_kmeans(scaled_data,n_clusters)
    
    compute_dbscan(scaled_data, eps)
    
    compute_spectralclustering(scaled_data, n_clusters)

def importData(load_data):

    ''' This function loads the data
       input: - load_data: name of the data set (blobs, circles, moons)
    
       return:- data: raw data
           
    '''
    ##########################################################################
       
    num_samples = 1000
    
    if(load_data=='blobs'):
        print('Load blobs data')
        data,_ = datasets.make_blobs(n_samples=num_samples, centers=[[1,1], [0,0], [-1,-1]], cluster_std=[0.3, 0.3, 0.2], random_state=7)

    elif(load_data=='circles'):
        print('Load circles data')
        data,_ = datasets.make_circles(n_samples=num_samples, factor = 0.6, noise = 0.05, random_state=7)

    elif(load_data=='moons'):
        print('Load moons data')
        data,_ = datasets.make_moons(n_samples=num_samples,noise = 0.08,random_state=7)
        
    else:
        sys.exit('Data not defined!! Define statement to load data in function importData()')    

    return data

def scaleData(data):
    
    ''' This function scales the raw data
       input: - raw_data
    
       return:- scaled data
           
    '''
    ##########################################################################
       
    print('Scale data to have zero mean and unit variance')
    data=StandardScaler().fit_transform(data)
    data = np.array(data)
    
    return data

def plotData(data):
    
    # Plot the scaled data
    
    plt.figure()
    plt.scatter(data[:,0],data[:,1])
    plt.show()
    
def plotDendogram(data):

    ''' This function calculates and plot the dendogram
       input: - scaled data
           
    '''
    ##########################################################################
           
    data_subsample = resample(data, n_samples = 50, random_state=7)
    
    plt.figure(figsize=(18,7))
    plt.title('Dendrogram from agglomerative hierarchical clustering for the dataset')
    plt.xlabel('Data points')
    plt.ylabel('Cluster distances')
    plt.grid(True)
    dendrogram = sch.dendrogram(sch.linkage(data_subsample, method = 'complete'))
    plt.show()    
    
def compute_kmeans(data,n_clusters):
    
    ''' This function computed and plot 
        k-means clustering of the scaled data
       
        input: - data: scaled_data
                 n_clusters: number of cluster
               
    '''
    ##########################################################################
           
    kmeans = cluster.KMeans(n_clusters)
    kmeans.fit(data)
    
    plt.figure()
    plt.title("Data set clustered by k-means")
    scatter = plt.scatter(data[:,0],data[:,1],c=kmeans.labels_)
    plt.legend(*scatter.legend_elements())
    plt.show()
    
def compute_dbscan(data,eps):
    
    ''' This function computed and plot 
        db-scan clustering of the scaled data
       
        input: - data: scaled_data
                 eps: neighbouring distance
               
    '''
    ##########################################################################

    dbscan = cluster.DBSCAN(eps=eps,min_samples=10)
    dbscan.fit(data)

    plt.figure()
    plt.title("Data set clustered by DBScan")
    scatter = plt.scatter(data[:,0],data[:,1], c=dbscan.labels_)
    plt.legend(*scatter.legend_elements())
    plt.show()
    
def compute_spectralclustering(data,n_clusters):
    
    ''' This function computed and plot 
        spectral clustering of the scaled data
       
        input: - data: scaled_data
                 n_clusters: number of cluster
               
    '''
    ##########################################################################

    spectral = cluster.SpectralClustering(n_clusters=n_clusters,affinity="nearest_neighbors")
    spectral.fit(data)

    plt.figure()
    plt.title("Data set clustered by spectral clustering")
    scatter = plt.scatter(data[:,0],data[:,1],c=spectral.labels_)
    plt.legend(*scatter.legend_elements())
    plt.show()    

if __name__ == '__main__':
    main()
    