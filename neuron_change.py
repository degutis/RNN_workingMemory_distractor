import numpy as np
import plotting_subspaces
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def calculate_activity_neurons(stim, hidden, timePoints, positionDistractor):

    average_activity, average_activity_distractor, stim_bins, stim_2d, distractor_2d, hid_2d = plotting_subspaces.calculate_bins(stim, hidden, timePoints, positionDistractor)

    activity_dep_mem_dist = np.zeros((stim_bins.size,stim_bins.size,150,7))

    for h in range(7):
        for mem in np.unique(stim_2d):
            for dist in np.unique(stim_2d):
                activity_dep_mem_dist[mem,dist,:,h] = np.mean(hid_2d[(stim_2d==mem) & (distractor_2d==dist),:,h],axis=0)

    return activity_dep_mem_dist


def plotNeurons_2d(stim, hidden, timePoints, positionDistractor, seed):

    activity_dep_mem_dist = calculate_activity_neurons(stim, hidden, timePoints, positionDistractor)

    feature_coding_memory = np.transpose(np.nanmean(activity_dep_mem_dist[:,:,:,:],axis=1),(2,0,1))
    feature_coding_distractor = np.transpose(np.nanmean(activity_dep_mem_dist[:,:,:,:],axis=0),(2,0,1))

    differenceInCoding = np.abs(activity_dep_mem_dist[:,:,:,6]-activity_dep_mem_dist[:,:,:,2])

    plotAll(feature_coding_memory,15,10, "all_neurons_memory", 1, seed)
    plotAll(feature_coding_distractor, 15,10, "all_neurons_distractor", 1, seed)
    plotAll(differenceInCoding, 15,10, "all_neurons_DistractorMinusMemory", 2, seed)

    getPrototype(feature_coding_memory,5, "prot_neurons_memory", 1, seed)
    getPrototype(feature_coding_distractor,5, "prot_neurons_distractor", 1, seed)
    getPrototype(differenceInCoding,5, "prot_neurons_DistractorMinusMemory", 2, seed)


def plotAll(matrix,num_rows, num_cols, plotTitle, plotType, seed):

    if plotType==1:
        xTick = [0,8,17]
        yTick = [0,3,6]
        xLabel = ["0","90","180"]
        yLabel = ["0","3","6"]
    elif plotType==2:
        xTick = [0,8,17]
        yTick = xTick
        xLabel = ["0","90","180"]
        yLabel = xLabel

    scaler = StandardScaler()

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 30))
    count_index = 0
    # Loop through the matrices and plot them in the subplot grid
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate the index of the matrix in the list

            scaled_data = scaler.fit_transform(matrix[:,:,count_index])

            axes[i, j].imshow(scaled_data,cmap='plasma')
            axes[i, j].invert_yaxis()
            axes[i, j].set_title(f'Neuron {count_index}')

            axes[i, j].set_xticks(xTick)
            axes[i, j].set_yticks(yTick)

            axes[i, j].set_xticklabels(xLabel)
            axes[i, j].set_yticklabels(yLabel)

            count_index+=1

    # Adjust layout for better spacing
    plt.tight_layout()    
    plt.rcParams['svg.fonttype'] = "none"
    path2save = os.path.join("results","figures","neuron_change",plotTitle+f"{seed}.svg")
    plt.savefig(path2save)


def getPrototype(matrices, X, plotTitle, plotType, seed):

    if plotType==1:
        xTick = [0,8,17]
        yTick = [0,3,6]
        xLabel = ["0","90","180"]
        yLabel = ["0","3","6"]
    elif plotType==2:
        xTick = [0,8,17]
        yTick = xTick
        xLabel = ["0","90","180"]
        yLabel = xLabel

    reshape_matrix = [matrices.shape[0],matrices.shape[1]]

    flattened_matrices = np.reshape(matrices,(matrices.shape[0]*matrices.shape[1],matrices.shape[2]))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(flattened_matrices)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data.T)

    kmeans = KMeans(n_clusters=X)
    kmeans.fit(reduced_data)

    centroids_reduced = kmeans.cluster_centers_

    prototypes = []
    for centroid_reduced in centroids_reduced:
        centroid_original = pca.inverse_transform(centroid_reduced)
        prototype = centroid_original.reshape(reshape_matrix[0], reshape_matrix[1])
        prototypes.append(prototype)
    
    num_rows = X
    fig, axes = plt.subplots(num_rows, figsize=(20, 10))
    count_index = 0

    for i,prot in enumerate(prototypes):

        axes[i].imshow(prot, cmap='plasma')
        axes[i].invert_yaxis()
        axes[i].set_title(f'Prototypical neuron {count_index}')

        axes[i].set_xticks(xTick)
        axes[i].set_yticks(yTick)

        axes[i].set_xticklabels(xLabel)
        axes[i].set_yticklabels(yLabel)
        count_index+=1

    # Adjust layout for better spacing
    plt.tight_layout()    
    plt.rcParams['svg.fonttype'] = "none"
    path2save = os.path.join("results","figures","neuron_change",plotTitle+f"{seed}.svg")
    plt.savefig(path2save)
