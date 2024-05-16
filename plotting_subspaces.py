import numpy as np
import os
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

def calculate_bins(stim, hidden, timePoints, positionDistractor, stim_bin_select=1):
    
    if stim_bin_select==1:
        stim_bins = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180])
    elif stim_bin_select==2:
        stim_bins = np.array([30,60,90,120,150,180])
    
    stim_binned = np.digitize(stim,stim_bins)

    #252 300 150
    
    trials = hidden.shape[0]
    trials_in_run = trials//timePoints
    numRuns = hidden.shape[1]
    numNeurons = hidden.shape[2]

    hid_2d = np.zeros((trials_in_run*numRuns,numNeurons,timePoints))

    for h in range(timePoints):
        hid_2d[:,:,h] = np.reshape(hidden.detach().numpy()[h:trials:timePoints,:,:],(trials_in_run*numRuns,-1))

    stim_2d = np.reshape(stim_binned[0:trials:timePoints,:],(-1))
    distractor_2d = np.reshape(stim_binned[positionDistractor:trials:timePoints,:],(-1))

    average_activity = np.zeros((stim_bins.size,numNeurons,timePoints))
    average_activity_distractor = np.zeros((stim_bins.size,numNeurons,timePoints))

    for h in range(timePoints):
        for i in np.unique(stim_2d):
            average_activity[i,:,h] = np.mean(hid_2d[stim_2d==i,:,h],axis=0)
            average_activity_distractor[i,:,h] = np.mean(hid_2d[distractor_2d ==i,:,h],axis=0)# %%
  
    return average_activity, average_activity_distractor, stim_bins, stim_2d, distractor_2d, hid_2d

def plotSubspace_singleSub(stim, hidden, timePoints, positionDistractor, seed):
    
    average_activity, average_activity_distractor, stim_bins, stim_2d, distractor_2d, hid_2d = calculate_bins(stim, hidden, timePoints, positionDistractor)

    angleBins = average_activity.shape[0]    
    numPC_comp = 3

    fig, axs = plt.subplots(2,timePoints, subplot_kw={'projection': '3d'}, figsize=(3*8,6))

    for h in range(timePoints):
        # Set up PCA
        pca = PCA(n_components=numPC_comp) # reduce dimensionality to 2
        pc_activity = pca.fit_transform(average_activity[:,:,h]) 
        pc_weights = pca.components_
        
        # Set up PCA
        pca_dist = PCA(n_components=numPC_comp) # reduce dimensionality to 2
        pc_activity_dist = pca_dist.fit_transform(average_activity_distractor[:,:,h]) 
        pc_weights_dist = pca_dist.components_
        
        projection_mem_on_mem = np.matmul(average_activity[:,:,h],pc_weights.T)
        projection_dist_on_mem = np.matmul(average_activity_distractor[:,:,h],pc_weights.T)
        
        projection_dist_on_dist = np.matmul(average_activity_distractor[:,:,h],pc_weights_dist.T)
        projection_mem_on_dist = np.matmul(average_activity[:,:,h], pc_weights_dist.T)


        cmap = plt.get_cmap('twilight')
        values_color = np.linspace(0, 1, stim_bins.size)
        colors = cmap(values_color)   
        
        if angleBins == 18:
            labels = ["10","20","30","40","50","60","70","80","90","100","110","120","130","140","150","160","170","180"]
        elif angleBins==6:
            labels = ["30","60","90","120","150","180"]
        
        for i in range(len(projection_mem_on_mem[:,0])):
            axs[0,h].scatter(projection_mem_on_mem[i,0], projection_mem_on_mem[i,1], projection_mem_on_mem[i,2], c=colors[i], label= labels[i])
            axs[0,h].scatter(projection_dist_on_mem[i,0], projection_dist_on_mem[i,1], projection_dist_on_mem[i,2], c=colors[i])
            axs[1,h].scatter(projection_dist_on_dist[i,0], projection_dist_on_dist[i,1], projection_dist_on_dist[i,2], c=colors[i], label= labels[i])
            axs[1,h].scatter(projection_mem_on_dist[i,0], projection_mem_on_dist[i,1], projection_mem_on_dist[i,2], c=colors[i])
            
        axs[0,h].set_aspect('equal','box')
        axs[1,h].set_aspect('equal','box')

        axs[0,h].add_collection3d(Poly3DCollection([list(zip(projection_mem_on_mem[:,0],projection_mem_on_mem[:,1],projection_mem_on_mem[:,2]))],facecolor = "b", alpha=0.15))
        axs[0,h].add_collection3d(Poly3DCollection([list(zip(projection_dist_on_mem[:,0],projection_dist_on_mem[:,1],projection_dist_on_mem[:,2]))],facecolor = "r", alpha=0.15))

        axs[1,h].add_collection3d(Poly3DCollection([list(zip(projection_mem_on_dist[:,0],projection_mem_on_dist[:,1],projection_mem_on_dist[:,2]))],facecolor = "b", alpha=0.15))
        axs[1,h].add_collection3d(Poly3DCollection([list(zip(projection_dist_on_dist[:,0],projection_dist_on_dist[:,1],projection_dist_on_dist[:,2]))],facecolor = "r", alpha=0.15))
        
        axs[0,h].set_title("Memory subspace. Time: "+str(h))
        axs[1,h].set_title("Distractor subspace. Time: "+str(h))


    plt.legend()
    plt.tight_layout()
    # Add legend outside the subplots
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.rcParams['svg.fonttype'] = "none"
    path2save = os.path.join("results","figures","subspaces",f"memory_distractor_subspaces_seed_{seed}.svg")
    plt.savefig(path2save)

def dynamicSubspacesPlot_2D(stim,hidden,timePoints, positionDistractor, seed):

    average_activity, average_activity_distractor, stim_bins, stim_2d, distractor_2d, hid_2d = calculate_bins(stim, hidden, timePoints, positionDistractor, stim_bin_select=2)

    timePoints = average_activity.shape[2]
    angleBins = average_activity.shape[0]
    numPC_comp = 2

    fig, axs = plt.subplots(2,timePoints, figsize=(3*8,6))

    for h in range(timePoints):
        # Set up PCA
        pca = PCA(n_components=numPC_comp) # reduce dimensionality to 2
        pc_activity = pca.fit_transform(average_activity[:,:,h]) 
        pc_weights = pca.components_
        
        # Set up PCA
        pca_dist = PCA(n_components=numPC_comp) # reduce dimensionality to 2
        pc_activity_dist = pca_dist.fit_transform(average_activity_distractor[:,:,h]) 
        pc_weights_dist = pca_dist.components_
        
        projection_mem_acrossTime = np.zeros((angleBins,numPC_comp,timePoints))
        projection_dist_acrossTime = np.zeros((angleBins,numPC_comp,timePoints))

        cmap = plt.get_cmap('twilight')
        values_color = np.linspace(0, 1, stim_bins.size)
        colors = cmap(values_color)

        if angleBins == 18:
            labels = ["10","20","30","40","50","60","70","80","90","100","110","120","130","140","150","160","170","180"]
        elif angleBins==6:
            labels = ["30","60","90","120","150","180"]

        for t in range(timePoints):
            projection_mem_acrossTime[:,:,t] = np.matmul(average_activity[:,:,t],pc_weights.T)
            projection_dist_acrossTime[:,:,t] = np.matmul(average_activity[:,:,t],pc_weights_dist.T)
        
        for line_plt in range(angleBins):
            axs[0,h].plot(projection_mem_acrossTime[line_plt,0,:],projection_mem_acrossTime[line_plt,1,:], alpha=0.5, c=colors[line_plt])
            axs[1,h].plot(projection_dist_acrossTime[line_plt,0,:],projection_dist_acrossTime[line_plt,1,:], alpha=0.5, c=colors[line_plt])
               
        for i in range(angleBins):
            for p in range(timePoints):
                if p%6==0:
                    axs[0,h].scatter(projection_mem_acrossTime[i,0,p], projection_mem_acrossTime[i,1,p], c=colors[i], label= labels[i], alpha=0.16+0.1*p)
                    axs[1,h].scatter(projection_dist_acrossTime[i,0,p], projection_dist_acrossTime[i,1,p], c=colors[i], label= labels[i], alpha=0.16+0.1*p)
                else:
                    axs[0,h].scatter(projection_mem_acrossTime[i,0,p], projection_mem_acrossTime[i,1,p], c=colors[i], alpha=0.16+0.1*p)
                    axs[1,h].scatter(projection_dist_acrossTime[i,0,p], projection_dist_acrossTime[i,1,p], c=colors[i], alpha=0.16+0.1*p)


        axs[0,h].set_aspect('equal','box')
        axs[1,h].set_aspect('equal','box')
            
        axs[0,h].set_title("Memory subspace. Time: "+str(h))
        axs[1,h].set_title("Distractor subspace. Time: "+str(h))


    plt.legend()
    plt.tight_layout()
    # Add legend outside the subplots
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.rcParams['svg.fonttype'] = "none"
    path2save = os.path.join("results","figures","subspaces",f"memory_distractor_temporal_subspaces_seed_{seed}.svg")
    plt.savefig(path2save)


def angles_plot(stim,hidden,timePoints, positionDistractor, seed):
     
    average_activity, average_activity_distractor, stim_bins, stim_2d, distractor_2d, hid_2d = calculate_bins(stim, hidden, timePoints, positionDistractor, stim_bin_select=2)

    angleBins = average_activity.shape[0]
    numPC_comp = 3

    angles_sub = np.zeros(timePoints)

    for t in range(average_activity.shape[2]):

        pca = PCA(n_components=numPC_comp) # reduce dimensionality to 2
        pc_activity = pca.fit_transform(average_activity[:,:,t]) 
        pc_weights = pca.components_
        
        # Set up PCA
        pca_dist = PCA(n_components=numPC_comp) # reduce dimensionality to 2
        pc_activity_dist = pca_dist.fit_transform(average_activity_distractor[:,:,t]) 
        pc_weights_dist = pca_dist.components_

        angles_sub[t] = np.rad2deg(subspace_angles(pc_weights.T,pc_weights_dist.T)[0])

    path2save = os.path.join("results","figures","subspaces",f"memory_distractor_angles_seed_{seed}.svg")
    
    plt.rcParams['svg.fonttype'] = "none"
    fig, ax = plt.subplots()
    ax.plot(np.arange(timePoints)[1:],angles_sub[1:])
    fig.savefig(path2save)
    
    return angles_sub