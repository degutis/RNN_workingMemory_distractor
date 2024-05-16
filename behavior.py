import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import vonmises

def createDataframe(stim_angle, dist_angle, predicted_angle_total):

    performance_results = pd.DataFrame()
    performance_results['Target'] = circularConversion(stim_angle.reshape(stim_angle.shape[0]*stim_angle.shape[1]))
    performance_results['Distractor'] = circularConversion(dist_angle.reshape(dist_angle.shape[0]*stim_angle.shape[1]))
    performance_results['Target_predicted'] = circularConversion(predicted_angle_total.reshape(stim_angle.shape[0]*stim_angle.shape[1]))
    performance_results['Error'] = circularConversion(performance_results['Target'] - performance_results['Target_predicted'])
    performance_results['TargetMinDistractor'] = circularConversion(performance_results['Target'] - performance_results['Distractor'])
    return performance_results

def circularConversion(data):
    return (data+90)-(np.floor((data+90)/180))*180 - 90


def runBehavioral(stim,dist, predicted_angle, seed):

    performance_results = createDataframe(stim, dist, predicted_angle)

    fig, axes = plt.subplots(2, figsize=(10, 20))
    axes[0].scatter(performance_results.Target,performance_results.Target_predicted,alpha=0.05)
    axes[0].plot(np.arange(-90,91),np.arange(-90,91),"g-")        
    axes[0].set_title("Target vs. Target predicted")
    axes[0].set_xlabel("Target (degrees)")
    axes[0].set_ylabel("Target predicted (degrees)")

    axes[1].scatter(performance_results.Target,performance_results.Error,alpha=0.05)
    axes[1].plot(np.arange(-90,90),np.zeros(180),"-g")
    axes[1].set_title("Target vs. Error")
    axes[1].set_xlabel("Target (degrees)")
    axes[1].set_ylabel("Error (degrees)")
   
    plt.rcParams['svg.fonttype'] = "none"
    path2save = os.path.join("results","figures","behavior",f"TargetVsError_seed_{seed}.svg")
    plt.savefig(path2save)

    return performance_results

def movingAverage(xD,yD,bins=5):

    y_mean = np.empty(180-bins+1)
    y_sem = np.empty(180-bins+1)
    x_mean = np.empty(180-bins+1)

    for a in range(180-bins+1):
        y_mean[a] = yD[(xD>=-90+a) & (xD<-90+bins+a)].mean()
        y_sem[a] = yD[(xD>=-90+a) & (xD<-90+bins+a)].sem()
        x_mean[a] = -90+a+(bins/2)
    return y_mean, y_sem, x_mean


def combineBehavior(path2save):
    csv_files = [file for file in os.listdir(path2save) if file.startswith('behavior')]
    df_list = []  
    y_mean_across_subs_targetDist = np.empty((len(csv_files),176)) 
    y_mean_across_subs_targetError = np.empty((len(csv_files),176)) 
    
    for idx, file, in enumerate(csv_files):
        file_path = os.path.join(path2save, file)
        df = pd.read_csv(file_path)
        df['Seed'] = int(file[-1])
        df_list.append(df)
        [y_mean_across_subs_targetDist[idx,:], y_sem, x_mean] = movingAverage(df['TargetMinDistractor'], df['Error'])
        [y_mean_across_subs_targetError[idx,:], y_sem, x_mean] = movingAverage(df['Target'], df['Error'])
    
    combined_df = pd.concat(df_list, ignore_index=True)

    outputFile = os.path.join("figures","behavior",f"Across_subjects_TargetDistractorError.svg")
    plotAcrossSubs(y_mean_across_subs_targetDist,x_mean,"Target - Distractor Error","Target - Distractor","Error",outputFile)
    outputFile = os.path.join("figures","behavior",f"Across_subjects_TargetError.svg")
    plotAcrossSubs(y_mean_across_subs_targetError,x_mean,"Target Error","Target","Error",outputFile)

    return combined_df


def plotAcrossSubs(y,x,title,xlabel,ylabel,outputFile):
    
    y_std = np.std(y,axis=0)
    y_mean = np.mean(y,axis=0)
    lower=y_mean-y_std
    upper=y_mean+y_std
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(x, y_mean, label='signal mean')
    ax.plot(x, lower, color='tab:blue', alpha=0.1)
    ax.plot(x, upper, color='tab:blue', alpha=0.1)
    ax.fill_between(x, lower, upper, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   
        
    ax.plot((0, 0), (-20, 20), 'k--',alpha=0.25)
    ax.plot((-90, 90), (0, 0), 'k--',alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xlim([-90,90])
    plt.ylim([-2,2])

    plt.rcParams['svg.fonttype'] = "none"
    plt.savefig(outputFile)

def calculateVonMises(combined_df):

    bins = [-91, -25, -15, -7.5, -2.5, 2.5, 7.5, 15, 25, 91]
    combined_df['Bin'] = pd.cut(combined_df['Error'], bins=bins, labels=['-57.5','-20','-10','-5','0','5','10','20','57.5'])

    grouped = combined_df.groupby(['Seed', 'Bin']).size().reset_index(name='count')
    total_counts = grouped.groupby('Seed')['count'].sum()
    grouped = grouped.merge(total_counts, on='Seed', suffixes=('', '_total'))
    grouped['frequency'] = grouped['count'] / grouped['count_total']

    result = grouped.groupby('Bin')['frequency'].agg(['mean', 'std']).reset_index()

    vm_params = vonmises.fit(np.deg2rad(combined_df["Error"]))
#    x = np.linspace(np.deg2rad(min(combined_df["Error"])), np.deg2rad(max(combined_df["Error"])), 1000)
    x = np.linspace(np.deg2rad(-90), np.deg2rad(90),1000)
    pdf = vonmises.pdf(x, kappa=vm_params[0], scale=vm_params[2], loc=vm_params[1])

    total_area = np.trapz(pdf, x)
    normalized_pdf = pdf/total_area



    plt.plot(np.rad2deg(x),pdf)
    plt.errorbar(result["Bin"].values.astype(float), result["mean"], yerr=result["std"], fmt="o", capsize=10)
    plt.xlim(-60,60)