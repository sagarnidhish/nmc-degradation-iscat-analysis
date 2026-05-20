# -*- coding: utf-8 -*-
"""
Combining the object detection outputs of a batch processed .txt input
Degradation Specific processing.

Created on Thu Nov 23 11:04:51 2023

@author: Alek
"""


'''
Things to analyse:
    1. Number of particles per cycle
    2. Mean particle intensities per cycle (start, end, top) (against capacity and potential)
    3. ...


1. Particle data - Combine object detection outputs into a single dataframe - add new column as cycleNo identifier
2. Ensemble - Import electrochemical time points - start, top and end of charge
3. Ensemble - Import charge/discharge capacities and coulombic efficiency for each cycle.
4. 


'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# nFiles = 28 # Number of individual cycles that were processed with object detection
nFiles = 88 # Number of individual cycles that were processed with object detection

fNames = ['']*nFiles

# with open(r'D://NMC_degradation_3_160623_Halfthedata/objDetAddrs_10_2_2.txt', 'r') as f:
with open(r'D://NMC_degradation_3_160623_Halfthedata/objDetAddrs_2_1_1.txt', 'r') as f:

    print(f.readline())
    
    for i in range(0,nFiles):
        # a.append(f.readline())
        fNames[i] = f.readline()
        fNames[i] = fNames[i][:-1] #Removes '\n' from the end of each string
    print('Processed filenames imported')
    

# Load dataframe containing all relevant echem data

# echemDF = pd.read_csv('D://NMC_degradation_3_160623_Halfthedata/echemDF_partial_10_2_2.csv')
echemDF = pd.read_csv('D://NMC_degradation_3_160623_Halfthedata/echemDF_partial_2_1_1.csv')

echemDF = echemDF.assign(noParticles = np.nan) #empty column to record number of particles per cycle.

'''
to do 30/11:
2. Import particle traces with added cycleNo column 
'''
particleInfo = pd.DataFrame()
particleTraces = pd.DataFrame()

# Creating paths that point to each processed cycle's output directory
# Importing electrochemistry data (echemDF)
# Importing object detection outputs (particleInfo)
# Importing object detection traces (particleTraces)

particleTracesKeys = ['NumParticlesRaw', 'IntensitiesRaw', 'CutoutIntensitiesRaw', 'SizesPix', 'cycleNo', 'frameRate']

for i in range(0,len(fNames)):
    splitName = fNames[i].split('\\')
    
    fNames[i] = splitName[0] + '\\'+ splitName[1] + '\\Output - ' + splitName[2] 
    
    #Import no. particles detected per cycle into echem dataframe
    temp_particleInfo = pd.read_csv(fNames[i] + '\\particleInfo.csv')
    
    echemDF.loc[echemDF['cycleNo'] == echemDF['cycleNo'].unique()[i], 'noParticles'] = temp_particleInfo.shape[0]
    # print(temp_particleInfo.shape[0])
    #Combine particleInfo dataframes into a single dataframe with cycleNo identifier column
    temp_particleInfo = temp_particleInfo.assign(cycleNo = echemDF['cycleNo'].unique()[i])
    particleInfo = particleInfo.append(temp_particleInfo,ignore_index=True)
    # pd.concat(particleInfo, temp_particleInfo, ignore_index=True) #ideally should use pd.concat instead of append

    #Combine particleTraces dataframes into a single dataframe with cycleNo identifier column
    temp_particleTraces = pd.read_csv(fNames[i] + '\\particleTraces.csv')
    # print(temp)
    temp_particleTraces = temp_particleTraces.assign(cycleNo = echemDF['cycleNo'].unique()[i])
    particleTraces = particleTraces.append(temp_particleTraces[particleTracesKeys], ignore_index=True)


# Read in dataframe containing the number of frames for each cycle
# No. frames and no. particles together allows for unpacking of the particleTraces frame
cycleFrames = pd.read_csv('D://NMC_degradation_3_160623_Halfthedata/cycleFrames.csv', header=None)

# Load manually selected particles to use as examples
exampleParticles = pd.read_csv('D://NMC_degradation_3_160623_Halfthedata/exampleParticles.csv')

#%% Example particle extraction:

    
#Find cycle with most particles detected - base search on these particle indices
# maxCycle = particleInfo[particleInfo['Unnamed: 0'] == np.amax(particleInfo['Unnamed: 0'].unique())]['cycleNo'].iloc[0]
#Extract particleInfo for that cycle only to use as reference
# maxParticleInfo = particleInfo[particleInfo['cycleNo']==maxCycle]


#To view first (processed) cycle particle information to pick IDs:
# particleInfo[particleInfo['cycleNo']==2]

#Selected particles of interest:
particleIndices = exampleParticles.iloc[0][1:5]

#Initialise results dataframe:
# exParts = pd.DataFrame(particleIndices)

'''
To do: add a section to iterate over ALL particles in first cycle and sort them by number of successful matches across the dataset - INPUT ALL INITIAL PARTICLES AND COUNT OCCURRANCES IN THE OUTPUT DATAFRAME
'''
 
def exampleParticlesIndices(particleIndices, etaXY=20, etaSize=50):
    

    #Reference to first cycle - extract the first cycle particleInfo
    refParticleInfo = particleInfo[(particleInfo['cycleNo']==2) & (particleInfo['Unnamed: 0'].isin(particleIndices))].reset_index().drop(columns='Unnamed: 0')


    #Iterate over cycles
    for cycleNo in particleInfo['cycleNo'].unique():
        print(cycleNo)
        #temporary dataframe, df, holds the current cycle particleInfo
        df = particleInfo[particleInfo['cycleNo']==cycleNo] #could condense this into one line with the below
        
        for i in range(0,particleIndices.shape[0]):
            matchedParts = df[(df['x'].apply(np.isclose, b=refParticleInfo['x'][i], atol=etaXY)) & (df['y'].apply(np.isclose, b=refParticleInfo['y'][i], atol=etaXY)) & (df['npix'].apply(np.isclose, b=refParticleInfo['npix'][i], atol=etaSize))]
            # matchedParts = df[(df['x'].apply(np.isclose, b=refParticleInfo['x'][i], atol=eta)) & (df['y'].apply(np.isclose, b=refParticleInfo['y'][i], atol=eta))]

            matchedParts.insert(0,'index',refParticleInfo['index'].iloc[i])
    
            refParticleInfo = pd.concat([refParticleInfo,matchedParts],ignore_index=True).drop(columns='Unnamed: 0')


    # #Ensure uniqueness of particles:
    # for cycleNo in particleInfo['cycleNo'].unique():
        
    #     for i in range(0,particleIndices.shape[0]):
            
    #         if refParticleInfo[(refParticleInfo['cycleNo']==cycleNo) & (refParticleInfo['index']==i)].shape[0] > 1:
    #             '''
    #             compare closest to original and eliminate others
    #             '''
    #         else:
    #             pass
    
    

    return refParticleInfo


particleIndices = particleIndices.to_numpy()

#Type in the particle index of interest (found in particleInfo['cycleNo']==2)
particleIndices = np.array([164])
# particleIndices = particleInfo[particleInfo['cycleNo']==2]['Unnamed: 0'].unique()[0:100]
exampleParticle = exampleParticlesIndices(particleIndices, etaXY=20)

#%%

#Alternative particle matching:
particleIndices = particleInfo[particleInfo['cycleNo']==2]['Unnamed: 0'].unique() #all particles
refParticleInfo = particleInfo[(particleInfo['cycleNo']==2) & (particleInfo['Unnamed: 0'].isin(particleIndices))].reset_index().drop(columns='Unnamed: 0') #reference cycle (cycleNo==2)


#columns to include in particle matching criteria
# cols = ['npix','x','y','a','b','theta']
cols = ['x','y','npix']

#initialise output dataframe
matchedParticles = pd.DataFrame(columns=particleInfo.columns).drop(columns='Unnamed: 0')

#iterate cycle numbers
for cycleNo in particleInfo['cycleNo'].unique():
# for cycleNo in [2,4]:
# cycleNo=2
    print(cycleNo)
    
    #Dataframe of particleInfo for the current cycle number:
    currCycle = particleInfo[particleInfo['cycleNo']==cycleNo]
    
    # #iterate through particles indices in the INITAL cycle
    for i in range(0,refParticleInfo.shape[0]):
        # i=355
        refValues = refParticleInfo[refParticleInfo['index']==i][refParticleInfo.columns[refParticleInfo.columns.isin(cols)]].iloc[0] #single particle dataframe with selected columns removed
        
        # df = np.absolute(currCycle.sub(refValues,axis=1).div(refValues).mean(axis=1))
        df = np.absolute(currCycle.sub(refValues,axis=1)).mean(axis=1)*100
        #df = np.absolute(currCycle.sub(refValues,axis=1).div(refValues,axis=1)).mean(axis=1)*100
    
        # print(df.min())
        
        # if df.min() > 5:
        #     pass
        # else:
            
        idx = df[df==df.min()].index[0]
        # print(idx)
        toAdd = currCycle.loc[idx]
    
        # toAdd.insert(0,'index',i) #here we use 'i' to record the index of the particle in the REFERENCE cycle, not the index it matched to in the current (idx)
    
        matchedParticles = pd.concat([matchedParticles,toAdd.to_frame().T],ignore_index=False)
    
    
'''
Takes ~1.5mins to perform matching on every particle for every cycle (406x88)
'''
#%% Which particles (indices from first cycle) have most matches:

occs = pd.DataFrame(columns=['index','occs'],index=range(matchedParticles['Unnamed: 0'].unique().shape[0]))    

# occs = np.zeros([matchedParticles['Unnamed: 0'].unique().shape[0],2])

counter=0
for i in matchedParticles['Unnamed: 0'].unique():
    # type(i)
    # print(type(i))
    # print(matchedParticles[matchedParticles['Unnamed: 0']==i]['npix'].count())
    occs['index'][counter] = i
    occs['occs'][counter] = matchedParticles[matchedParticles['Unnamed: 0']==i]['npix'].count()
    counter +=1
    
occs
#%%
def intensityMap():
    
    #extract the intensities for all cycles for a particle. Plot as a time vs. cycle no. map
    

        
    return arr




#%% Save the imported and combined dataframes to disk for quicker loading

'''
echemDF
particleTraces
particleInfo
cycleFrames
exampleParticles
'''

#%% Number of particles found per cycle

def numParticles(sigma=1,illustrator=False, path=None):
    df=[]
    import matplotlib as mpl
    plt.rcParams['pdf.fonttype']=42
    mpl.rcParams['figure.dpi'] = 108
    mpl.rcParams['savefig.dpi'] = 108
    
    for i in echemDF['cycleNo'].unique():
        df.append(echemDF[echemDF['cycleNo'] == i]['noParticles'].unique()[0])
        
    
    #Calculate mean and standard deviation for outlier exlusion
    stdev = np.nanstd(df)
    meanPart = np.nanmean(df)
    '''
    Note that the outliers are being calculated based on ALL cycles 
        - coincidentally this works as the later cut-off particles are all within ~1.5sigma
    '''

    #Plotting
    fig,ax = plt.subplots(1,1,figsize=[6,4])
    
    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=1.5)
    plt.tick_params(width=1.5)
    
    if illustrator == True:
        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])
        
        plt.gca().axes.set_xticks([0,40,80,120,160])
        plt.gca().axes.set_yticks([350,400,450,500])

    else:
        plt.title(r'No. particles found per cycle, $>2\sigma$ as red circ.')
        plt.xlabel('cycle number')
        plt.ylabel('no. particles')
    
    
    #Shading upper cut-off voltage regions:
    plt.fill_between([0,120], 320, 520, alpha=0.1, color='tab:green') #4.2V
    plt.fill_between([120, 136], 320, 520, alpha=0.1, color='tab:orange') #4.3V
    plt.fill_between([136, 165], 320, 520, alpha=0.1, color='tab:red') #4.3V
    
    # mean and standard deviations - calculated across ALL cycles at once
    #boundaries for axhlines
    b=[0,120/165,136/165,1]

    #4.2V cut-off
    plt.axhline(np.nanmean(df[0:53]),xmin=b[0],xmax=b[1], ls='-',color='k')
    plt.axhline(np.nanmean(df[0:53])+sigma*np.nanstd(df[0:53]),xmin=b[0],xmax=b[1],ls='--',color='k')
    plt.axhline(np.nanmean(df[0:53])-sigma*np.nanstd(df[0:53]),xmin=b[0],xmax=b[1], ls='--',color='k')
    
    #4.4V cut-off
    plt.axhline(np.nanmean(df[53:66]),xmin=b[1],xmax=b[2], ls='-',color='k')
    plt.axhline(np.nanmean(df[53:66])+sigma*np.nanstd(df[53:66]),xmin=b[1],xmax=b[2],ls='--',color='k')
    plt.axhline(np.nanmean(df[53:66])-sigma*np.nanstd(df[53:66]),xmin=b[1],xmax=b[2], ls='--',color='k')
    
    # #4.6V cut-off
    plt.axhline(np.nanmean(df[66:]),xmin=b[2],xmax=b[3], ls='-',color='k')
    plt.axhline(np.nanmean(df[66:])+sigma*np.nanstd(df[66:]),xmin=b[2],xmax=b[3],ls='--',color='k')
    plt.axhline(np.nanmean(df[66:])-sigma*np.nanstd(df[66:]),xmin=b[2],xmax=b[3], ls='--',color='k')
    

    
    nParticles = pd.DataFrame(columns=['cycleNo','nParticles','nParticlesOutlier'],index=range(echemDF['cycleNo'].unique().shape[0]))    
    counter = 0
    
    for i in echemDF['cycleNo'].unique():
        
        n = echemDF[echemDF['cycleNo'] == i]['noParticles'].unique()[0]    
        
        if n < meanPart + 2*stdev and n > meanPart - 2*stdev: 
            # nParticles.append([i,n])
            nParticles['nParticles'][counter] = n
        else:
            nParticles['nParticlesOutlier'][counter] = n
            
        nParticles['cycleNo'][counter] = i
    
        counter+=1
        
    
    plt.plot(nParticles['cycleNo'][:],nParticles['nParticles'][:],'xk')
    plt.plot(nParticles['cycleNo'][:],nParticles['nParticlesOutlier'][:],'or')
  
    plt.xlim(0,165)
    plt.ylim(320,520)
    
    
    if illustrator == True:
        plt.savefig(path+'Figure S1a.pdf',bbox_inches='tight')
    else:
        plt.savefig(r'D://NMC_degradation_3_160623_Halfthedata/Figures/noParticles.pdf')
    
    plt.show()
    
    return df

pathS1 = r'D://NMC_degradation_3_160623_Halfthedata/Figures/Figure S1/'
df = numParticles(illustrator=False, path=pathS1, sigma=1)


#%% Average (mean, median, mode, Std. Dev.) particle size per cycle

def particlePopulationStats(particleDF, exampleParticles=False, illustrator = False, path=None, legend=True):
    

    fig,ax = plt.subplots(1,1,figsize=[6,4])

    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=1.5)
    plt.tick_params(width=1.5)

    ylims=[0,4]
    xlims=[0,170]
    #Shading upper cut-off voltage regions:
    plt.fill_between([0,120], ylims[0], ylims[1], alpha=0.1, color='tab:green') #4.2V
    plt.fill_between([120, 136], ylims[0], ylims[1], alpha=0.1, color='tab:orange') #4.4V
    plt.fill_between([136, 170], ylims[0], ylims[1], alpha=0.1, color='tab:red') #4.6V
    plt.xlim(xlims)
    plt.ylim(ylims)
    '''
    make the ylim dynamic
    '''
    means=[]
    medians=[]
    stds=[]
    modes=[]
    
    sizeCorr = (95*1e-3)**2
    
    for i in particleInfo['cycleNo'].unique():
        series = particleInfo[particleInfo['cycleNo'] == i]['npix']
        means.append(series.mean())
        medians.append(series.median())
        stds.append(series.std())
        modes.append(series.mode()[0])
                
        if exampleParticles == True:
            IDs = particleDF[particleDF['cycleNo']==i]['index'].unique()

            for j in range(0,particleDF[particleDF['cycleNo']==i].shape[0]):
                try:
                    # plt.plot(i,particleDF[particleDF['cycleNo']==i]['npix'].iloc[j],'k',markersize=20,marker='${}$'.format(IDs[j]))
                    plt.plot(i,particleDF[particleDF['cycleNo']==i]['npix'].iloc[j],'db')
                except:
                    pass
        else:
            pass

    
    means=np.array(means)*sizeCorr
    medians=np.array(medians)*sizeCorr
    stds=np.array(stds)*sizeCorr
    modes=np.array(stds)*sizeCorr
    
    plt.axhline(25*sizeCorr,linestyle='--',color='k', label='Min. Facet Size')
    
    plt.plot(particleInfo['cycleNo'].unique(),means,'xk', label='Mean')
    plt.plot(particleInfo['cycleNo'].unique(),medians,'xr', label='Median')
    plt.plot(particleInfo['cycleNo'].unique(),stds,'xg', label='St. Dev.')
    # plt.plot(particleInfo['cycleNo'].unique(),modes,'xy', label='mode')
    

    if illustrator == True:

        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])
        
        plt.gca().axes.set_xticks([0,40,80,120,160])
        # ylabels = np.array([0,50,100,150,200,250,300])*95**2
        plt.gca().axes.set_yticks([0,1.0,2.0,3.0])
    else:
        plt.xlabel('cycle number')
        plt.ylabel('particle size / pixels')
        
        
    if legend == True:
        plt.legend()
        fName = 'Figure S1b legend.pdf'
    else:
        fName = 'Figure S1b.pdf'
        pass
    
    plt.show()
    
    if illustrator == True:
        plt.savefig(path+fName,bbox_inches='tight')
    else:
        plt.savefig(r'D://NMC_degradation_3_160623_Halfthedata/Figures/particlePopulationStats.pdf')
    

particlePopulationStats(particleInfo,exampleParticles=False, illustrator=False, path=pathS1, legend=False)
#%% Extract maximum intensity of each particle per cycle, and time at which it occurred

import sys
sys.path.append("C://Users/alek/Desktop/analysis")
import DataManipulation as dm

nFrames = cycleFrames[1][:].to_numpy()

topChargeInts = np.zeros([np.nanmax(echemDF['noParticles'].unique()).astype(int),cycleFrames[1][:].shape[0]])
topChargeTimes = np.zeros(topChargeInts.shape)
topChargeInds = np.zeros(topChargeInts.shape)
topChargeIntsRaw = np.zeros(topChargeInts.shape)

startChargeIntsRaw = np.zeros(topChargeInts.shape)
endDischargeIntsRaw = np.zeros(topChargeInts.shape)
startChargeInts = np.zeros(topChargeInts.shape)
endDischargeInts = np.zeros(topChargeInts.shape)

cycleInd = 0
# i --> cycle number
for i in particleInfo['cycleNo'].unique().astype(int):
    print('cycle no.: ',i)
    arr = dm.unflatten(particleTraces[particleTraces['cycleNo']==i],'IntensitiesRaw')
    fRate = particleTraces[particleTraces['cycleNo']==i]['frameRate'].unique()[0]

    # j --> particle index
    for j in range(0,particleTraces[particleTraces['cycleNo']==i]['NumParticlesRaw'].unique()[0].astype(int)):
        #Normalise to first time point (traces are correctly starting and ending at echem defined points)
        
        topChargeIntsRaw[j,cycleInd] = np.nanmax(arr[j,:]) #raw intensities
        startChargeIntsRaw[j,cycleInd] = arr[j,0]
        endDischargeIntsRaw[j,cycleInd] = arr[j,-1]     
        
        arr[j,:] = arr[j,:]/arr[j,0]    
    
        topChargeInts[j,cycleInd] = np.nanmax(arr[j,:]) #normalised intensities

        startChargeInts[j,cycleInd] = arr[j,0]
        endDischargeInts[j,cycleInd] = arr[j,-1]        

        try:
            topChargeInds[j,cycleInd] = np.where(arr[j,:]==topChargeInts[j,cycleInd])[0]
            
        except:

            topChargeInds[j,cycleInd] = np.nan
            
        topChargeTimes[j,cycleInd] = topChargeInds[j,cycleInd]/fRate
        

    #Increment cycle INDEX
    cycleInd +=1


# Data cleaning - residual zeros due to initialisation as np.zeros() array
topChargeTimes[topChargeTimes==0] = np.nan
topChargeInds[topChargeInds==0] = np.nan
topChargeInts[topChargeInts==0] = np.nan
topChargeIntsRaw[topChargeIntsRaw==0] = np.nan

#manual checks of the below conditions have revealed that these particles' max. intensities occur at unreasonable times
mask = np.where(topChargeTimes>6000)
topChargeTimes[mask] = np.nan
mask = np.where(topChargeTimes<3000)
topChargeTimes[mask] = np.nan

echemChargeTimes = np.zeros(topChargeInts.shape[1])
cycleInd = 0

#Find echem-defined top of charge for each cycle:
for i in particleInfo['cycleNo'].unique().astype(int):
    # print('cycle no.: ', i)
    fRate = particleTraces[particleTraces['cycleNo']==i]['frameRate'].unique()[0]

    df = echemDF[echemDF['cycleNo']==i]
    t1 = df[df['Block'] == df['Block'].unique()[1]]['Time (s)'].iloc[0]
    t0 = df[df['Block'] == df['Block'].unique()[0]]['Time (s)'].iloc[0]
    echemChargeTimes[cycleInd] = t1-t0
    cycleInd +=1


#%% Plotting the mean, median and standard deviations of top of charge heterogeneity (in time)

def timeHeterogeneity(illustrator=False,path=None, topChargeTimes=topChargeTimes,echemChargeTimes=echemChargeTimes):
    from matplotlib.lines import Line2D
    
    size=[6,4]
    size[0] = size[0]*1.6*2
    size[1] = size[1]*1.6
    markersize = 10
    mew=2
    linewidth = 1.5*1.6
    
    fig,ax = plt.subplots(1,1,figsize=[size[0]-0.5,size[1]])
    
    ylims = np.array([-3000,3000])
    ylims = ylims/60 #for y-axis in minutes

    
    means = np.zeros(particleInfo['cycleNo'].unique().shape[0])
    medians = np.zeros(means.shape)
    stds = np.zeros(means.shape)
    uniqueCycles = particleInfo['cycleNo'].unique().astype(int)
    

    for cycleInd in range(0, uniqueCycles.shape[0]):
    
        means[cycleInd] = np.nanmean(topChargeTimes[:,cycleInd]-echemChargeTimes[cycleInd])
        medians[cycleInd] = np.nanmedian(topChargeTimes[:,cycleInd]-echemChargeTimes[cycleInd])
        stds[cycleInd] = np.nanstd(topChargeTimes[:,cycleInd]-echemChargeTimes[cycleInd])
        
    means=means/60
    medians=medians/60
    stds=stds/60
        
    plt.plot(uniqueCycles, means,'xr', ms=markersize, lw=linewidth,mew=mew)
    plt.plot(uniqueCycles, medians, 'xb',ms=markersize, lw=linewidth,mew=mew)
    plt.plot(uniqueCycles, stds, 'xk',ms=markersize, lw=linewidth,mew=mew)
    # plt.plot(uniqueCycles, (means+1*stds)/60, '--k')
    # plt.plot(uniqueCycles, (means-1*stds)/60, '--k')
    # plt.fill_between(uniqueCycles,means-stds,means+stds)

    plt.axhline(0, color='k', linestyle='--',linewidth=linewidth)
    plt.axvline(120,color='k',linestyle='--',linewidth=linewidth)
    plt.axvline(136,color='k',linestyle='--',linewidth=linewidth)

    
    #Shading upper cut-off voltage regions:
    plt.fill_between([-5,120], ylims[0], ylims[1], alpha=0.1, color='tab:green') #4.2V
    # plt.txt((120--5)/2, 190, 'test')
    plt.fill_between([120, 136], ylims[0], ylims[1], alpha=0.1, color='tab:orange') #4.3V
    plt.fill_between([136, 170], ylims[0], ylims[1], alpha=0.1, color='tab:red') #4.3V
    plt.xlim(0,160)
    plt.ylim(ylims[0], ylims[1])
    
    #Custom Legend
    legend_elements = [Line2D([0], [0], color='r', marker = 'x', lw=0, label='Mean'),
                       Line2D([0], [0], color='b', marker = 'x', lw=0, label='Median'),
                       Line2D([0], [0], color='k', marker = 'x', lw=0, label='St. Dev.')]
    
    plt.legend(handles=legend_elements)
    
    if illustrator == True:
        pass
    else:
        plt.title(r'$\Delta T$ between echem-defined top of charge and time of max. intensity')
        plt.xlabel('Cycle No.')
        plt.ylabel(r"$\Delta T$ /mins")
    
    plt.show()
    plt.savefig(path+'Figure 4b.pdf',bboc_inches='tight')
    
    return means, stds

meansTime, stdsTime=timeHeterogeneity(path=path4,illustrator=True)
#%% Plotting the mean, median and standard deviations of top of charge heterogeneity (in intensity)
import matplotlib as mpl

plt.rcParams['pdf.fonttype']=42
mpl.rcParams['figure.dpi'] = 108
mpl.rcParams['savefig.dpi'] = 108

def intensityHeterogeneity(path=None, illustrator=False,topChargeInts=topChargeInts, startChargeInts=startChargeInts, endDischargeInts=endDischargeInts):
    from matplotlib.lines import Line2D
    
    size=[6,4]
    size[0] = size[0]*1.6*2
    size[1] = size[1]*1.6
    markersize = 10
    mew=2
    linewidth = 1.5*1.6
    
    
    fig,ax = plt.subplots(1,1,figsize=[size[0]-0.5,size[1]])
    
    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=1.5)
    plt.tick_params(width=1.5)
    ax.tick_params(width=1.5*1.6,length=5)
    

    
    '''
    ylims must be toggled manually
    '''
    # ylims = np.array([1500,15000])
    ylims = np.array([-0.2,2.2])
    
    meansTop = np.zeros(particleInfo['cycleNo'].unique().shape[0])
    meansStart = np.zeros(meansTop.shape)
    meansEnd = np.zeros(meansTop.shape)
    # medians = np.zeros(means.shape)
    # stds = np.zeros(means.shape)
    uniqueCycles = particleInfo['cycleNo'].unique().astype(int)
    
    startEndDiff = np.zeros(meansTop.shape)
    
    cycleInd=0
    for i in uniqueCycles:
        
        # means[cycleInd] = np.nanmean(topChargeIntsRaw[:,cycleInd])
        # medians[cycleInd] = np.nanmedian(topChargeIntsRaw[:,cycleInd])
        # stds[cycleInd] = np.nanstd(topChargeIntsRaw[:,cycleInd])
        meansTop[cycleInd] = np.nanmean(topChargeInts[:,cycleInd])
        # plt.plot(uniqueCycles, medians, 'xb',ms=markersize, lw=linewidth)
        # plt.plot(uniqueCycles, stds, 'xk',ms=markersize, lw=linewidth)
        
        
        meansStart[cycleInd] = np.nanmean(startChargeInts[:,cycleInd])
        
        meansEnd[cycleInd] = np.nanmean(endDischargeInts[:,cycleInd])
        # print(means.shape)
        startEndDiff[cycleInd] = np.abs(meansEnd[cycleInd] - meansStart[cycleInd])
        
        
        cycleInd+=1
    plt.plot(uniqueCycles, meansTop,'xr', ms=markersize, lw=linewidth,mew=mew)
    plt.plot(uniqueCycles, meansStart,'xg', ms=markersize, lw=linewidth,mew=mew)
    plt.plot(uniqueCycles, meansEnd,'xb', ms=markersize, lw=linewidth,mew=mew)
    plt.plot(uniqueCycles, startEndDiff, 'xk',ms=markersize, lw=linewidth,mew=mew)

    plt.axhline(np.mean(meansTop[0:6]), color='r', linestyle='--',alpha=1,linewidth=linewidth)
    plt.axhline(np.mean(meansStart[0:6]), color='g', linestyle='--',alpha=1,linewidth=linewidth)
    plt.axhline(np.mean(meansEnd[0:6]), color='b', linestyle='--',alpha=1,linewidth=linewidth)
    plt.axhline(np.mean(startEndDiff[0:6]), color='k', linestyle='--',linewidth=linewidth)

    plt.axvline(120,color='k',linestyle='--',linewidth=linewidth)
    plt.axvline(136,color='k',linestyle='--',linewidth=linewidth)

    
    #Shading upper cut-off voltage regions:
    plt.fill_between([-5,120], ylims[0], ylims[1], alpha=0.1, color='tab:green') #4.2V
    plt.fill_between([120, 136], ylims[0], ylims[1], alpha=0.1, color='tab:orange') #4.3V
    plt.fill_between([136, 170], ylims[0], ylims[1], alpha=0.1, color='tab:red') #4.3V
    plt.xlim(0,160)
    plt.ylim(ylims[0], ylims[1])
    
    #Custom Legend
    legend_elements = [Line2D([0], [0], color='r', marker = 'x', lw=0, label='Top'),
                       Line2D([0], [0], color='g', marker = 'x', lw=0, label='Start'),
                       Line2D([0], [0], color='b', marker = 'x', lw=0, label='End'),
                       Line2D([0], [0], color='k', marker = 'x', lw=0, label='Diff.')]
    
    
    # plt.legend(handles=legend_elements)
    
    if illustrator == True:
        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])
        plt.gca().axes.set_xticks([0,40,80,120,160])
        plt.gca().axes.set_yticks([0,0.5,1.0,1.5,2])

    else:
        plt.xlabel('Cycle No.')
        plt.ylabel('Intensity /arb. units')
        plt.title('Distribution of particle intensities at the top of charge')
    
    plt.show()
    plt.savefig(path+'Figure 4a.pdf',bbox_inches='tight')
    
    return meansTop

path4 = r'D://NMC_degradation_3_160623_Halfthedata/Figures/Figure 4/'

meansInt = intensityHeterogeneity(path=path4,illustrator=True)


#%% Correlating intensity and time outliers with size
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
def outlierCorr(means, stds, nSigma=2, plot=False, topChargeTimes=topChargeTimes,echemChargeTimes=echemChargeTimes):
    '''
    NOTE: means and stds must be tuples in the order (meansInt, meansTime)
    '''
    from scipy.stats.stats import pearsonr

    #Toggle yVar with either 'intensity' or 'time'
    
    corrs = (np.empty([means[0].shape[0],3])*np.nan, np.empty([means[0].shape[0],3])*np.nan)    
    outlierInt = np.zeros([2,np.nanmax(echemDF['noParticles'].unique()).astype(int),cycleFrames[1][:].shape[0]])
    outlierTime = np.zeros(outlierInt.shape)

    cycleInd = 0
    # i --> cycle number
    for i in particleInfo['cycleNo'].unique().astype(int):
        print('cycle no.: ', i)
        
        arr = dm.unflatten(particleTraces[particleTraces['cycleNo']==i],'IntensitiesRaw')
        sizeArr = dm.unflatten(particleTraces[particleTraces['cycleNo']==i],'SizesPix')
            
        # j --> particle index
        for j in range(0,particleTraces[particleTraces['cycleNo']==i]['NumParticlesRaw'].unique()[0].astype(int)):
            
            # maxInt = np.nanmax(arr[j,:])
            
            arr[j,:] = arr[j,:]/arr[j,0]    
            maxInt = np.nanmax(arr[j,:]) #normalised intensities
            
            maxTime = topChargeTimes[j,cycleInd]-echemChargeTimes[cycleInd]
            
            for k in range(0,2):   
                if k == 0:
                    if maxInt >= means[k][cycleInd]+2*stds[k][cycleInd] or maxInt <= means[k][cycleInd]-2*stds[k][cycleInd]:
                        outlierInt[0,j,cycleInd] = maxInt
                        outlierInt[1,j,cycleInd] = np.nanmax(sizeArr[j,:])
                        #We select the LARGEST size value - strictly not exactly the same time point of intensity <--> size              
                    else:
                        pass
                else:
                    if maxTime >= means[k][cycleInd]+2*stds[k][cycleInd] or maxTime <= means[k][cycleInd]-2*stds[k][cycleInd]:
                        outlierTime[0,j,cycleInd] = maxTime
                        outlierTime[1,j,cycleInd] = np.nanmax(sizeArr[j,:])
                        #We select the LARGEST size value - strictly not exactly the same time point of intensity <--> size              
                    else:
                        pass
            

    
        #Create tuples to store values to be correlated
        tupleInt = (outlierInt[0,:,cycleInd][outlierInt[0,:,cycleInd]!=0],outlierInt[1,:,cycleInd][outlierInt[1,:,cycleInd]!=0])
        tupleTime = (outlierTime[0,:,cycleInd][outlierTime[0,:,cycleInd]!=0],outlierTime[1,:,cycleInd][outlierTime[1,:,cycleInd]!=0]) 

        
        for k in range(0,2):
            if k == 0:
                if tupleInt[0].shape[0] < 5:
                    corrs[k][cycleInd] = np.nan
                else:
                    corrs[k][cycleInd,0:2] = pearsonr(tupleInt[0],tupleInt[1])
                
                corrs[k][cycleInd,2] = tupleInt[0].shape[0] #record the number of particles used in calculation
            
            else:
                if tupleTime[0].shape[0] < 5:
                    corrs[k][cycleInd] = np.nan
                else:
                    corrs[k][cycleInd,0:2] = pearsonr(tupleTime[0],tupleTime[1])
                
                corrs[k][cycleInd,2] = tupleTime[0].shape[0] #record the number of particles used in calculation
        
        
        cycleInd +=1
    

    if plot == True:
        
        fig, ax = plt.subplots(1,1,figsize=[10,4])
        ylims = np.array([-1,1])
        
        #Shading upper cut-off voltage regions:
        plt.fill_between([-5,120], ylims[0], ylims[1], alpha=0.1, color='tab:green') #4.2V
        plt.fill_between([120, 136], ylims[0], ylims[1], alpha=0.1, color='tab:orange') #4.3V
        plt.fill_between([136, 170], ylims[0], ylims[1], alpha=0.1, color='tab:red') #4.3V
        plt.xlim(0,160)
        plt.ylim(ylims)
        
            
        idxInt = np.isfinite(corrs[0][:,0]) #identify non-nan value indices (polyfit doesnt accept nan)
        cycleVecInt = particleInfo['cycleNo'].unique().astype(int)[idxInt]
        
        idxTime = np.isfinite(corrs[1][:,0]) #identify non-nan value indices (polyfit doesnt accept nan)
        cycleVecTime = particleInfo['cycleNo'].unique().astype(int)[idxTime]
    
        
        fitInt=np.polyfit(cycleVecInt,corrs[0][idxInt,0],deg=1)
        fitTime=np.polyfit(cycleVecTime,corrs[1][idxTime,0],deg=1)

        def linFunc(fitRes, x):
            return fitRes[0]*x+fitRes[1]
            
            
        plt.plot(cycleVecInt, corrs[0][idxInt,0], 'xk')
        plt.plot(cycleVecInt, linFunc(fitInt,cycleVecInt), 'k')
            
        plt.plot(cycleVecTime, corrs[1][idxTime,0], 'xb')
        plt.plot(cycleVecTime, linFunc(fitTime,cycleVecTime), 'b')
        
        #Custom Legend
        legend_elements = [Line2D([0], [0], color='k', marker = 'x', lw=0, label='Int. Corr.'),
                           Line2D([0], [0], color='b', marker = 'x', lw=0, label='Time Corr.'),
                           mpatches.Patch(color='black',lw=0.1,linestyle='-',label='Int. Corr. - m={:.4f}'.format(fitInt[0])),
                           mpatches.Patch(color='blue',lw=0.1,linestyle='-',label='Time corr. - m={:.4f}'.format(fitTime[0]))]
        
        plt.legend(handles=legend_elements)

        plt.ylabel('Pearson Corr. Coeff.')
        
        plt.xlabel('Cycle Number')
        plt.tight_layout()
        
        
        
    else:
        pass
    
    
    
    
    return corrs, outlierInt, outlierTime

corrs, outlierInt, outlierTime = outlierCorr((meansInt,meansTime), (stdsInt,stdsTime), nSigma=1, plot=True) #means and stds variables come from intensityHeterogeneity() function



#%%
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

plt.figure()

ylims = np.array([-1,1])

#Shading upper cut-off voltage regions:
plt.fill_between([-5,120], ylims[0], ylims[1], alpha=0.1, color='tab:green') #4.2V
plt.fill_between([120, 136], ylims[0], ylims[1], alpha=0.1, color='tab:orange') #4.3V
plt.fill_between([136, 170], ylims[0], ylims[1], alpha=0.1, color='tab:red') #4.3V
plt.xlim(0,160)
plt.ylim(ylims)

idx = np.isfinite(corrs[:,0]) #identify non-nan value indices (polyfit doesnt accept nan)
cycleVec = particleInfo['cycleNo'].unique().astype(int)[idx]

fit=np.polyfit(cycleVec,corrs[idx,0],deg=1)

def linFunc(fitRes, x):
    return fitRes[0]*x+fitRes[1]
    
    
plt.plot(cycleVec, corrs[idx,0], 'xk')
plt.plot(cycleVec, linFunc(fit,cycleVec), 'b')
    
#Custom Legend
legend_elements = [Line2D([0], [0], color='k', marker = 'x', lw=0, label='Corr. Coeff.'),
                   mpatches.Patch(color='blue',lw=0.1,linestyle='-',label='Linear Fit - m={:.4f}'.format(fit[0]))]

plt.legend(handles=legend_elements)

plt.ylabel('Pearson Corr. Coeff. (intensity)')
plt.xlabel('Cycle Number')
plt.tight_layout()
plt.show()

#%% Spatial visualisation of top of charge time heterogeneity
    
'''
Currently not particle-linked

Add spatial plotting - trivial for single cycle, how to evaluate correlation through time??
'''

def spatialTimeHetero(meansTime=meansTime, stdsTime=stdsTime, particleTraces=particleTraces, matchedParticles=matchedParticles, echemChargeTimes=echemChargeTimes):
    # from PIL import Image
    
    # segMap = np.array(Image.open(r'D://NMC_degradation_3_160623_Halfthedata/Figures/2_cycle1_segmap.png'))
    
    # plt.figure()
    # plt.imshow(segMap,cmap='Greys_r')
    # plt.show()
    
    # deltaTimes = np.zeros([len(particleInfo[particleInfo['cycleNo']==2]['Unnamed: 0'].unique()),meansTime.shape[0]])
    deltaTimes = np.zeros([550,meansTime.shape[0]])*np.nan
    
    counter = 0
    
    #Single cycle only to start with:
    for cycleNo in particleInfo['cycleNo'].unique():
        # cycleNo = 2

        currCycle = particleTraces[particleTraces['cycleNo']==cycleNo]
        nFrames = int(currCycle.shape[0]/currCycle['NumParticlesRaw'].unique()[0])
        frameRate = currCycle['frameRate'].unique()[0]

        # particleIDs = np.array(matchedParticles[matchedParticles['cycleNo']==cycleNo]['Unnamed: 0'][:]).astype(int)
        
        for particleID in range(0,currCycle['NumParticlesRaw'].unique()[0]):
            # print(particleID)
            # particleID = np.where(particleIDs==particleID)[0][0]
            trace = np.array(currCycle['IntensitiesRaw'].iloc[particleID*nFrames:(particleID+1)*nFrames])
            
            #nan check:
            if np.isnan(trace).sum()>0:
                deltaTimes[particleID,counter]=np.nan
            else:     
                # print(trace)
                # print(counter)
                # print(cycleNo)
                # print(nFrames)
                # print(frameRate)
                # print(particleID)
                
                maxTime = (np.where(trace==np.amax(trace)))/frameRate
                print('break')
                print(maxTime)
                print(stdsTime[counter])
                print(echemChargeTimes[counter])
                # print((echemChargeTimes[counter]-maxTime[0][0])/stdsTime[counter])
                deltaTimes[particleID,counter] = (echemChargeTimes[counter]-maxTime[0][0])#/stdsTime[counter] #how many standard deviations from t0 did the particle hit max. intensity
            
        
        
        counter +=1

    return deltaTimes


deltaTimes = spatialTimeHetero()

#%% 

plt.figure()
plt.plot(deltaTimes[:,0],'xr')
plt.show()

#%%Figure 4 panels:

    


#%% Figure 5 panels:
import matplotlib as mpl

plt.rcParams['pdf.fonttype']=42
mpl.rcParams['figure.dpi'] = 108
mpl.rcParams['savefig.dpi'] = 108

# meansTime, stdsTime = timeHeterogeneity()
# meansInt, stdsInt = intensityHeterogeneity(topChargeInts=topChargeIntsRaw)
# corrsInt, corrsTime, outlierParticles = outlierCorr()


#%% Figure S1 panels validating the :


#%% Gamma distribution fitting:
from matplotlib import cm
from matplotlib.colors import Normalize
import scipy.stats as stats
from scipy.optimize import curve_fit    
from matplotlib import ticker

def gammaFunc(x,k,theta):
    from scipy.special import gamma
    #when k=1, we retrieve the exponential func.
    
    return ((1/gamma(k)*theta**k)*(x**(k-1))*(np.exp(-x/theta)))

def bigammaFunc(x,k1,k2,t1,t2):
    return gammaFunc(x,k1,t1)+gammaFunc(x,k2,t2)

def func(x,a,b):
    return a*np.exp(-b*x)

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
# mpl.rcParams['text.usetex'] = True

cycles = [2,50,125,136,156]
pixSize = 95*1e-3 #nanometres per pixel converted to microns

norm = Normalize(vmin=0,vmax=160)#cmap set-up
cmap = cm.get_cmap('viridis', np.amax(echemDF['cycleNo'].unique()[:])-np.amin(echemDF['cycleNo'].unique()[:])+1)


fig,ax=plt.subplots(1,1,figsize=[6,4])
plt.tight_layout()
plt.setp(ax.spines.values(), linewidth=1.5)
plt.tick_params(width=1.5)

# cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),label='Cycle Number') #WITH NUMBERS
cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),format = ticker.FuncFormatter(lambda x, pos: '')) #WITHOUT NUMBERS

cbar.set_ticks([0,80,160]) 
cbar.ax.tick_params(width=1.5)
cbar.outline.set_linewidth(1.5)   

for i in cycles:
    print(i)
    series = np.sort(particleInfo[particleInfo['cycleNo'] == i]['npix'])
    binNum = int((series.max()-series.min())/10)
    bins = np.linspace(series.min(),series.max(),binNum)
    midBins  = (bins[1:] + bins[:-1])/2 # shifts the x values to be centred in the bins for fitting and plotting 
    histo = np.histogram(series,bins)
    
    
    #convenience
    xdata = midBins[0:90]
    ydata = histo[0][0:90]
    # xdata = midBins
    # ydata = histo[0]
    
    # ydata = np.where(ydata==0, np.nan, ydata)
    
    
    #initial guesses:
    mean = ydata.mean() #for gammaFunc mean=k*theta
    var = ydata.var() #for gammaFunc var = k*theta^2
    p0=[0,0]
    p0[1] = var/mean #theta = k*theta^2/k*theta (=var/mean)
    p0[0] = mean/p0[1]
    
    popt,pcov = curve_fit(gammaFunc,xdata,ydata,p0=p0,maxfev=10000,method='lm') #y+1 to remove zeros, undo during plotting
    # popt2,pcov2 = curve_fit(bigammaFunc,xdata,ydata+1,p0=[p0+[0.5,15.5]],maxfev=10000)
    # popt,pcov = curve_fit(func,xdata,ydata+1,p0=p0,maxfev=10000,method='lm') #y+1 to remove zeros, undo during plotting
    
    rSquared = 1-(np.sum((ydata - gammaFunc(xdata,*popt)) ** 2))/(np.sum((ydata - np.mean(ydata)) ** 2))
    
    # x=np.linspace(series.min(),series.max(),1000)
    x=np.linspace(xdata.min(),xdata.max(),1000)
    
    
    plt.plot(xdata*pixSize**2,gammaFunc(xdata,*popt),'-',color=cmap.colors[int(i)],label='k={:.2f},theta={:.2f}, R^2={:.2f}'.format(popt[0],popt[1], rSquared))
    
    if i == 2:
        plt.plot(xdata*pixSize**2,ydata,'xk', alpha=0.5) 
    else:
        pass
    
    # plt.plot(x,func(x,*popt))
    # plt.plot(x,bigammaFunc(x,*popt2)-1,'--b',label='bigammaFunc')
    
    print(rSquared)


    
    
# plt.yscale('log')

# plt.xlabel('facet size / um^2')
# plt.ylabel('frequency')

plt.legend()

plt.axvline(25*pixSize**2,linestyle='--',color='black')

illustrator=True
if illustrator == True:
    
        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])
    
        plt.gca().axes.set_xticks([0,2,4,6,8])

        plt.gca().axes.set_yticks([0,20,40,60])
else:
    pass

plt.show()

plt.savefig(pathS1+'Figure S1c - legend.pdf',bbox_inches='tight')

#%% Figure 1 panel d:
import sys
sys.path.append("C://Users/alek/Desktop/analysis")

import DataManipulation as dm

meanArr = np.zeros([matchedParticles['cycleNo'].unique().shape[0],1300])*np.nan
exParts = np.zeros([2,1300])*np.nan #second dim bigger than whatever the longest cycle is

counter = 0
for cycleNo in matchedParticles['cycleNo'].unique():

    arr = dm.unflatten(particleTraces[particleTraces['cycleNo']==cycleNo],'CutoutIntensitiesRaw')
    meanArr[counter,0:arr.shape[1]] = np.nanmean(arr,axis=0)
    counter+=1
    
    
    #%%
plt.figure()
# for i in range(0,meanArr.shape[0]):
plt.plot(meanArr[2])
    
plt.show()

    
    
    