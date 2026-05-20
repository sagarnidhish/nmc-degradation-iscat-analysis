# -*- coding: utf-8 -*-
"""
Ordering of raw or stabilised optical data using computer time

Created on Thu Oct 19 11:09:14 2023

@author: Alek
"""

import numpy as np
import tables
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
import sys
from glob import glob
from collections import defaultdict


expDir = "D://NMC_degradation_3_160623_Halfthedata/"

choppedDirs = glob(expDir+"*/", recursive = True)
choppedDirs.pop(-1)

choppedDirs=choppedDirs[:-1] #removes the Figures folder from choppedDirs

#Ordering of the chopped data folders using computer time
expOrder = {}

for i in range(0,len(choppedDirs)): #Iterate through chopped folders
    
    expNames = glob(choppedDirs[i]+'*.hdf5') 
    # print(expNames)
    with tables.File(expNames[0],'r') as f: #Only open cycle0 file
    
        expOrder[choppedDirs[i]] = f.root.camera_timing[0,0]/1e7/3600 #conversion to hours

print('Iterated through chopped folders')

#Sort dict by value
expOrder = sorted(expOrder.items(), key = lambda x:x[1])
expOrder = dict(expOrder)
    
#Load electrochemistry into a single dataframe to minimise I/O overheads:
nRows = 0
cycleNo = 0


def findExpNames(key):
    tempExpNames = glob(key+'*.hdf5') 
    expNames = []
    
    #Remove any "Output - ..." folders from expNames (their paths contain '.hdf5')
    for i in range(0,len(tempExpNames)):
        if 'Output' in tempExpNames[i]:
            pass
        else:
            expNames.append(tempExpNames[i])
    
    return expNames

for key in expOrder.keys():
    
    expNames = findExpNames(key)
    print(key)
    
    if expOrder[key] == 451666.32764634304: #This accounts for the corrupt 3_c2_x10 file which cant be recovered.
        cycleNo += 10
        
    else:
        pass
    
    for exp in expNames:

        with tables.File(exp,'r') as f:
            
            tf = pd.read_hdf(exp)            
            nRows += tf.shape[0]
            
            if cycleNo == 0:
                tf.insert(tf.shape[1], 'cycleNo', np.zeros(tf.shape[0]))
                tf.insert(tf.shape[1], 'addrs', "")
                cols = tf.columns
                
            else:
                pass



# print(nRows)
echemDF = pd.DataFrame(index=range(nRows), columns=cols)
# print(echemDF.shape, echemDF.columns)
#%% Populate electrochemistry dataframe with values from all cycles
nRows = 0
cycleNo = 0

for key in expOrder.keys():
    
    expNames = findExpNames(key)
    print(key)
    
    if expOrder[key] == 451666.32764634304: #This accounts for the corrupt 3_c2_x10 file which cant be recovered.
        cycleNo += 10
        
    else:
        pass
    
    for exp in expNames:

        with tables.File(exp,'r') as f:
            
            tf = pd.read_hdf(exp)    
            tf.insert(tf.shape[1], 'cycleNo', cycleNo)
            tf.insert(tf.shape[1], 'addrs', exp)
            # tf['cycleNo'][:] = cycleNo
            for key in cols:
                
                echemDF[key][nRows:nRows+tf.shape[0]] = tf[key][:]
                
            
            nRows += tf.shape[0]
            cycleNo +=1

#%% Save full echemDF

#Full dataframe ~2GB
echemDF.to_csv('D://NMC_degradation_3_160623_Halfthedata/echemDF_full.csv')


#%% Plotting initial, final and max average intensity values for all cycles:

def meanIntScatter(realTime = False,sparsity=1,norm=False, illustrator=False, path=None, size=[6,4]):
   
    
    cycleNo = 0
    mSize = 15
    mThick = 2.5
    

    
    lw=1.5*1.6

    size[0] = size[0]*1.6*2
    size[1] = size[1]*1.6
    
    fig,ax = plt.subplots(figsize=(size[0]-0.5,size[1]))    

    
    #custom settings for publication figures: (no colourbar variant, single pair of axes)
    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=lw)
    plt.tick_params(width=lw,length=5)
    
    
    if illustrator == True:
        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])
    else:
        plt.title('Initial, Final and Max. Intensities for each cycle')
        # plt.xlabel('unix time /us')
        plt.ylabel('Mean intensity /arb. units')
        
        if realTime == True:
            plt.xlabel('Time /hours')
        else:
            plt.xlabel('Cycle No.')
        
    
    for key in expOrder.keys():
        
        expNames = findExpNames(key)
        print(key)
        
        if expOrder[key] == 451666.32764634304: #This accounts for the corrupt 3_c2_x10 file which cant be recovered.
            cycleNo += 10
            
        else:
            pass
        
        for exp in expNames:
                
            with tables.File(exp,'r') as f:
                if cycleNo == 0:
                    initTime = f.root.camera_timing[0,0]/1e7/1e6
                    # initTime=0
                    
                    if norm == True:
                        startFactor = f.root.average_intensity[0]
                        # endFactor = startFactor
                        # topFactor = startFactor
                        endFactor = f.root.average_intensity[-1]
                        topFactor = np.amax(f.root.average_intensity[:])
                        
                    else:
                        startFactor = 1
                        endFactor = 1
                        topFactor = 1
                else:
                    pass
    
                # print(startF)
                
                if realTime == True:
                    # plotting with real time axis:        
                    if cycleNo%sparsity == 0:
                        plt.plot((f.root.camera_timing[0,0]/1e7/1e6)-initTime, f.root.average_intensity[0]/startFactor,'xr',ms=mSize,mew=mThick) 
                        plt.plot((f.root.camera_timing[0,-1]/1e7/1e6)-initTime, f.root.average_intensity[-1]/endFactor,'xb',ms=mSize,mew=mThick) 
                        plt.plot((f.root.camera_timing[0,np.argmax(f.root.average_intensity == np.amax(f.root.average_intensity[:]))]/1e7/1e6)-initTime, np.amax(f.root.average_intensity[:])/topFactor,'xg',ms=mSize,mew=mThick) 
                    else:
                        pass
                else:
                    # plotting with cycle no. time axis:
                    if cycleNo%sparsity == 0:
                        plt.plot(cycleNo, f.root.average_intensity[0]/startFactor,'xr',ms=mSize,mew=mThick) 
                        plt.plot(cycleNo, f.root.average_intensity[-1]/endFactor,'xb',ms=mSize,mew=mThick) 
                        plt.plot(cycleNo, np.amax(f.root.average_intensity[:])/topFactor,'xg',ms=mSize,mew=mThick) 
                    # print(f.root.average_intensity[:].shape)
                    else:
                        pass
 
                cycleNo +=1
                    
                # print((f.root.camera_timing[0,1]-f.root.camera_timing[0,0])/1e7/3600)
                # print((f.root.camera_timing[1,1]-f.root.camera_timing[1,0]))
                camTime = f.root.camera_timing[:,:]


    
    #Shading upper cut-off voltage regions:
    plt.fill_between([-5,120], 7800, 8500, alpha=0.1, color='tab:green') #4.2V
    plt.fill_between([120, 136],  7800, 8500, alpha=0.1, color='tab:orange') #4.3V
    plt.fill_between([136, 165], 7800, 8500, alpha=0.1, color='tab:red') #4.3V

    ax.axvline(120, color='k', linestyle='--',lw=lw)
    ax.axvline(136, color='k', linestyle='--',lw=lw)

    # Custom axis limits
    plt.xlim([-5,165])
    plt.ylim([7800,8500])
    
    #Custom Legend
    legend_elements = [Line2D([0], [0], color='r', marker = 'x', lw=0, label='Start Charge'),
                       Line2D([0], [0], color='b', marker = 'x', lw=0, label='End Discharge'),
                       Line2D([0], [0], color='g', marker = 'x', lw=0, label='Top of Charge')]
    
    
    # plt.legend(handles=legend_elements)
    
    if illustrator == True:
        plt.savefig(path+'Figure 3c.pdf',bbox_inches='tight')
    else:
        pass

    plt.show()

#%%
# Plotting capacity and coulombic efficiency:

def CapCoul(activeMass=4.77, firstCycles = True, illustrator=False, path=None, size=[6,4]):
    
    
    #marker settings:
    ms = 10
    mew = 2
    
    lw=1.5*1.6

    size[0] = size[0]*1.6*2
    size[1] = size[1]*1.6

    fig, ax1 = plt.subplots(figsize=(size[0]-0.5,size[1]))
    
    plt.setp(ax1.spines.values(), linewidth=lw)
    plt.tight_layout()
    ax1.set_xlim([0,165])
    ax1.set_ylim([100,200])
    ax1.tick_params(width=lw,length=5)

    ax2 = ax1.twinx()
    ax2.set_ylim(90,105)
    ax2.set_yticks([90,92,94,96,98,100])
    ax2.tick_params(width=lw, length=5)


    
    if illustrator == True:
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        
    else:
        plt.title('Capacity and Coulombic Efficiency as a Function of Cycling where Upper Cut-off Voltage Increases')
        ax1.set_xlabel('Cycle No.')
        ax1.set_ylabel('Capacity / mAh g-1')
        ax2.set_ylabel('Coulombic Efficiency /%')
        
    coulEff = np.array([])
    
    cycleNo = 0
    cycleInd = 0
    
    def CalcCapacity(Time, Current, ActiveMass):
            from scipy.integrate import cumtrapz
            Capacity = cumtrapz(Current, x = Time,initial=0)
            Capacity = (Capacity/3600)/(ActiveMass/1000) #convert s to hours and mg to grams
            
            return Capacity
            
    for key in expOrder.keys():
        
        exlCycle = 0 #exlCycle is a counter used to exclude the first cycle of each group from plotting
        expNames = findExpNames(key)
        print(key)
    
        if expOrder[key] == 451666.32764634304: #This accounts for the corrupt 3_c2_x10 file which cant be recovered.
        #This should be done with a string comparison but the way glob returns paths includes escape characters
            cycleNo += 10
        else:
            pass
        
        for exp in expNames:
            with tables.File(exp,'r') as f:
    
                # plotting with cycle no. time axis:
                # plt.plot(cycleNo, f.root.average_intensity[0],'xr') 
                # plt.plot(cycleNo, f.root.average_intensity[-1],'xb') 
                # plt.plot(cycleNo, np.amax(f.root.average_intensity[:]),'xg') 
                if firstCycles == True:
                    tf = pd.read_hdf(exp)
                    cap = CalcCapacity(tf['Time (s)'][:].astype(float), tf['Current (mA)'][:], activeMass)
                    
                    chargeCap = np.amax(cap) - cap[0]
                    disCap = abs(cap[-1] - np.amax(cap))
                    coulEff = np.append(coulEff,[100*disCap/chargeCap])
                    
                    ax1.plot(cycleNo, chargeCap, 'xr', ms=ms,mew=mew)
                    ax1.plot(cycleNo, disCap, 'xb', ms=ms,mew=mew)
                    ax2.plot(cycleNo, coulEff[cycleInd], 'xk', ms=ms,mew=mew)
                    cycleInd +=1

                    
                else:
                    if exlCycle == 0:
                        pass
                    else:
                        tf = pd.read_hdf(exp)
                        cap = CalcCapacity(tf['Time (s)'][:].astype(float), tf['Current (mA)'][:], activeMass)
                        
                        chargeCap = np.amax(cap) - cap[0]
                        disCap = abs(cap[-1] - np.amax(cap))
                        coulEff = np.append(coulEff,[100*disCap/chargeCap])
            
                        
                        ax1.plot(cycleNo, chargeCap, 'xr', ms=ms,mew=mew)
                        ax1.plot(cycleNo, disCap, 'xb', ms=ms,mew=mew)
                        ax2.plot(cycleNo, coulEff[cycleInd], 'xk', ms=ms,mew=mew)
                        cycleInd +=1

                print('Cycle Number: ',cycleNo)                
                cycleNo +=1
                exlCycle =1
                
                
    
    #Shading upper cut-off voltage regions:
    ax1.fill_between([0,120], 0, 250, alpha=0.1, color='tab:green') #4.2V
    ax1.fill_between([120, 136], 0, 250, alpha=0.1, color='tab:orange') #4.3V
    ax1.fill_between([136, 165], 0, 250, alpha=0.1, color='tab:red') #4.3V
    
    ax1.axvline(120, color='k', linestyle='--',lw=lw)
    ax1.axvline(136, color='k', linestyle='--',lw=lw)
    ax2.axhline(100, color='k', linestyle='--',lw=lw)

    
    legend_elements = [Line2D([0], [0], color='r', marker = 'x', lw=0, label='Charge', ms=ms,mew=mew),
                       Line2D([0], [0], color='b', marker = 'x', lw=0, label='Discharge', ms=ms,mew=mew),
                       Line2D([0], [0], color='k', marker = 'x', lw=0, label='Coul. Eff.', ms=ms,mew=mew)]
    
    
    # plt.legend(handles=legend_elements)
    
    plt.show()
    # plt.savefig(path+'Figure 2c.png',bbox_inches='tight')
    plt.savefig(path+'Figure 2c.pdf',bbox_inches='tight')

    return coulEff

# Plotting Potential vs. Capacity:
activeMass = 4.77 #mass in mg

from matplotlib import cm
from matplotlib.colors import Normalize

def CalcCapacity(Time, Current, ActiveMass):
        from scipy.integrate import cumtrapz
        Capacity = cumtrapz(Current, x = Time,initial=0)
        Capacity = (Capacity/3600)/(ActiveMass/1000) #convert s to hours and mg to grams
        
        return Capacity #mAh g-1
#%%
def PotVsCap(activeMass=4.77, cycleSparsity = [10,1,2],illustrator=False, path=None):
    fig, axs = plt.subplots(1,3,figsize=(6,4))
    cycleNo = 0
    counter = 0
    cmap = cm.get_cmap('viridis', np.amax(echemDF['cycleNo'].unique()[:])-np.amin(echemDF['cycleNo'].unique()[:])+1)
    norm = Normalize(vmin=0,vmax=160)


    plt.tight_layout()

    if illustrator == False:
        fig.suptitle('Potential (V) vs. Capacity (mAh g-1)')
        axs[0].set_title('4.2V Cut-off')
        axs[1].set_title('4.4V Cut-off')
        axs[2].set_title('4.6V Cut-off')
    
        axs[1].set_xlabel('Capacity /mAh g-1')
        axs[0].set_ylabel('Potential /V')
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),label='Cycle Number')
        cbar.set_ticks([0,80,160]) 
        
    else:
        for i in range(0,3):
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
            
    #Shading upper cut-off voltage regions:
    axs[0].fill_between([0,200], 0, 250, alpha=0.1, color='tab:green') #4.2V
    # plt.txt((120--5)/2, 190, 'test')
    axs[1].fill_between([0,200], 0, 250, alpha=0.1, color='tab:orange') #4.3V
    axs[2].fill_between([0,200], 0, 250, alpha=0.1, color='tab:red') #4.3V
    
    
    # Dynamic or static y-axis:
    for i in range(axs.shape[0]):
        # axs[i].set_ylim(2.9,4.3+i*0.2)
        axs[i].set_ylim(2.9,4.7)
        axs[i].set_yticks(np.arange(3,5,0.4))
        axs[i].set_xlim(0,200)
        axs[i].set_xticks(np.arange(0,250,50))
        axs[i].tick_params(width=1.5)
        plt.setp(axs[i].spines.values(), linewidth=1.5)

        # axs[i].axhline(y=4.2,xmin=0,xmax=100,linestyle='--', color='black')
        # axs[i].axhline(y=4.4,xmin=0,xmax=100,linestyle='--', color='black')
        # axs[i].axhline(y=4.6,xmin=0,xmax=100,linestyle='--', color='black')
        # axs[i].axhline(y=3.0,xmin=0,xmax=100,linestyle='--', color='black')

    
    #Remove whitespace between vertical axes
    fig.subplots_adjust(wspace=0)

    #remove duplicated ticks and axis numbers
    axs[0].tick_params(labelbottom=False)
    axs[2].tick_params(labelbottom=False)
    axs[1].tick_params(left=False,labelleft=False)
    axs[2].tick_params(left=False,labelleft=False)
        
    
        
    #Evaluate cycle no. lists once drastically speeds up performance:
    a = echemDF['cycleNo'].unique()[2:110:cycleSparsity[0]]
    b = echemDF['cycleNo'].unique()[123-10:135-10:cycleSparsity[1]]
    c = echemDF['cycleNo'].unique()[136-10:159-10:cycleSparsity[2]]
    
    for cycleNo in a:
        
        tf = echemDF[echemDF['cycleNo'] == cycleNo]
        currKeys = tf['Block'].unique()
         
        cap = CalcCapacity(tf[tf['Block']==currKeys[0]]['Time (s)'][:].astype(float), tf[tf['Block']==currKeys[0]]['Current (mA)'][:], activeMass)
        axs[0].plot(cap, tf[tf['Block']==currKeys[0]]['Potential (V)'][:], color=cmap.colors[int(cycleNo)])
        
        cap = CalcCapacity(tf[tf['Block']==currKeys[1]]['Time (s)'][:].astype(float), tf[tf['Block']==currKeys[1]]['Current (mA)'][:], activeMass)
        axs[0].plot(abs(cap), tf[tf['Block']==currKeys[1]]['Potential (V)'][:], color=cmap.colors[int(cycleNo)])
        
        
        # capDF = pd.DataFrame(cap)
        # sns.lineplot(data=tf[tf['Block']==currKeys[0]], x = 'Time (s)', y = 'Potential (V)', hue='cycleNo', palette = 'rocket')
        # sns.lineplot(data=tf)
        print(cycleNo)
        counter +=1
        
    print('break')
    
    
    for cycleNo in b:
        
        tf = echemDF[echemDF['cycleNo'] == cycleNo]
        currKeys = tf['Block'].unique()
        
        cap = CalcCapacity(tf[tf['Block']==currKeys[0]]['Time (s)'][:].astype(float), tf[tf['Block']==currKeys[0]]['Current (mA)'][:], activeMass)
        axs[1].plot(cap, tf[tf['Block']==currKeys[0]]['Potential (V)'][:], color=cmap.colors[int(cycleNo)])
        
        cap = CalcCapacity(tf[tf['Block']==currKeys[1]]['Time (s)'][:].astype(float), tf[tf['Block']==currKeys[1]]['Current (mA)'][:], activeMass)
        axs[1].plot(abs(cap), tf[tf['Block']==currKeys[1]]['Potential (V)'][:], color=cmap.colors[int(cycleNo)])
        # capDF = pd.DataFrame(cap)
        # sns.lineplot(data=tf[tf['Block']==currKeys[0]], x = 'Time (s)', y = 'Potential (V)', hue='cycleNo', palette = 'rocket')
        # sns.lineplot(data=tf)
        print(cycleNo)
    print('break')
    
    
    for cycleNo in c:
        
        tf = echemDF[echemDF['cycleNo'] == cycleNo]
        currKeys = tf['Block'].unique()
        
        cap = CalcCapacity(tf[tf['Block']==currKeys[0]]['Time (s)'][:].astype(float), tf[tf['Block']==currKeys[0]]['Current (mA)'][:], activeMass)
        axs[2].plot(cap, tf[tf['Block']==currKeys[0]]['Potential (V)'][:], color=cmap.colors[int(cycleNo)])
        
        cap = CalcCapacity(tf[tf['Block']==currKeys[1]]['Time (s)'][:].astype(float), tf[tf['Block']==currKeys[1]]['Current (mA)'][:], activeMass)
        axs[2].plot(abs(cap), tf[tf['Block']==currKeys[1]]['Potential (V)'][:], color=cmap.colors[int(cycleNo)])
        # capDF = pd.DataFrame(cap)
        # sns.lineplot(data=tf[tf['Block']==currKeys[0]], x = 'Time (s)', y = 'Potential (V)', hue='cycleNo', palette = 'rocket')
        # sns.lineplot(data=tf)
        print(cycleNo)
    
    plt.show()
    
    if illustrator==True:
        plt.savefig(path+'Figure 2a.pdf',bbox_inches='tight')
    else:
        pass
    
# PotVsCap(activeMass = 4.77, cycleSparsity = [50,20,5],illustrator=True,path=path)

#%%
# Plotting Capacity vs. Mean Intensity (entire frame)
from scipy.signal import savgol_filter
from math import ceil, floor

def IntVsCap(normInt = False, activeMass = 4.77, cycleSparsity = [10,1,2], window_length = 10, xAxis = 'cap', grad=False, illustrator=False, path=None):
    from matplotlib import ticker
    '''
    1. Extract echem dataframe for single cycle
    2. Find start and end times of echem blocks (computer time)
    3. Extract corresponding mean intensity trace from file (split into echem blocks)
    4. Interpolate echem (time, current and potential vectors) onto the more sparse intensity time vector
    5. Calculate capacity
    6. Plot intensity vs. capacity and potential in same manner as potential vs. capacity plot with subplots for cut-off potentials.
    '''
    
    from matplotlib import cm
    from matplotlib.colors import Normalize
    
    fig, axs = plt.subplots(1,3, figsize=[6,4])
    
    cmap = cm.get_cmap('viridis', np.amax(echemDF['cycleNo'].unique()[:])-np.amin(echemDF['cycleNo'].unique()[:])+1)
    norm = Normalize(vmin=0,vmax=160)
    
    plt.tight_layout()
    for i in range(0,3):
        plt.setp(axs[i].spines.values(), linewidth=1.5)
        axs[i].tick_params(width=1.5)

    
    
    if illustrator == True:
        for i in range(0,3):
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),format = ticker.FuncFormatter(lambda x, pos: ''))

    else:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),label='Cycle Number')
        axs[0].set_title('4.2V Cut-off')
        axs[1].set_title('4.4V Cut-off')
        axs[2].set_title('4.6V Cut-off')
                 
    cbar.set_ticks([0,80,160])  
    cbar.ax.tick_params(width=1.5)
    cbar.outline.set_linewidth(1.5)   
    
    if xAxis == 'cap':
        
        
        if illustrator==False:
            fig.suptitle('Mean Intensity (arb. units) vs. Capacity (mAh g-1)')
            axs[1].set_xlabel('Capacity / mAh g-1')
        else:
            pass
        
    elif xAxis == 'pot':
        if grad == False:
            if illustrator==False:

                fig.suptitle('Mean Intensity (arb. units) vs. Potential (V)')
                axs[1].set_xlabel('Potential / V')
            else:
                pass
            
        elif grad == True:
            if illustrator==False:

                fig.suptitle('Derivative Mean Intensity (arb. units) vs. Potential (V) - dInt/dV')
                axs[1].set_xlabel('Potential / V')
            else:
                pass
        else:
            ValueError
    else:
        ValueError('Set key = cap or pot')
  
 
    cycleNo = 0
    
    #Evaluate cycle no. lists once drastically speeds up performance:
    cycleList = echemDF['cycleNo'].unique()[2:110:cycleSparsity[0]]
    cycleList = np.append(cycleList, echemDF['cycleNo'].unique()[123-10:135-10:cycleSparsity[1]])
    cycleList = np.append(cycleList, echemDF['cycleNo'].unique()[136-10:159-10:cycleSparsity[2]])
    
    for key in expOrder.keys(): #Iterate over each recorded dataset (blocks of 10 cycles typically but not always)
        counter = 0 #used to exclude the first cycle of each dataset
        expNames = findExpNames(key) #list of individual cycles in a folder
        
        if expOrder[key] == 451666.32764634304: #This accounts for the corrupt 3_c2_x10 file which cant be recovered.
        #This should be done with a string comparison but the way glob returns paths includes escape characters
            cycleNo += 10
        else:
            pass
        
        for exp in expNames:
            if counter == 0:
                pass
            else:
                with tables.File(exp,'r') as f:
                    if cycleNo in cycleList:
                        print(cycleNo)
                        
                        #Extract echem based on the cycle number:
                        tf = echemDF[echemDF['cycleNo'] == cycleNo] 
                        currKeys = tf['Block'].unique()
                        
                        for currKey in currKeys:
                            '''
                            1. Find time elapsed from start to ToC, and ToC to end
                            2. Match to intensity trace
                            3. Interpolate echem (pot/curr --> cap) onto intensity basis
                            '''
                            # print(currKey)
        
                            camTime = f.root.camera_timing[1,:]
                            
                            tStartEchem = float(tf[tf['Block']==currKey]['Computer time (s)'].iloc[0])
                            tEndEchem = float(tf[tf['Block']==currKey]['Computer time (s)'].iloc[-1])
                            echemTime = camTime[np.abs(camTime-tStartEchem).argmin():np.abs(camTime-tEndEchem).argmin()]
        
                            intVec = f.root.average_intensity[np.abs(camTime-tStartEchem).argmin():np.abs(camTime-tEndEchem).argmin()]
                            
                            # print('cam time ', (camTime[np.abs(camTime-tEndEchem).argmin()]-camTime[np.abs(camTime-tStartEchem).argmin()])) #same as unix time (from camera_timing) to multiple decimal places
                            
                            #Interpolate echem onto intensity time axes
                            potVec = np.interp(echemTime, np.asarray(tf[tf['Block']==currKey]['Computer time (s)'][:]).astype(float), np.asarray(tf[tf['Block']==currKey]['Potential (V)'][:]).astype(float))
                            
                            currVec = np.interp(echemTime, np.asarray(tf[tf['Block']==currKey]['Computer time (s)'][:]).astype(float), np.asarray(tf[tf['Block']==currKey]['Current (mA)'][:]).astype(float))
                            capVec = abs(CalcCapacity(echemTime, currVec, activeMass))
            
                            #Smoothing - aesthetic 
                            potVec = savgol_filter(potVec, window_length=window_length, polyorder= 1)
                            currVec = savgol_filter(currVec, window_length=window_length, polyorder= 1)
                            capVec = savgol_filter(capVec, window_length=window_length, polyorder= 1)
                            intVec = savgol_filter(intVec, window_length=window_length, polyorder= 1)
        

                            if normInt == True:
                                intVec = intVec/np.amin(intVec)
                            else:
                                pass
                                
                            if xAxis == 'cap':
                                
                                if cycleNo < 121:
                                    axs[0].plot(capVec, intVec, color=cmap.colors[int(cycleNo)])
                                    
                                elif cycleNo < 135:
                                    axs[1].plot(capVec, intVec, color=cmap.colors[int(cycleNo)])
                                    
                                else:
                                    axs[2].plot(capVec, intVec, color=cmap.colors[int(cycleNo)])
                                    
                            elif xAxis == 'pot':
                                # intVec = np.gradient(intVec)
                                if cycleNo < 121:
                                    axs[0].plot(potVec, intVec, color=cmap.colors[int(cycleNo)])
                                    
                                elif cycleNo < 135:
                                    axs[1].plot(potVec, intVec, color=cmap.colors[int(cycleNo)])
                                    
                                else:
                                    axs[2].plot(potVec, intVec, color=cmap.colors[int(cycleNo)])
                            else:
                                ValueError
                 
                    else:
                        pass
            counter += 1
            cycleNo += 1
            
            
    #Set y-limits/ticks:
    if grad == False:
        if normInt == True:
            if illustrator==False:
                axs[0].set_ylabel('Normalised Mean Intensity /arb. units')
            else:
                pass
            for i in range(axs.shape[0]):
                axs[i].set_ylim(1,np.amax(intVec/np.amin(intVec))+0.005)
                
        elif normInt == False:
            
            if illustrator==False:
                axs[0].set_ylabel('Mean Intensity /arb. units')
            else:
                pass
            for i in range(axs.shape[0]):
                axs[i].set_ylim(floor(np.amin(intVec)/100)*100, 1.01*ceil((np.amax(intVec))/100)*100)
                axs[i].set_yticks(np.arange(floor(np.amin(intVec)/100)*100, 1.01*ceil(np.amax(intVec)/100)*100,100))
                
    elif grad == True:
            if illustrator==False:
                axs[0].set_ylabel('Derivative Mean Intensity - dInt/dV /arb. units')
            else:
                pass
    else:
        ValueError
            
    #Remove whitespace between vertical axes
    fig.subplots_adjust(wspace=0)

    #remove duplicated ticks and axis numbers
    axs[0].tick_params(labelbottom=False)
    axs[2].tick_params(labelbottom=False)
    axs[1].tick_params(left=False,labelleft=False)
    axs[2].tick_params(left=False,labelleft=False)
        
    if xAxis == 'pot':
        #Shading upper cut-off voltage regions:
        axs[0].fill_between([3,4.6], 1, np.amax(intVec/np.amin(intVec))+0.005, alpha=0.1, color='tab:green') #4.2V
        # plt.txt((120--5)/2, 190, 'test')
        axs[1].fill_between([3,4.6], 1, np.amax(intVec/np.amin(intVec))+0.005, alpha=0.1, color='tab:orange') #4.3V
        axs[2].fill_between([3,4.6], 1, np.amax(intVec/np.amin(intVec))+0.005, alpha=0.1, color='tab:red') #4.3V
        
        
        #Set static xlims (same for all panels)
        for i in range(0,axs.shape[0]):
            axs[i].set_xticks(np.arange(3.0,4.8,0.4))
            axs[i].set_xlim([3.0,4.6])
    else:
        
        #Shading upper cut-off voltage regions:
        cols = ['tab:green', 'tab:orange', 'tab:red']
        
        for i in range(0,axs.shape[0]):
            axs[i].fill_between([0,200],1,np.amax(intVec)*1.005, alpha=0.1, color=cols[i])
            axs[i].set_xticks(np.arange(0,250,50))
            axs[i].set_xlim([0,200])
            axs[i].set_ylim([1,np.amax(intVec)*1.005])

    
    plt.show()
    
    if illustrator ==True:
        
        if xAxis == 'pot':
            plt.savefig(path+'Figure 3b - time.pdf', bbox_inches='tight')
        else:
            plt.savefig(path+'Figure 3b - cap.pdf', bbox_inches='tight')

    else:
        pass
    
#%% Differential plotting - dQ/dV

'''
https://www.frontiersin.org/articles/10.3389/fenrg.2022.1023555/full
'''

def dqdvSingle(capacity, voltage, 
                    polynomial_spline=3, s_spline=1e-5,
                    polyorder_1 = 5, window_size_1=101,
                    polyorder_2 = 5, window_size_2=1001,
                    final_smooth=True):

    import pandas as pd
    import numpy as np
    from scipy.interpolate import splrep, splev

    df = pd.DataFrame({'Capacity': capacity, 'Voltage':voltage})
    unique_v = df.astype(float).groupby('Voltage').mean().index
    unique_v_cap = df.astype(float).groupby('Voltage').mean()['Capacity']

    x_volt = np.linspace(min(voltage), max(voltage), num=int(1e4))
    f_lit = splrep(unique_v, unique_v_cap, k=1, s=0.0)
    y_cap = splev(x_volt, f_lit)
    smooth_cap = savgol_filter(y_cap, window_size_1, polyorder_1)

    f_smooth = splrep(x_volt, smooth_cap, k=polynomial_spline, s=s_spline)
    dqdv = splev(x_volt, f_smooth, der=1)
    smooth_dqdv = savgol_filter(dqdv, window_size_2, polyorder_2)
    
    if final_smooth:
        return x_volt, smooth_dqdv, smooth_cap
    else:
        return x_volt, dqdv, smooth_cap



def dqdvMulti(echemDF, cycleSparsity = [10,2,2],illustrator=False,path=None):
    
    from matplotlib import ticker
    
    fig,ax = plt.subplots(figsize=[6,4])
    cycles = echemDF['cycleNo'].unique()[2:110:cycleSparsity[0]]
    cycles = np.append(cycles, echemDF['cycleNo'].unique()[123-10:135-10:cycleSparsity[1]])
    cycles = np.append(cycles, echemDF['cycleNo'].unique()[136-10:159-10:cycleSparsity[2]])
    
    
    cmap = cm.get_cmap('viridis', np.amax(echemDF['cycleNo'].unique()[:])-np.amin(echemDF['cycleNo'].unique()[:])+1)
    norm = Normalize(vmin=0,vmax=160)
    
    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=1.5)
    plt.tick_params(width=1.5)

    if illustrator == False:
        plt.xlabel('Potential (V)')
        plt.ylabel('dQ/dV /mAh g-1 V-1')
        plt.title('dQ/dV with increasing cut-off potentials')
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),label='Cycle Number')
        cbar.set_ticks([0,40,80,120,160])  
        
    else:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),format = ticker.FuncFormatter(lambda x, pos: ''))
        
        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])
        cbar.set_ticks([0,40,80,120,160])  
        # cbar.set_labels([])
    
    cbar.ax.tick_params(width=1.5)
    cbar.outline.set_linewidth(1.5)    

    for cycle in cycles:
        print(cycle)
        tf = echemDF[echemDF['cycleNo'] == cycle]
        currKeys = tf['Block'].unique()

        for i in range(0,len(currKeys)):
            cap = CalcCapacity(tf[tf['Block']==currKeys[i]]['Time (s)'][:].astype(float), tf[tf['Block']==currKeys[i]]['Current (mA)'][:], activeMass)
            
            
            x_volt, dqdv, smooth_cap = dqdvSingle(cap,tf[tf['Block']==currKeys[i]]['Potential (V)'][:], window_size_1 = 51, window_size_2=501)
                
            # if timeAxis == True:
            #     dt = np.mean(np.diff(tf['Time (s)'][:].astype('float'))) #should be 0.100 seconds
            #     tVec = np.linspace(0,)
            # else:
            #     pass
            
            if i == 0:
                plt.plot(x_volt, dqdv/1000, color=cmap.colors[int(cycle)])
            elif i == 1:
                plt.plot(x_volt, -dqdv/1000, color=cmap.colors[int(cycle)])
            else:
                pass

            
        else:
            ValueError
            
            
        
    #Shading upper cut-off voltage regions:
    plt.fill_between([3,4.2], -0.5, 0.6, alpha=0.1, color='tab:green') #4.2V
    # plt.fill_between([-0.5,0.5], 3, 4.2, alpha=0.1, color='tab:green') #4.2V
    plt.fill_between([4.2,4.4], -0.5, 0.6, alpha=0.1, color='tab:orange') #4.3V
    plt.fill_between([4.4,4.6], -0.5, 0.6, alpha=0.1, color='tab:red') #4.3V
    
    #Vertical blacklines for aesthetics
    plt.axvline(4.2,color='black',linestyle='--',lw=1.5)
    plt.axvline(4.4,color='black',linestyle='--',lw=1.5)
    #Axis lims
    plt.ylim([-0.45,0.55])
    plt.xlim([3,4.6])
    
    #Set ticks
    plt.xticks(np.arange(3,4.8,0.4))
    plt.yticks(np.arange(-0.5,0.75,0.25))
    
    plt.show() 
    plt.savefig(path+'Figure 2b.pdf',bbox_inches='tight')
    
#%%       
# Mean intensities as a function of cycle and cut-off voltage
from scipy.signal import savgol_filter
from scipy.interpolate import splrep, splev


def meanInt(echemDF, cycleSparsity = [10,2,2], normalise = False, smoothing = False, deriv = False, illustrator = False, path=None):
    
    # if deriv == True:
    #     smoothing = True
    # else:
    #     pass
    
    #Evaluate cycle no. lists once drastically speeds up performance:
    a = echemDF['cycleNo'].unique()[2:110:cycleSparsity[0]]
    a = np.append(a, echemDF['cycleNo'].unique()[123-10:135-10:cycleSparsity[1]])
    a = np.append(a, echemDF['cycleNo'].unique()[136-10:159-10:cycleSparsity[2]])
    
    fig,ax = plt.subplots(figsize=[6,4])

    cmap = cm.get_cmap('viridis', np.amax(echemDF['cycleNo'].unique()[:])-np.amin(echemDF['cycleNo'].unique()[:])+1)
    norm = Normalize(vmin=0,vmax=160)

    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=1.5)
    plt.tick_params(width=1.5)

    if illustrator == True:
        plt.gca().axes.set_xticklabels([])
        plt.gca().axes.set_yticklabels([])
    else:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),label='Cycle Number')
        cbar.set_ticks([0,40,80,120,160]) 
        
        cbar.ax.tick_params(width=1.5)
        cbar.outline.set_linewidth(1.5)    
        
        plt.xlabel('Time / hours')
        plt.title('Mean Intensity as a function of cycle as cut-off voltage increases')

        if normalise == False:
            plt.ylabel('Mean Intensity / arb. units')
        elif normalise == True:
            plt.ylabel('Norm. Mean Intensity /arb. units')
        else:
            pass
    
    
    for cycleNo in a:

        #Extract echem based on the cycle number:
        tf = echemDF[echemDF['cycleNo'] == cycleNo] 
        currKeys = tf['Block'].unique()
  
        with tables.File(echemDF[echemDF['cycleNo'] == cycleNo]['addrs'].iloc[0] ,'r') as f:
            print(cycleNo)
            
            camTime = f.root.camera_timing[1,:]
            intVec = np.empty(0)
            sRate = np.mean(np.diff(f.root.camera_timing[1,:]))
            
            for currKey in currKeys:
                
                tStartEchem = float(tf[tf['Block']==currKey]['Computer time (s)'].iloc[0])
                tEndEchem = float(tf[tf['Block']==currKey]['Computer time (s)'].iloc[-1])
                # echemTime = camTime[np.abs(camTime-tStartEchem).argmin():np.abs(camTime-tEndEchem).argmin()]
                
                intVec = np.append(intVec, f.root.average_intensity[np.abs(camTime-tStartEchem).argmin():np.abs(camTime-tEndEchem).argmin()])
                
                
            
            tVec = np.arange(0,intVec.shape[0]*sRate, sRate)/3600 #idk why this vector doesnt match the length of intVec

            if smoothing == True:
                intVec = savgol_filter(intVec, window_length=int(intVec.shape[0]/40), polyorder=1)
            else:
                pass


            
            if normalise == False:
                pass
            elif normalise == True:

                # intVec = intVec/np.amin(intVec)
                intVec = intVec/intVec[0]
            else:
                ValueError
                
            if deriv == True:
                
                spline = splrep(tVec[:intVec.shape[0]],intVec,k=1,s=0.0)
                intVec = splev(tVec[:intVec.shape[0]], spline, der=1)
                intVec = savgol_filter(intVec, window_length=int(intVec.shape[0]/20), polyorder= 1)
            else:
                pass
            
            
            plt.plot(tVec[:intVec.shape[0]],intVec,color=cmap.colors[int(cycleNo)])  


    plt.show()
    
    if illustrator == True:
        plt.savefig(path+'Figure 3a.pdf',bbox_inches='tight')
    else:
        pass
    
    
#%%
from scipy.signal import savgol_filter
from scipy.interpolate import splrep, splev


def intPhases(echemDF, cycleSparsity = [10,2,2], normalise = False, smoothing = False, deriv = False):
    
    
    #Evaluate cycle no. lists once drastically speeds up performance:
    a = echemDF['cycleNo'].unique()[2:110:cycleSparsity[0]]
    a = np.append(a, echemDF['cycleNo'].unique()[123-10:135-10:cycleSparsity[1]])
    a = np.append(a, echemDF['cycleNo'].unique()[136-10:159-10:cycleSparsity[2]])
    
    
    fig = plt.figure()
    
    plt.xlabel('Time / hours')
    if normalise == False:
        axs[0].set_ylabel('Mean Intensity / arb. units')
    elif normalise == True:
        axs[0].set_ylabel('Norm. Mean Intensity /arb. units')
    else:
        pass

    cmap = cm.get_cmap('viridis', np.amax(echemDF['cycleNo'].unique()[:])-np.amin(echemDF['cycleNo'].unique()[:])+1)
    norm = Normalize(vmin=0,vmax=160)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),label='Cycle Number')
    cbar.set_ticks([0,80,160]) 
    

    for cycleNo in a:

        #Extract echem based on the cycle number:
        tf = echemDF[echemDF['cycleNo'] == cycleNo] 
        currKeys = tf['Block'].unique()
  
        with tables.File(echemDF[echemDF['cycleNo'] == cycleNo]['addrs'].iloc[0] ,'r') as f:
            print(cycleNo)
            
            camTime = f.root.camera_timing[1,:]
            intVec = np.empty(0)
            sRate = np.mean(np.diff(f.root.camera_timing[1,:]))
            
            for currKey in currKeys:
                
                tStartEchem = float(tf[tf['Block']==currKey]['Computer time (s)'].iloc[0])
                tEndEchem = float(tf[tf['Block']==currKey]['Computer time (s)'].iloc[-1])
                # echemTime = camTime[np.abs(camTime-tStartEchem).argmin():np.abs(camTime-tEndEchem).argmin()]
                
                intVec = np.append(intVec, f.root.average_intensity[np.abs(camTime-tStartEchem).argmin():np.abs(camTime-tEndEchem).argmin()])
                
                
            
            tVec = np.arange(0,intVec.shape[0]*sRate, sRate)/3600 #idk why this vector doesnt match the length of intVec

            if smoothing == True:
                intVec = savgol_filter(intVec, window_length=int(intVec.shape[0]/40), polyorder=1)
            else:
                pass


            
            if normalise == False:
                pass
            elif normalise == True:

                intVec = intVec/np.amin(intVec)
            else:
                ValueError
                
            if deriv == True:
                
                spline = splrep(tVec[:intVec.shape[0]],intVec,k=1,s=0.0)
                intVec = splev(tVec[:intVec.shape[0]], spline, der=1)
                intVec = savgol_filter(intVec, window_length=int(intVec.shape[0]/20), polyorder= 1)
            else:
                pass
            
            
            plt.plot(tVec[:intVec.shape[0]],intVec,color=cmap.colors[int(cycleNo)])
            # if cycleNo < 110:
            #     axs[0].plot(tVec[:intVec.shape[0]],intVec,color=cmap.colors[int(cycleNo)])

                
            #     pot = np.linspace(3.5,4.5,100)
            #     pot =np.append(pot,np.flip(pot))
            #     axs[1].plot(np.arange(0,3.5,3.5/200),pot,'-r')
                
            # elif cycleNo < 125:
            #     axs[0].plot(tVec[:intVec.shape[0]],intVec,color=cmap.colors[int(cycleNo)])  

            # else:
            #     axs[0].plot(tVec[:intVec.shape[0]],intVec,color=cmap.colors[int(cycleNo)])  


            # ax2 = ax1.twiny()
            
            # pot = np.linspace(3.5,4.5,100)
            # pot =np.append(pot,np.flip(pot))
            # ax2.plot(pot,'-r')
            
            

    
    # ax2.set_xlabel('test x')
    # ax2.set_ylabel('test y')
    
    fig.suptitle('Mean Intensity as a function of cycle as cut-off voltage increases')
    fig.show()

    # intPhases(echemDF, cycleSparsity = [50,5,5], normalise=True, smoothing=True, deriv = True)


#%%
import numpy as np
import matplotlib.pyplot as plt

# Create some mock data
x2 = np.linspace(0,1,11)
y2 = np.random.rand(11)

x3 = np.linspace(1,0,101)
y3 = np.random.rand(101)*20+20


fig, axs = plt.subplots(1,3)
# ax2 = fig.add_subplot(111, label="second axes")
# ax2.set_facecolor("none")
# axs[1].set_xlabel('x1', color='red')
# axs[0].set_ylabel('y1', color='red')

for i in range(len(axs)):

    # axs[i].plot(x2, y2, color='red')
    # axs[i].tick_params(colors='red')
    ax2=axs[i].twiny()
    ax2.set_xlabel('x2', color='blue')
    # ax2.set_ylabel('y2', color='blue')
    ax2.plot(x2, y2, color='blue')
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top') 
    ax2.tick_params(colors='blue')
    
    
    ax2=axs[i].twinx()
    ax2.set_ylabel('y2', color='blue')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(colors='blue')


    ax3=axs[i].twiny()
    ax3.plot(x3,y3,color='red', zorder=10)

    ax3.xaxis.tick_bottom()
    ax3.tick_params(colors='red')
    
    ax3=axs[i].twinx()
    ax3.set_ylabel('y3', color='red')
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    # for which in ["top", "right"]:
    #     ax2.spines[which].set_color("blue")
    #     ax2.spines[which].set_visible(False)
    # for which in ["bottom", "left"]:
    #     ax3.spines[which].set_color("red")
    #     ax3.spines[which].set_visible(False)
    


plt.show()
#%%
'''
####
Commands to generate paper figures in cells below:
####
'''

#%% Figure 2 panels: 
import matplotlib as mpl

plt.rcParams['pdf.fonttype']=42
mpl.rcParams['figure.dpi'] = 108
mpl.rcParams['savefig.dpi'] = 108
path2=r'D:/NMC_degradation_3_160623_Halfthedata/Figures/Figure 2/'
'''
illustrator kwarg removes all text including axis labels
'''

# PotVsCap(activeMass = 4.77, cycleSparsity = [10,1,1],illustrator=True, path=path2)
# dqdvMulti(echemDF, cycleSparsity=[10,1,1],illustrator=True, path=path2)
# CapCoul(activeMass = 4.77, firstCycles=False, illustrator=True, path=path2)

#%% Figure 3 - time version panels:
    
path3=r'D:/NMC_degradation_3_160623_Halfthedata/Figures/Figure 3/'

# meanInt(echemDF, cycleSparsity = [10,1,1], normalise=False, smoothing=False, deriv = False,illustrator=True,path=path3)
# IntVsCap(normInt = True, activeMass = 4.77, cycleSparsity=[10,1,1], window_length = 50, xAxis = 'pot', grad=False, illustrator=True,path=path3)
# meanIntScatter(realTime=False, sparsity=2, norm=False, illustrator=True, path=path3) #sparsity = 0 or 1 will plot all data    

#%% Figure 3 - capacity version panels:
path3=r'D:/NMC_degradation_3_160623_Halfthedata/Figures/Figure 3/'
plt.rcParams['pdf.fonttype']=42
mpl.rcParams['figure.dpi'] = 108
mpl.rcParams['savefig.dpi'] = 108
# meanInt(echemDF, cycleSparsity = [10,2,2], normalise=False, smoothing=False, deriv = False) 
IntVsCap(normInt = True, activeMass = 4.77, cycleSparsity=[10,1,1], window_length = 50, xAxis = 'cap', grad=False, illustrator=True, path=path3)
# meanIntScatter(realTime=False, sparsity=2, norm=False) #sparsity = 0 or 1 will plot all data    


#%% Mean intensity with smoothing, normalisation and derivative options
meanInt(echemDF, cycleSparsity = [50,2,2], normalise=True, smoothing=False, deriv = False)

#%% Differential plotting - dIntensity/dV

IntVsCap(normInt = False, activeMass = 4.77, cycleSparsity=[10,1,1], window_length = 100, xAxis = 'pot', grad=True)

'''
y-axis limits are messed up
'''

#%% Intensity vs. Potential

IntVsCap(normInt = False, activeMass = 4.77, cycleSparsity=[10,1,1], window_length = 100, xAxis = 'pot')

#%% Capacity and Coulombic efficiency

coulEff = CapCoul(activeMass = 4.77, firstCycles=False)


#%% Potential vs. capacity

PotVsCap(activeMass = 4.77, cycleSparsity = [5,1,1])

#%% Intensity vs. capacity
    
IntVsCap(normInt = False, activeMass = 4.77, cycleSparsity=[10,2,2])
    
#%% dq/dv

dqdvMulti(echemDF, cycleSparsity=[5,1,1])

#%% Generate list of cycles to process using object detection:
    
cycleSparsity = [2,1,1]

cycleOffset = 0 #original value = 0

objDetCycles = echemDF['cycleNo'].unique()[2+cycleOffset:110:cycleSparsity[0]]
objDetCycles = np.append(objDetCycles, echemDF['cycleNo'].unique()[123-10+cycleOffset:135-10:cycleSparsity[1]])
objDetCycles = np.append(objDetCycles, echemDF['cycleNo'].unique()[136-10+cycleOffset:159-10:cycleSparsity[2]])

objDetAddrs = ['']*len(objDetCycles)

cycleFrames = np.zeros([len(objDetCycles),2])


for i in range(0,len(objDetCycles)):
    objDetAddrs[i] = echemDF[echemDF['cycleNo'] == objDetCycles[i]]['addrs'].iloc[0]
    with tables.File(objDetAddrs[i],'r') as f:
        print('cycle: ',objDetCycles[i])
        print('no. frames: ', f.root.average_intensity.shape[0])
        cycleFrames[i,0] = objDetCycles[i]
        cycleFrames[i,1] = f.root.average_intensity.shape[0]
    
#Generate subset of echemDF only containing the cycles that are going to be exported for further processing
partialEchemDF = echemDF.mask(~echemDF['cycleNo'].isin(objDetCycles)).dropna()
#%%
# Save partial echemDF to .csv
# partialEchemDF.to_csv('D://NMC_degradation_3_160623_Halfthedata/echemDF_partial_2_1_1.csv')
df = pd.DataFrame(cycleFrames)
df.to_csv('D://NMC_degradation_3_160623_Halfthedata/cycleFrames.csv', header=False, index=False)
#%%
# open file in write mode
with open(r'D://NMC_degradation_3_160623_Halfthedata/objDetAddrs_2_1_1.txt', 'w') as fp:
    for item in objDetAddrs:
        # write each item on a new line
        fp.write("%s\n" % item)
        
    print('Done')
    
#%% Count frames per dataset



for cycle in echemDF['cycleNo'].unique():
    with tables.File(echemDF[echemDF['cycleNo']==cycle]['addrs'].unique()[0],'r') as f:
        print('no. frames: ', f.root.average_intensity.shape[0])











