# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 10:56:05 2022

@author: ar2071
"""

#Plotting library

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import DataManipulation as dm
import napari
import tables
import matplotlib.patheffects as pe
import config as cfg
from cycler import cycler

def sizeHistogram(particlesSizes, saveFig = False):
    
    #Setting default figure formatting for papers etc.
    rcParams['font.family'] = 'arial'
    rcParams['font.size'] = '8'
    rcParams['figure.figsize'] = (3.15, 1.94) 
    rcParams['lines.linewidth'] = 1.2
    
    #Histogram of particle sizes
    pixelSizeum = 1e-6*(cfg.pixelSizenm)**2 #converts pixel area to um^2 area
    bins = np.linspace(0,3000*pixelSizeum*0.25,50)
    

    
    if saveFig == True:
        
        plt.figure()
        plt.hist(particlesSizes[:]*pixelSizeum, bins)
        plt.xlabel(r'Particle Area /$\mu m^2$')
        plt.ylabel('No. particles')
        plt.title('Histogram of active particle sizes')
        fig = plt.gcf()
        fig.savefig(cfg.FileAddressOut[0]+'/Plots/'+'/sizeHistogram'+'.svg')
        fig.savefig(cfg.FileAddressOut[0]+'/Plots'+'/sizeHistogram' + '.png', dpi = 600, facecolor = 'w', edgecolor = 'w', transparent = False, bbox_inches = 'tight', pad_inches = 0.5)

        plt.show()
   
    else:
        
        plt.figure()
        plt.hist(particlesSizes[:]*pixelSizeum, bins)
        plt.xlabel(r'Particle Area /$\mu m^2$')
        plt.ylabel('No. particles')
        plt.title('Histogram of active particle sizes')
        plt.show()
        


def plotTrace(particleTraces, traceList, mean = False, saveFig = False, bokkeh = False, particleIndex = None, nStdDev = 1):
    
    
    #Setting default figure formatting for papers etc.
    rcParams['font.family'] = 'arial'
    rcParams['font.size'] = '8'
    rcParams['figure.figsize'] = (3.15*1.5, 1.94*1.5) 
    rcParams['lines.linewidth'] = 1.2
    

    
    #traceList is a list of strings corresponding to names of columns in the particleTraces dataframe
    for i in range(len(traceList)):
        
        #unflatten traces
        temp = dm.unflatten(particleTraces, traceList[i])


        
        
        if particleIndex == None:
            trace = temp
        else:
            nanarray = np.zeros(temp.shape)*np.nan
            nanarray[particleIndex] = 1
            trace = temp*nanarray
            
            #set colour to light blue if only a single particle is plotted
            #rcParams['axes.prop_cycle'] =  cycler(color = ['#007FFF']) #blue
            #rcParams['axes.prop_cycle'] = cycler(color = ['#ff0000']) #red
            
        #Generate time axis
        # timeVec = np.linspace(0,(len(trace)-1)*cfg.frameRate, len(trace))
        # print('time', len(timeVec))
        # print('other', len(trace))
        
        
        if saveFig == True:
            
            mpl_fig = plt.figure()
            
            if mean == True:

                for j in range(len(trace)):
                    
                    timeVec = np.linspace(0,(len(trace[j])-1)*cfg.frameRate, len(trace[j])) #generate real time axis
                    plt.plot(timeVec,trace[j], alpha = 0.3)
                    print(j)
                mTrace, stdvec = dm.meanTrace(particleTraces, traceList[i])
                plt.plot(timeVec,mTrace, color='k', lw=2.5)
                plt.plot(timeVec,mTrace+nStdDev*stdvec, '--k', lw=1.5)
                plt.plot(timeVec,mTrace-nStdDev*stdvec, '--k', lw=1.5)
                
                plt.axvline(np.argmax(stdvec), color = 'k', linestyle = '--')
                '''^^^REPLACE WITH TIME POINT AT WHICH THE CURRENT POLARITY ACTUALLY SWITCHES'''
                
            else:    
                for j in range(len(trace)):
                    timeVec = np.linspace(0,(len(trace[j])-1)*cfg.frameRate, len(trace[j])) #generate real time axis
                    plt.plot(timeVec,trace[j])
                    
            
            plt.plot()
            plt.xlabel('Time / seconds')
            plt.ylabel('Normalised Relative Intensity / arb. units')
            #plt.title(traceList[i])
            fig = plt.gcf()
            fig.savefig(cfg.FileAddressOut[0]+'/Plots/'+ traceList[i] + '.svg')
            fig.savefig(cfg.FileAddressOut[0]+'/Plots/'+traceList[i] + '.png', dpi = 600, facecolor = 'w', edgecolor = 'w', transparent = False, bbox_inches = 'tight')
            plt.show()
        
        else:
            
            
            mpl_fig = plt.figure()
            if mean == True:
                for j in range(len(trace)):
                    
                    timeVec = np.linspace(0,(len(trace[j])-1)*cfg.frameRate, len(trace[j])) #generate real time axis
                    plt.plot(trace[j], alpha = 0.3)
                mTrace, stdvec = dm.meanTrace(particleTraces, traceList[i])
                
                plt.plot(mTrace, color='k', lw=2.5)
                plt.plot(mTrace+nStdDev*stdvec, '--k', lw=1.5)
                plt.plot(mTrace-nStdDev*stdvec, '--k', lw=1.5)
                
                plt.axvline(np.argmax(stdvec), color = 'k', linestyle = '--') 
                '''^^^REPLACE WITH TIME POINT AT WHICH THE CURRENT POLARITY ACTUALLY SWITCHES'''
                
            else:
                for j in range(len(trace)):
                    timeVec = np.linspace(0,(len(trace[j])-1)*cfg.frameRate, len(trace[j])) #generate real time axis
                    plt.plot(trace[j])

            
            plt.xlabel('Time / seconds')
            plt.ylabel('Counts')
            #plt.title(traceList[i])
            plt.show()
            return mpl_fig


def stackViewer(filename, maxObjects, particleIndex = None):
    
    #Napari-based stack scrubbing
    
    #adjust image frame to show a single particle
    if particleIndex != None:
        
        #arbitrary multiplier to pad the frame
        xmin = maxObjects['xmin'][particleIndex]*0.9
        xmax = maxObjects['xmax'][particleIndex]*1.1
        
        ymin = maxObjects['ymin'][particleIndex]*0.9
        ymax = maxObjects['ymax'][particleIndex]*1.1
    
    if particleIndex == None:
        with tables.File(cfg.FileAddressOut[0]+"/"+filename, 'r') as h5f:
            
            particleFrames = h5f.root.data[:,:,:]
            size = h5f.root.data.shape

    else:
        with tables.File(cfg.FileAddressOut[0]+"/"+filename, 'r') as h5f:
            
            particleFrames = h5f.root.data[:, ymin:ymax, xmin:xmax]
            size = h5f.root.data[0, ymin:ymax, xmin:xmax].shape         
        
    #print(size)
    
    #points = np.array([[100, 100], [150, 140], [50, 100]])
    viewer = napari.view_image(particleFrames)
    #viewer.add_points(points)

    napari.run()

    