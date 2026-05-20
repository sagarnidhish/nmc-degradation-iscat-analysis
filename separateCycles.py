# -*- coding: utf-8 -*-
"""
Splitting the degradation data into single cycles using echem triggers:

Created on Tue Oct 17 10:46:07 2023

@author: Alek
"""


import numpy as np
import tables
import h5py
import pandas as pd

import glob
import os
import sys
sys.path.append("C://Users/alek/Desktop/guilEchem")

from analysis_library import Experiment

expDir = "D://NMC_degradation_3_160623_Halfthedata/"

os.chdir(expDir)
expNames = glob.glob(expDir+'*.hdf5')

#%%

# for i in range(0,len(expNames)):
i = 11
    
'''
Problem files:
    3_c2 - appears to be corrupt, external HDF5 viewer also returns error
    3_c2 has unique error - OSError: Unable to open file (addr overflow, addr = 26678648, size = 104, eoa = 2048)
'''


print('\n\nEXP. NAME: ', expNames[i][41:-5])
exp = Experiment(expNames[i])
file = exp.file

#Create output directory for the individual cycles
outputDir = expDir+'/'+expNames[i][41:-5]+'chopped'
# os.mkdir(outputDir, exist_ok=True)

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#Difference between camera and potentiostat timings:
echemBlocks = exp.get_blocks()
nCycles = int(len(echemBlocks)/2)

print('\nFilename: ', expNames[i][41:])
print('\nNo. cycles in file: ',nCycles)

potTiming = echemBlocks[-1][-1] #Final time point recored by the potentiostat
camTiming = file['camera_timing'][1][-1] #Final time point recorded by the camera

spf = np.mean(np.diff(file['camera_timing'][1][:])) #Frame rate (in seconds per frame - spf) derived from camera timing
fps = 1/spf #Frame rate in frames per second - fps

timingDiffSec = abs(potTiming-camTiming) #Difference in timings in seconds

print('\nSeconds difference between potTiming and camTiming: ', timingDiffSec)
print('\nSeconds per frame: ', spf)
print('\nNo. frames difference between the timings: ', timingDiffSec/spf)

nPreFrames = 5 #Number of pre-frames recorded - always set to 5
nPostFrames = np.ceil(timingDiffSec/spf - nPreFrames).astype(int) #Calculate post-frames based on pre-frames and total timing difference

print('\nNo. pre/post-frames: {}, {}'.format(nPreFrames,nPostFrames))

#Removal of said frames from movie and splitting into individual cycles:
echemStr = 'galvanostatic_{}'

with tables.File(expNames[i],'r') as fIn:
    
    #Import all echem as dataframe:
    echemDF = pd.read_hdf(expNames[i])
    
    #Identify frame numbers for start and end of each cycle
    camera_timingIn = fIn.root.camera_timing[:,:-(nPreFrames+nPostFrames)] #Only import the corrected time vector
    average_intensityIn = fIn.root.average_intensity[0,:]
    for j in range(0,nCycles):
        
        with tables.File(outputDir+'/'+expNames[i][41:-5] + '_cycle'+str(j)+'.hdf5','w') as fOut:

            if j == 0:
                cycleInd = 1
                start = nPreFrames
                end = (np.abs(camera_timingIn[1,:] - echemBlocks[cycleInd][2])).argmin() + nPreFrames
                echemOut = echemDF[(echemDF['Block']==(echemStr.format(cycleInd))) | (echemDF['Block'] == (echemStr.format(cycleInd-1)))]
                
            elif j == nCycles:
                start = end
                end = fIn.root.movie.shape[0]-nPostFrames
                echemOut = echemDF[(echemDF['Block']==(echemStr.format(cycleInd))) | (echemDF['Block'] == (echemStr.format(cycleInd-1)))]

            else:
                cycleInd = cycleInd + 2
                start = end
                end = (np.abs(camera_timingIn[1,:] - echemBlocks[cycleInd][2])).argmin()
                echemOut = echemDF[(echemDF['Block']==(echemStr.format(cycleInd))) | (echemDF['Block'] == (echemStr.format(cycleInd-1)))]
            

            print('\nstart: ', start)
            print('end: ', end)
            print('len: ', end-start)

            #Export camera timing, movie, average intensity and echem
            fOut.create_array(fOut.root, 'camera_timing', camera_timingIn[:,start:end])
            fOut.create_array(fOut.root, 'movie', fIn.root.movie[start:end,:,:])
            fOut.create_array(fOut.root, 'average_intensity', average_intensityIn[start:end])
            echemOut.to_hdf(outputDir+'/'+expNames[i][41:-5] + '_cycle'+str(j)+'.hdf5',key='echem' ,mode = 'a')
            echemOut.to_csv(outputDir+'/'+expNames[i][41:-5] + '_cycle'+str(j)+'.csv')
            

    
            
#%% Test cell
# exp.summarize()
# meanInt = exp.average_intensity()

# with tables.File('D://NMC_degradation_3_160623_Halfthedata/2_c2_x14_200623.hdf5','r') as tf:
with tables.File('D://NMC_degradation_3_160623_Halfthedata/4_c2_x10_240623chopped/4_c2_x10_240623_cycle1.hdf5','r') as tf:
    # camtime1 = tf.root.camera_timing[:]
    # print(round(1/(camtime[1,1]-camtime[1,0]),3))
    # meanInt = tf.root.movie[0,:,:]
    meanInt1 = tf.root.average_intensity[:]
    
                
with tables.File('D://NMC_degradation_3_160623_Halfthedata/4_c2_x10_240623chopped/4_c2_x10_240623_cycle0.hdf5','r') as tf:
    # camtime0 = tf.root.camera_timing[:]
    # print(round(1/(camtime[1,1]-camtime[1,0]),3))
    # meanInt = tf.root.movie[0,:,:]
    meanInt0 = tf.root.average_intensity[:]
                       
                
                