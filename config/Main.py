# -*- coding: utf-8 -*-
"""
Title: Main.py
Description: Main script to run
Author: Aleksandar Radic - ar2071
Date: 18/11/21
"""

#%%
#RUN ONCE TO ADD CURRENT FOLDER TO PATH IF HEADER FILES ARE NOT FOUND
import sys
sys.path.append("C://Users/alekr/OneDrive - University of Cambridge/PhD/Python/iSCAT Data Analysis/Unified Current")
sys.path.append("C://Users/ar2071/OneDrive - University of Cambridge/PhD/Python/iSCAT Data Analysis/Unified Current")

#Module Imports
import config as cfg
import DataManipulation as dm
import EChem as echem
import DataIO as dio
import multiprocessing_lib as multi 
import Plotting as p


import tables
import numpy as np

#%%Data input and path setup:
dio.dataPathSetup()
#%%
'''
##############
#CHANGE THE FILESIZE*2 WITHIN readFileDims FUNCTION IF/WHEN ZARR IS IMPLEMENTED TO SHRINK STABILISED DATA TO FLOAT16
###############
'''

multi.mainfunc()

#%%
#Generate differential and background subtracted image stacks
dm.diffStack(nFrames = 0, zeroing = True)

#%%
dm.bkgSubStack()
#dm.noBkgStack(FileAddressOut, segMapStack)

#%%
# while cfg.close_flag == 0:
#     sleep(0.5)
# else:    
dio.echemPathSetup()
#%%
#EChem Processing - outputs generic EChem plots
echem.EChemProcessing() #returns cfg.time_volt_curr_directory variable
#eChemParams = {"active mass": cfg.ActiveMass, "lithiums per tm": cfg.LiX, "molar mass": cfg.MolarMass, "cathode or anode": cfg.CatOrAn}

#%% Load outputs from previously processed datasets:
'''REPLACE cfg.FileAddressOut with cfg.procFileAdd everywhere - obtained from new import function'''
particleTraces = dio.loadDataFrame('particleTraces.csv')
maxObjects = dio.loadDataFrame('maxObjects.csv')
segMapStack = np.load(cfg.FileAddressOut[0]+'/segMapStack.npy') #index needed because it's a list object
maxSegMap = np.load(cfg.FileAddressOut[0]+'/maxSegMap.npy')
trajectory = np.loadtxt(cfg.FileAddressOut[0]+'/trajectoryArrayCoarse.npy')
# timesIntensity = np.load(cfg.FileAddressOut[0]+'/timesIntensity.npy')
# frameRate = np.load(cfg.FileAddressOut[0]+'/frameRate.npy')



'''PLOTTING SECTION BELOW'''
#%%
dm.noBkgStack(segMapStack, fileIn = '/imageStackStab.h5')
#%% Particle Size Histogram
#TO DO: Make the binning adjust automatically
p.sizeHistogram(particleTraces['SizesPix'], saveFig = True) #go to function definition to adjust binning


#%% Trace plotting
p.plotTrace(particleTraces, ['IntensitiesNormFilt'], mean = True, saveFig = True, particleIndex = None, nStdDev = 2)

#%% Scrub through an image stack
p.stackViewer('ImageStackStab.h5', maxObjects)
#%%
dm.noBkgStack(segMapStack, fileIn = '/imageStackStab.h5')

#%%
#nFrames = 0 is diff stack where first frame is the divisor
#nFrames of any positive integer divides by the current frame minus nFrames
#zeroing == True takes 1 away from diff stack so a colourmap may be centred on zero

dm.diffStack(fileIn = '/ImageStackStab.h5', nFrames = 0, zeroing = True)
#%% Export a trace from particleTraces structure

#particleIndex is an optional argument, if left out all particles will be exported for the chosen trace type
#will output a file with the name of the trace type followed by the index of the particle as a .csv
dio.exportTrace(particleTraces, ['IntensitiesNormFilt'], particleIndex = 1, frameRate=None, save = True)

#%% Load entire .h5 file into memory - currently only reads entire stack

stack, size = dio.readH5(fileIn = '/ImageStackStab.h5')
#%% Save 

particleIndex = 2
xmin = maxObjects['xmin'][particleIndex]*0.9
xmax = maxObjects['xmax'][particleIndex]*1.1

ymin = maxObjects['ymin'][particleIndex]*0.9
ymax = maxObjects['ymax'][particleIndex]*1.1
    
with tables.File(cfg.FileAddressOut+'/ImageStackStab.h5', 'r') as h5f:
        
    #particleFrames = h5f.root.data[:, ymin:ymax, xmin:xmax]
    particleFrames = h5f.root.data[:, :, :]
    #size = h5f.root.data[0, ymin:ymax, xmin:xmax].shape   

np.save(cfg.FileAddressOut+'/ImageStackStab1.npy', particleFrames)
#%%
np.save('ImageStackStab.npy', stack)

#%%Average intensity over entire frame
from tqdm import trange

frame_sum = np.zeros(stack.shape[0]-3)

for i_ in trange(0,stack.shape[0]-3):
    
    frame_sum[i_] = np.sum(stack[i_])/(stack.shape[1]*stack.shape[2])
    #frame_sum[i_] = frame_sum[i_]/frame_sum[0]
    
#%%
trace = dm.intensityWholeFrame('/ImageStackStab.h5')
#%%
import matplotlib.pyplot as plt
from matplotlib import rcParams

#Setting default figure formatting for papers etc.
rcParams['font.family'] = 'arial'
rcParams['font.size'] = '8'
rcParams['figure.figsize'] = (3.15*2, 1.5*1.94) 
rcParams['lines.linewidth'] = 1.2

plt.figure()

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)


plt.plot(frame_sum,'k-')



plt.savefig(cfg.FileAddressOut[0]+'/Plots/'+'phase_relaxation' + '.png', dpi = 600, facecolor = 'w', edgecolor = 'w', transparent = False, bbox_inches = None, pad_inches = 0.1)
plt.savefig(cfg.FileAddressOut[0]+'/Plots/'+ 'phase_relaxation' + '.svg')

#plt.show()

