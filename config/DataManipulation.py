# -*- coding: utf-8 -*-
"""
Title: DataManipulation.py
Description: Header file containing functions for conversion of .tdms to .npy files, linear frame interpolation and GUI area selection on an image
Author: Aleksandar Radic - ar2071
Date: 18/11/21
"""

import numpy as np
from tqdm import trange
import time
from vidstab import VidStab
from scipy.signal import savgol_filter
import icecream
import pandas as pd
import tables
import os
import psutil
from math import ceil
icecream.ic.disable()
import config as cfg
#Video stabilisation#
import DataIO as dio
from pymlfunc import normxcorr2

import matplotlib.pyplot as plt
import photutils.centroids as cen
import DataManipulation as dm



def fineStabilisation(fileIn, stabType = 'global', method = 'com'):
    
    #Load h5 (stabilised file)
    stack,_ = dio.readH5(fileIn, border = 2, minusFrames = 3)
    
    frames = stack
    
    #Fourier filter
    #Fourier Filtering of original image
    fftframes = np.fft.fftshift(frames)
    
    hist, bins = np.histogram(fftframes)
    histind = np.unravel_index(hist.argmax(), hist.shape) #finds the index of the histogram data with highest no. occurrances
    
    fftframes[fftframes < bins[histind[0]+1]] = 0
    frames = np.fft.ifftshift(fftframes).real
    
    print('fourier filter cutoff freq = ', bins[histind[0]+1])
    
    
    nxc = np.zeros([frames.shape[0], frames.shape[1]*2-1,frames.shape[1]*2-1])
               
        
    if stabType == 'global':        
        for i in trange(0,frames.shape[0]-1):
            nxc[i] = normxcorr2(frames[0],frames[i+1])
    elif stabType == 'seq': #sequential residual calculation isnt set up yet
        for i in trange(0,frames.shape[0]-1):
            nxc[i] = normxcorr2(frames[i],frames[i+1])
    else:
        ValueError
        
    #Find centre of mass
    xtemp = np.zeros([nxc.shape[0]-1])
    ytemp = np.zeros([nxc.shape[0]-1])
    timevec = np.zeros([frames.shape[0]-1])
    
    #Initial guess for box centre:
    '''
    CHANGE TO EITHER USER INPUT (CENTRE OF SELECTED STABILISATION BOX) OR ALTERNATIVE
    '''
    
    xp, yp = np.unravel_index(nxc[0].argmax(), nxc[0].shape)
    #nxc[:,:,:] = nxc[:,:,:]-3e4
        
    #Centre finding algorithm selection:
    if method == 'com':
        
        func = cen.centroid_com
        
    elif method == '2dg':
        
        func = cen.centroid_2dg
        
    else:
        ValueError
    
    
    for i in trange(0,frames.shape[0]-1):
        
        xtemp[i],ytemp[i] = cen.centroid_sources(nxc[i], xp,yp, box_size = [21,21], centroid_func=func)
        xp = xtemp[i]
        yp = ytemp[i]
        
        
        timevec[i] = i
        
        
    #Position of CoM in original image coordinates:
    xpos = xtemp - int(nxc.shape[1]/4)
    ypos = ytemp - int(nxc.shape[2]/4)
    trajectoryArray = np.array([ypos, xpos]).T
    
    #linear interpolation
    framesStab = dm.interpolateStack(stack, trajectoryArray)
    
    
    return framesStab, xpos, ypos, timevec, nxc
    





def StabiliseStack(ind, ImageStack):
    
    #Load sub-image coords and size
    stabilizer = VidStab()
    
    sub_img_coord = [cfg.sub_img_coordx[ind], cfg.sub_img_coordy[ind]]
    sub_img_size = [cfg.sub_img_sizex[ind], cfg.sub_img_sizey[ind]]
    
    ImageStackTranspose = np.array((ImageStack[:, sub_img_coord[1]:sub_img_coord[1]+sub_img_size[1], sub_img_coord[0]:sub_img_coord[0]+sub_img_size[0]].T/np.amax(ImageStack[:, sub_img_coord[1]:sub_img_coord[1]+sub_img_size[1], sub_img_coord[0]:sub_img_coord[0]+sub_img_size[0]], axis=(0,1,2)))*255, dtype ='uint8')
    
    
    
    print("Video stabilisation progress:")
    time.sleep(0.2)
    for i_ in trange(0,ImageStack.shape[0]-2):
    
         frame = ImageStackTranspose[:,:,i_:i_+3]
         #Additions args for stab.stab_frame: smoothing_window = 20, border_type = 'black', border_size = 0, layer_func
         #Package uses Lucas-Kanade optical flow algorithm
         stabilizer.stabilize_frame(input_frame=frame, smoothing_window=10) 
    
    #stabilizer.plot_trajectory()

    #stabilizer.plot_transforms()

    #Retrieve stabilisation trajectory
    trajectory_array = stabilizer.trajectory #output contains the dx,dy per frame - not cumulative
    
    return trajectory_array


#Function to 2D interpolate a frame by shifting it a specified  (non-integer) number of pixels in x and y

def interpolateFrame(FrameOld, shiftx, shifty):

    xs_old = np.arange(0,len(FrameOld[0,:]))
    ys_old = np.arange(0,len(FrameOld[:,0]))
        
    x_new = xs_old + shiftx
    y_new = ys_old + shifty
    
    edgevalue=0
        
    FrameNew = np.zeros(np.shape(FrameOld))
    for row in np.arange(len(ys_old)):
        newrow = np.interp(x_new,xs_old,FrameOld[row,:],left=edgevalue,right=edgevalue)
        #newrow = np.interp(x_new,xs_old,FrameOld[row,:])
        FrameNew[row,:] = newrow
        
    for col in np.arange(len(xs_old)):
        newcol = np.interp(y_new,ys_old,FrameNew[:,col],left=edgevalue,right=edgevalue)
        #newcol = np.interp(y_new,ys_old,FrameNew[:,col])

        FrameNew[:,col] = newcol
        
    return FrameNew


def interpolateStack(ImageStack, trajectory_array):
    
    print("Interpolation progress:")
    time.sleep(0.2) #required wait
    
    #ImageStack_stab = np.float16(np.zeros(ImageStack.shape))
    ImageStack_stab = np.zeros(ImageStack.shape)
    for i_ in trange(0,trajectory_array.shape[0]):
        #ImageStack_stab[i_,:,:] = np.float16(interpolateFrame(ImageStack[i_,:,:], trajectory_array[i_,1], trajectory_array[i_,0]))
        ImageStack_stab[i_,:,:] = interpolateFrame(ImageStack[i_,:,:], trajectory_array[i_,1], trajectory_array[i_,0])
        
    return ImageStack_stab



#Function normalises a dataset such that the first element is equal to 1
def Normalise(dataArray):
    
    #Replacing all zeros with NaN to avoid division by zero errors
    #dataArray[dataArray == 0] = np.nan
    
    dataArrayNorm = np.zeros(dataArray.shape)
    
    for i_ in range(0,len(dataArray)):
        dataArrayNorm[i_,:] = dataArray[i_,:]/dataArray[i_,0]


    return dataArrayNorm

#Function standardises a dataset between 0 and 1
def Standardise(dataArray):
    
    #Replacing all zeros with NaN to avoid division by zero errors
    dataArray[dataArray == 0] = np.nan
    
    dataArrayStand = np.zeros(dataArray.shape)
    
    for i_ in range(0,len(dataArray)):
        
        dataArrayStand[i_,:] = dataArray[i_,:]-dataArray[i_,0]
        dataArrayStand[i_,:] = dataArray[i_,:]/np.nanmax(dataArray[i_,:])


    return dataArrayStand

def normAndStand(array, shape = (-1,-1), flatten = True):
    
    if isinstance(array, pd.Series) == True:
        array = np.array(array)
        
    else:
        pass
    
    if array.ndim == 1:
        array = array.reshape(shape)
    else:
        pass
    
    
    arrayNorm = Normalise(array)
    arrayStand = Standardise(array)
    
    if flatten == True:
        arrayNorm = arrayNorm.flatten()
        arrayStand = arrayStand.flatten()
    else:
        pass
    
    return arrayNorm, arrayStand

#icecream.ic.enable()
def nanParticleFilter(particleArray):
    
    particleArray = np.delete(particleArray,-1, 1)
    
    temparray = np.zeros(particleArray.shape)
    counter = 0
    
    for i_ in range(0,particleArray.shape[0]):
        
        numNans = np.isnan(particleArray[i_,:]).sum() #count total nans in a row
        
        
        if numNans < temparray.shape[1]*cfg.nanThreshold:
            
            #count number of valid particles
            temparray[counter,:] = particleArray[i_, :]
            counter +=1
            
        else:

            #If the particle has more than 1% NaN values, discard the particle
            pass
    
    particleArrayCleaned = np.zeros([counter, temparray.shape[1]])
    
    for i_ in range(0, counter):
         particleArrayCleaned[i_,:] = temparray[i_,:]
    
    return particleArrayCleaned

def nanInterp(array):
    import numpy as np
    #Function interpolates HORIZONTALLY - vectors in the direction of the x axis
    for i_ in range(0,array.shape[0]):
        
        ok = ~np.isnan(array[i_,:])
        xp = ok.ravel().nonzero()[0]
        fp = array[i_,~np.isnan(array[i_,:])]
        x  = np.isnan(array[i_,:]).ravel().nonzero()[0]
        

        array[i_,np.isnan(array[i_,:])] = np.interp(x, xp, fp)

            
    return array



def SavitskyGolayArrayFilter(array, shape = (-1,-1), flatten = True):
    

    if isinstance(array, pd.Series) == True:
        array = np.array(array)
        
    else:
        pass

    if array.ndim == 1:
        array = array.reshape(shape)
    else:
        pass
    
    
    #Iteration in the y-axis (per particle)
    for i_ in range(0,array.shape[0]):
        array[i_,:] = savgol_filter(array[i_,:], window_length = 9, polyorder = 2)


    if flatten == True:
        array = array.flatten()  
    else:
        pass
    
    return array


def dataSegmentation(fileIn, firstFrame = None):
    
    #Determine segmentation of data based on available system memory:
    fileSize = os.path.getsize(cfg.FileAddressOut[0]+fileIn)
    svmem = psutil.virtual_memory()

    if fileSize < svmem.available*0.75: #allow for a quarter of available memory to remain free
        nSegments = 1

    else:
        nSegments = ceil(fileSize/(svmem.available*0.75))
    
    #read in dimensions of image stack
    with tables.File(cfg.FileAddressOut[0]+fileIn, 'r') as fIn:
        
        if firstFrame == True:
            firstFrame = fIn.root.data[0,:,:]
            dims = fIn.root.data.shape
        else:
            dims = fIn.root.data.shape
    
    nFramesSegment = int(dims[0]/nSegments)

    return nFramesSegment, nSegments, dims, firstFrame


def diffStack(fileIn = '/ImageStackStab.h5', nFrames = 0, zeroing = False):

    nFramesSegment, nSegments, dims, firstFrame = dataSegmentation(fileIn, firstFrame = True)

    with tables.File(cfg.FileAddressOut[0]+fileIn, 'r') as fIn, tables.File(cfg.FileAddressOut[0]+'/diffStack.h5', mode = 'w') as fOut:
        atom = tables.Float32Atom()#defines the datatype as float32 to save space in stabilised image stack
        tempArray = fOut.create_earray(fOut.root, 'data', atom, (0, dims[1], dims[2])) #creates extendable array (earray) containing datatype float64



        print('\nGenerating differential image stack:\n')
        if nSegments == 1:
            for i_ in range(0, nSegments):
                stack = fIn.root.data[i_*nFramesSegment:nFramesSegment+i_*nFramesSegment,:,:]
                
                if nFrames == 0:
                    for j_ in trange(nFrames, stack.shape[0]):
                        
                        if zeroing == True: #centres the contrast on zero for diverging colourmaps
                            stack[j_,:,:] = (stack[j_,:,:]/firstFrame)-1
                            
                        else:
                            stack[j_,:,:] = stack[j_,:,:]/firstFrame
                
                    tempArray.append(stack)
                    
                else:
                    for j_ in trange(nFrames, stack.shape[0]):
                        if zeroing == True:
                            stack[j_,:,:] = (stack[j_,:,:]/stack[j_-nFrames,:,:])-1
                        else:
                            stack[j_,:,:] = stack[j_,:,:]/stack[j_-nFrames,:,:]
                    
                    tempArray.append(stack[nFrames:,:,:])
    
        else:
            
            for i_ in trange(0, nSegments):
                stack = fIn.root.data[i_*nFramesSegment:nFramesSegment+i_*nFramesSegment,:,:]
                
                if nFrames == 0:
                    for j_ in trange(nFrames, stack.shape[0]):
                        
                        if zeroing == True: #centres the contrast on zero for diverging colourmaps
                            stack[j_,:,:] = (stack[j_,:,:]/firstFrame)-1
                            
                        else:
                            stack[j_,:,:] = stack[j_,:,:]/firstFrame
                
                    tempArray.append(stack)
                    
                else:
                    for j_ in trange(nFrames, stack.shape[0]):
                        if zeroing == True:
                            stack[j_,:,:] = (stack[j_,:,:]/stack[j_-nFrames,:,:])-1
                        else:
                            stack[j_,:,:] = stack[j_,:,:]/stack[j_-nFrames,:,:]
                    
                    tempArray.append(stack[nFrames:,:,:])
    
    
    
    

def bkgSubStack(fileIn = '/ImageStackStab.h5'):

    nFramesSegment, nSegments, dims, firstFrame = dataSegmentation(fileIn, firstFrame = True)
    
    print('\nGenerating background subtracted stack:\n')
    with tables.File(cfg.FileAddressOut[0]+fileIn, 'r') as fIn, tables.File(cfg.FileAddressOut[0]+'/bkgSubStack.h5', mode = 'w') as fOut:
        atom = tables.Float32Atom()#defines the datatype as float32 to save space in stabilised image stack
        tempArray = fOut.create_earray(fOut.root, 'data', atom, (0, dims[1], dims[2])) #creates extendable array (earray) containing datatype float64

        if nSegments == 1:
            for i_ in range(0, nSegments):
                stack = fIn.root.data[i_*nFramesSegment:nFramesSegment+i_*nFramesSegment,:,:]
                
                for j_ in trange(0, stack.shape[0]):
                    stack[j_,:,:] = stack[j_,:,:] - firstFrame
                    
        else:
            for i_ in trange(0, nSegments):
                stack = fIn.root.data[i_*nFramesSegment:nFramesSegment+i_*nFramesSegment,:,:]
                
                for j_ in range(0, stack.shape[0]):
                    stack[j_,:,:] = stack[j_,:,:] - firstFrame
            
            
            tempArray.append(stack)


def noBkgStack(segMapStack, fileIn = '/ImageStackStab.h5'):
    
    nFramesSegment, nSegments, dims = dataSegmentation(fileIn)

    print('\nGenerating no background stack:\n')
    with tables.File(cfg.FileAddressOut[0]+fileIn, 'r') as fIn, tables.File(cfg.FileAddressOut[0]+'/noBkgStack.h5', mode = 'w') as fOut:
        atom = tables.Float32Atom()#defines the datatype as float32 to save space in stabilised image stack
        tempArray = fOut.create_earray(fOut.root, 'data', atom, (0, dims[1], dims[2])) #creates extendable array (earray) containing datatype float64

        if nSegments == 1:
            for i_ in range(0, nSegments):
                stack = fIn.root.data[i_*nFramesSegment:nFramesSegment+i_*nFramesSegment,:,:]
                
                for j_ in trange(0, stack.shape[0]):
                    stack[j_,:,:] = stack[j_,:,:]*segMapStack[j_,:,:]
            
            
            tempArray.append(stack)
            
        else:
            
            for i_ in trange(0, nSegments):
                stack = fIn.root.data[i_*nFramesSegment:nFramesSegment+i_*nFramesSegment,:,:]
                
                for j_ in range(0, stack.shape[0]):
                    stack[j_,:,:] = stack[j_,:,:]*segMapStack[j_,:,:]
            
            
            tempArray.append(stack)

def unflatten(particleTraces, traceName, particleIndex = None):
    
    temparr = particleTraces[traceName].to_numpy()
    
    outarr = np.reshape(temparr, [particleTraces['NumParticlesFilt'][0], int(len(particleTraces[traceName])/particleTraces['NumParticlesFilt'][0])])
    
    if particleIndex != None:
        outarr = outarr[particleIndex]
    else:
        pass

    return outarr

def meanTrace(particleTraces, traceName):
     
    arr = unflatten(particleTraces, traceName)
    traceOut = np.mean(arr,axis=0)
    
    stdvec = np.zeros(arr.shape[1])
    
    for i_ in range(0, arr.shape[1]):
            stdvec[i_] = np.std(arr[:,i_])
    
    return traceOut, stdvec




def intensityWholeFrame(fileIn):
    
    
    nFramesSegment, nSegments, dims = dataSegmentation(fileIn)
    
    trace = np.zeros(dims[0]-3) #initialise output variable
    
    print('\nGenerating whole frame mean intensity...\n')
    
    with tables.File(cfg.FileAddressOut[0]+fileIn, 'r') as fIn:
        if nSegments == 1:
        
                for i_ in range(0, nSegments):
                    stack = fIn.root.data[i_*nFramesSegment:nFramesSegment+i_*nFramesSegment,:,:]
                    temp = np.zeros([stack.shape[0]-3])
                    for j_ in trange(0,stack.shape[0]-3):
                        
                        temp[j_] = np.sum(stack[j_])/(stack.shape[1]*stack.shape[2])
                        
                        
                    trace[0:stack.shape[0]-3] = temp[:]
                
                
        else:
            
            print('untested - this bit might work - need to test with dataset larger than available memory')
            for i_ in trange(0, nSegments):
                stack = fIn.root.data[i_*nFramesSegment:nFramesSegment+i_*nFramesSegment,:,:]
                temp = np.zeros([stack.shape[0]-3])
                
                for j_ in trange(0,stack.shape[0]-3):
                        
                    temp[j_] = np.sum(stack[j_])/(stack.shape[1]*stack.shape[2])
                    
                    trace[i_*nFramesSegment:i_*nFramesSegment+nFramesSegment] = temp[:]
                    
                    
    
    return trace


# def npytoH5():
    
#     #Determine segmentation of data based on available system memory:
#     fileSize = os.path.getsize(cfg.FileAddressOut[0]+fileIn)
#     svmem = psutil.virtual_memory()

#     if fileSize < svmem.available*0.75: #allow for a quarter of available memory to remain free
#         nSegments = 1

#     else:
#         nSegments = ceil(fileSize/(svmem.available*0.75))
    
#     #read in dimensions of image stack
#     with tables.File(cfg.FileAddressOut[0]+fileIn, 'r') as fIn:
        
#         dims = fIn.root.data.shape
    
#     nFramesSegment = int(dims[0]/nSegments)
    
    
#     trace = np.zeros(dims[0]-3) #initialise output variable
    
#     print('\nGenerating whole frame mean intensity...\n')
    
#     with tables.File(cfg.FileAddressOut[0]+fileIn, 'r') as fIn, tables.File(cfg.FileAddressOut[0]+'/imageStackPreStab.h5', mode = 'w'):
#         atom = tables.Float32Atom()#defines the datatype as float32 to save space in stabilised image stack
#         tempArray = fOut.create_earray(fOut.root, 'data', atom, (0, dims[1], dims[2])) #creates extendable array (earray) containing datatype float64
        
    
    
    
#Area selection function used to select sub-image for video stabilisation
def line_select_callback(eclick, erelease):
    """
    Callback for line selection.
    
    *eclick* and *erelease* are the press and release events.
    """
    ind = cfg.ind
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    print(f" The buttons you used were: {eclick.button} {erelease.button}")
    
    
    if cfg.driftCorrectType == 'Coarse':
        cfg.sub_img_coordx[ind] = int(x1)
        cfg.sub_img_coordy[ind] = int(y1)
        
        cfg.sub_img_sizex[ind] = int(x2-x1)
        cfg.sub_img_sizey[ind] = int(y2-y1) #CHANGED TO ENFORCE SQUARE STAB BOXES
        #cfg.sub_img_sizey[ind] = cfg.sub_img_sizex[ind]+1
        
    elif cfg.driftCorrectType == 'Fine':
        cfg.sub_img_coordx_fine[ind] = int(x1)
        cfg.sub_img_coordy_fine[ind] = int(y1)
        
        cfg.sub_img_sizex_fine[ind] = int(x2-x1)
        cfg.sub_img_sizey_fine[ind] = int(y2-y1)

    
    
#Function which toggles the line_select_callback function on/off by pressing "t"
def toggle_selector(event):
    print(' Key pressed.')
    if event.key == 't':
        if toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        else:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)