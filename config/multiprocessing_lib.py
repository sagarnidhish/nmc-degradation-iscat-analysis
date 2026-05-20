# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:08:47 2021

@author: ar2071
"""

#Simulataneous data processing function
import ObjectDetection as od
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import DataManipulation as dm
import tables
import numpy as np
import DataIO as dio
import tifffile as tf
import os.path
import psutil
from math import ceil
import pandas as pd
import multiprocessing
import config as cfg
from time import sleep
import h5py
import radialcentre as rc
import pdb
from pymlfunc import normxcorr2
import photutils.centroids as cen


# to handle close event.
def handle_close(evt):
    cfg.close_flag = 1


def areaSelection(Image):
    cfg.close_flag = 0
    fig, ax = plt.subplots()
    #Plot the middle frame from the image stack
    # if ImageStack.shape[0] != 1:
    #     ax.pcolormesh(ImageStack[int(ImageStack.shape[0]/2),:,:])
    # else:
    #     ax.pcolormesh(ImageStack[0,:,:])
    ax.pcolormesh(Image)
    
    #First and last frame options commented out, one of the three can be displayed:
    #ax.pcolormesh(ImageStack[0,:,:])
    #ax.pcolormesh(ImageStack[ImageStack.shape[0],:,:])
    
    
    ax.set_title(
        "Click and drag to select sub-image area.\n"
        "Press 't' to toggle the selector on and off.")
    
    # drawtype is 'box' or 'line' or 'none'
    dm.toggle_selector.RS = RectangleSelector(ax, dm.line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # disable middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    
    fig.canvas.mpl_connect('key_press_event', dm.toggle_selector)
    
    plt.show()
    
    while cfg.close_flag == 0:
        fig.canvas.flush_events() # flush the GUI events for the figure.
        fig.canvas.mpl_connect('close_event', handle_close)
        if cfg.close_flag == 1:
            cfg.close_flag = 0
            break
    
    
    
    
    
def fineStab(ind, ImageStack, trajectoryArrayCoarse):
    
    #import area of interest and adjust window position per frame using trajectory from coarse correction
    # sub_img_coord = [cfg.sub_img_coordx_fine[ind], cfg.sub_img_coordy_fine[ind]]
    # sub_img_size = [cfg.sub_img_sizex_fine[ind], cfg.sub_img_sizey_fine[ind]]
    sub_img_coord = [cfg.sub_img_coordx[ind], cfg.sub_img_coordy[ind]]
    sub_img_size = [cfg.sub_img_sizex[ind], cfg.sub_img_sizey[ind]]
    
    sub_img_coord = [cfg.sub_img_coordx[ind], cfg.sub_img_coordy[ind]]
    sub_img_size = [cfg.sub_img_sizex[ind], cfg.sub_img_sizey[ind]]
    
    for i in range(0, ImageStack.shape[0]-1):
        croppedStack = np.flip(ImageStack[i, sub_img_coord[1]+int(trajectoryArrayCoarse[0:i+1,0].sum()):sub_img_coord[1]+sub_img_size[1]+int(trajectoryArrayCoarse[0:i+1,0].sum()), sub_img_coord[0]+int(trajectoryArrayCoarse[0:i+1,1].sum()):sub_img_coord[0]+sub_img_size[0]+int(trajectoryArrayCoarse[0:i+1,1].sum())], axis = 1)

    
    trajectoryArrayFine = np.zeros([croppedStack.shape[0],2])
    
    timevec = np.zeros([croppedStack.shape[0]])
    print("timevec shape", timevec.shape)
    for i in range(1, croppedStack.shape[0]-1): #should this be -2 as before? -2 ---> -1 compensates for the -1 end condition in the previous loop
        #Fourier Filter high frequencies
        #Operation for i-1th frame
        fftframe = np.fft.fftshift(croppedStack[i-1])

        hist, bins = np.histogram(fftframe)
        histind = np.unravel_index(hist.argmax(), hist.shape) #finds the index of the histogram data with highest no. occurrances
        
        fftframe[fftframe < bins[histind[0]+1]] = 0
        frame0 = np.fft.ifftshift(fftframe).real
        
        #repeat for i-th frame
        fftframe = np.fft.fftshift(croppedStack[i])

        hist, bins = np.histogram(fftframe)
        histind = np.unravel_index(hist.argmax(), hist.shape) #finds the index of the histogram data with highest no. occurrances
        
        fftframe[fftframe < bins[histind[0]+1]] = 0
        frame1 = np.fft.ifftshift(fftframe).real
        
        #Normalised cross-correlation
        nxc = normxcorr2(frame0,frame1)
        
        #Centroid centre of mass localisation
        xp, yp = np.unravel_index(nxc.argmax(), nxc.shape)
        xtemp,ytemp = cen.centroid_sources(nxc, xp,yp, box_size = [9,9], centroid_func=cen.centroid_com) #centroid_func arg changes localisation algorithm
        timevec[i] = i
        
        
        # trajectoryArrayFine[i,0] = xtemp - int(nxc.shape[0]/4) - 1 + int(trajectoryArrayCoarse[0:i,0].sum()) #might be the correct way to generate a cumulative trajectory
        # trajectoryArrayFine[i,1]= ytemp - int(nxc.shape[1]/4) - 1 + int(trajectoryArrayCoarse[0:i,1].sum())
        
        if i != 0:
            trajectoryArrayFine[i,0] = xtemp[i] - xtemp[i-1] + trajectoryArrayCoarse[i,0] #trajectory arrays are dx, dy between consecutive frames, not cumulative
            trajectoryArrayFine[i,1] = ytemp[i] - ytemp[i-1] + trajectoryArrayCoarse[i,1]

        
        else:
            trajectoryArrayFine[i,0] = xtemp + trajectoryArrayCoarse[i,0] - 0 
            trajectoryArrayFine[i,0] = ytemp + trajectoryArrayCoarse[i,1] - 0
    
    
    plt.figure()
    plt.title('coarse')
    plt.plot(timevec, trajectoryArrayCoarse[:,0])
    plt.plot(timevec, trajectoryArrayCoarse[:,1])
    plt.show()
    
    plt.figure()
    plt.title('fine')
    plt.plot(timevec, trajectoryArrayFine[:,0])
    plt.plot(timevec, trajectoryArrayFine[:,1])
    plt.show()
    
    return trajectoryArrayFine
    
#stabilise and interpolate a stack
def StabInt(ind, ImageStack):
    
    #Coarse Stabilisation
    trajectoryArrayCoarse = dm.StabiliseStack(ind, ImageStack) #sub-img coords import happens in this function
    print('coarse stab done')
    
    
    # trajectoryArray = fineStab(ind, ImageStack, trajectoryArrayCoarse)
    # print('fine stab done')

    
    ImageStack_stab = dm.interpolateStack(ImageStack, trajectoryArrayCoarse)
    np.savetxt(cfg.FileAddressOut[0]+'/trajectoryArrayCoarse.npy', trajectoryArrayCoarse)

    
    
    return ImageStack_stab


def ObjDetCombined(ind, xSize, ySize, filename):
    
    #Load every Nth frame from stabilised stack
    f = tables.open_file(cfg.FileAddressOut[ind]+"/"+filename, mode='r')
    
    if cfg.filetype == ".tif":
        ImageStacknFrames = np.zeros([int(cfg.ImageStackFrames[ind]/cfg.nFramesAnalysis)+1,ySize,xSize])
    elif cfg.filetype == ".tdms":
        ImageStacknFrames = np.zeros([int(cfg.ImageStackFrames[ind]/cfg.nFramesAnalysis)+1,xSize,ySize])
    elif cfg.filetype == '.npy':
        ImageStacknFrames = np.zeros([int(cfg.ImageStackFrames[ind]/cfg.nFramesAnalysis)+1,xSize,ySize])

            
    else:
        raise TypeError
    
    print('Loading stabilised data...')
    for i_ in range(0,cfg.ImageStackFrames[ind]-1, cfg.nFramesAnalysis):
        
        ImageStacknFrames[int(i_/cfg.nFramesAnalysis),:,:] =  f.root.data[i_,:,:]

    print('Stabilised data loaded')
    f.close()
    
    #Load mmap:
    #ImageStacknFrames = np.memmap(cfg.FileAddressOut[ind]+'/temp.npy',mode = 'r+', dtype='float32', shape = (cfg.ImageStackFrames[ind],ySize,xSize))


    #Single frame object detection
    objects, seg_map, img_sub = od.SingleFrameDetection(ImageStacknFrames[0,:,:])
    
    #plot frame with ellipses marking detected particles
    #od.EllipseHighlighting(objects, img_sub)
    
    #Make segmentation map a binary array and invert values so particles = 0, carbon matrix = 1 
    seg_map = od.InverseBinary(seg_map)
    

    #dataframe values (https://sep.readthedocs.io/en/stable/api/sep.extract.html#sep.extract):

    #Set up key object detection variables
    maxObjects, segMapStack, maxSegMap = od.ObjectDetectionSetup(ImageStacknFrames, ind)
    
    
    #Run object detection on every Nth frame and save the sizes (pixels) and intensities per particle
    particleSizes, particleFluxes, particleIntensities = od.ObjectDetectionStack(ImageStacknFrames, maxObjects, segMapStack, maxSegMap)
     
    #Remove NaN values by linear interpolation
    particleFluxes = dm.nanInterp(particleFluxes)
    particleSizes = dm.nanInterp(particleSizes)
    particleIntensities = dm.nanInterp(particleIntensities)
 

    return maxObjects, particleSizes, segMapStack, maxSegMap, particleFluxes, particleIntensities

    

def mainfunc(firstFrame=0): #firstFrame optional argument to skip frames at the start if non-zero
    
    if cfg.BENCHMARK_MODE == True:
            print('Benchmark mode:', cfg.BENCHMARK_MODE)
            cfg.sizeThreshold = 0
    else:
        pass
    
    #Initialise stabilisation related variables
    cfg.sub_img_coordx = ['']*len(cfg.OpticalDataInPath)
    cfg.sub_img_coordy = ['']*len(cfg.OpticalDataInPath)
    cfg.sub_img_sizex = ['']*len(cfg.OpticalDataInPath)
    cfg.sub_img_sizey = ['']*len(cfg.OpticalDataInPath)
    cfg.sub_img_coordx_fine = ['']*len(cfg.OpticalDataInPath)
    cfg.sub_img_coordy_fine = ['']*len(cfg.OpticalDataInPath)
    cfg.sub_img_sizex_fine = ['']*len(cfg.OpticalDataInPath)
    cfg.sub_img_sizey_fine = ['']*len(cfg.OpticalDataInPath)
    
    
    cfg.ImageStackFrames = ['']*len(cfg.OpticalDataInPath)
    cfg.nFramesSegment = ['']*len(cfg.OpticalDataInPath)
    
    cfg.ind = 0
    
    #Select area used in coarse drift correction:
    if cfg.stabCheck == 0:
        for cfg.ind in range(0,len(cfg.OpticalDataInPath)):
            
            cfg.driftCorrectType = 'Coarse'
    

            #Use the first frame of the image stack to select a region for video stabilisation
            if dio.DataFileType(cfg.OpticalDataInPath[cfg.ind]) == ".tdms":
                #Find total number of frames in entire dataset
                cfg.ImageStackFrames[cfg.ind], xSize, ySize, cfg.filetype = dio.readFileDims(cfg.ind)
                middleFrame,_, cfg.frameRate = dio.RAWtoNPY(cfg.ind)
                areaSelectFrame = middleFrame[0,:,:]
                areaSelection(areaSelectFrame)
                while len(cfg.sub_img_coordx) == 0:
                    sleep(0.5)
                    print(len(cfg.sub_img_coordx))
                else:
                    pass
        
            elif dio.DataFileType(cfg.OpticalDataInPath[cfg.ind]) ==".tif":
                
                cfg.ImageStackFrames[cfg.ind], xSize, ySize, cfg.filetype = dio.readFileDims(cfg.ind)
                areaSelectFrame = tf.imread(cfg.OpticalDataInPath[cfg.ind], key = 0)
                areaSelection(areaSelectFrame)
                while len(cfg.sub_img_coordx) == 0:
                    sleep(0.5)
                    print(len(cfg.sub_img_coordx))
                else:
                    pass
        
            elif dio.DataFileType(cfg.OpticalDataInPath[cfg.ind]) ==".npy":
                
                ImageStack = np.load(cfg.OpticalDataInPath[cfg.ind])
                #ImageStack = dtt[0]
                print(ImageStack.shape)
                cfg.ImageStackFrames[cfg.ind] = ImageStack.shape[0]
                xSize = ImageStack.shape[1]
                ySize = ImageStack.shape[2]
                cfg.filetype = ".npy"
                areaSelectFrame = ImageStack[0,:,:]
                areaSelection(areaSelectFrame)        
                while len(cfg.sub_img_coordx) == 0:
                    sleep(0.5)
                    print(len(cfg.sub_img_coordx))
                else:
                    pass
            else:
                TypeError("File type is not supported") 
    else:
        TypeError('FEATURE NOT YET IMPLEMENTED')
        print('\n\nImage stack will not be stabilised\n\n')
        #Convert tdms/npy/tiff 'stabilised' input to h5 (without full loading of data)
        
        if dio.DataFileType(cfg.OpticalDataInPath[cfg.ind]) == ".tdms":
                #Find total number of frames in entire dataset
                cfg.ImageStackFrames[cfg.ind], xSize, ySize, cfg.filetype = dio.readFileDims(cfg.ind)
                _,_, cfg.frameRate = dio.RAWtoNPY(cfg.ind)

        
        elif dio.DataFileType(cfg.OpticalDataInPath[cfg.ind]) ==".tif":
                
                cfg.ImageStackFrames[cfg.ind], xSize, ySize, cfg.filetype = dio.readFileDims(cfg.ind)


        
        elif dio.DataFileType(cfg.OpticalDataInPath[cfg.ind]) ==".npy":
                
                ImageStack = np.load(cfg.OpticalDataInPath[cfg.ind])
                #ImageStack = dtt[0]
                print(ImageStack.shape)
                cfg.ImageStackFrames[cfg.ind] = ImageStack.shape[0]
                xSize = ImageStack.shape[1]
                ySize = ImageStack.shape[2]
                
                cfg.filetype = ".npy"
        else:
            TypeError("File type is not supported")
    
    # #Select area for fine drift correction:
    # for cfg.ind in range(0,len(cfg.OpticalDataInPath)):
        
    #     cfg.driftCorrectType = 'Fine'
        
    #     #Use the middle frame from the entire image stack to select a region for video stabilisation
    #     if dio.DataFileType(cfg.OpticalDataInPath[cfg.ind]) == ".tdms":
            
    #         areaSelection(areaSelectFrame)
    #         while len(cfg.sub_img_coordx_fine) == 0:
    #             sleep(0.5)
    #             print(len(cfg.sub_img_coordx_fine))
    #         else:
    #             pass
    
    #     elif dio.DataFileType(cfg.OpticalDataInPath[cfg.ind]) ==".tif":
            
    #         areaSelection(areaSelectFrame)
    #         while len(cfg.sub_img_coordx_fine) == 0:
    #             sleep(0.5)
    #             print(len(cfg.sub_img_coordx_fine))
    #         else:
    #             pass
    
    #     elif dio.DataFileType(cfg.OpticalDataInPath[cfg.ind]) ==".npy":

    #         areaSelection(areaSelectFrame)        
    #         while len(cfg.sub_img_coordx_fine) == 0:
    #             sleep(0.5)
    #             print(len(cfg.sub_img_coordx_fine))
    #         else:
    #             pass
    #     else:
    #         TypeError("File type is not supported") 
                
              
                
    #Second loop iterates analysis over each dataset
    for ind in range(0,len(cfg.OpticalDataInPath)):
    
        #Determine segmentation of data based on available system memory:
        fileSize = os.path.getsize(cfg.OpticalDataInPath[ind])
        svmem = psutil.virtual_memory()
    
        if fileSize < svmem.available*0.4: #allow for a quarter of available memory to remain free
            nSegments = 1
    
        else:
            nSegments = ceil(fileSize/(svmem.available*0.75))

        cfg.nFramesSegment[ind] = int(cfg.ImageStackFrames[ind]/nSegments) #determine no. frames per segment (base on system RAM in future)
        
        # #Initialise and open hdf5 file:
        with tables.File(cfg.FileAddressOut[ind]+'/ImageStackStab.h5', mode = 'w') as fOut:
            #defines the datatype as float32 to save space in stabilised image stack
            atom = tables.Float32Atom()
            timesIntensity = np.zeros([2,cfg.ImageStackFrames[ind]])
            
            if cfg.filetype == '.tdms':    
                tempArray = fOut.create_earray(fOut.root, 'data', atom, (0, xSize, ySize)) #creates extendable array (earray) containing datatype float64
            
            elif cfg.filetype == '.tif' or cfg.filetype == '.npy':
                tempArray = fOut.create_earray(fOut.root, 'data', atom, (0, ySize, xSize))

            else:
                print('Filetype error')
                
            #NOTE: zarr format might be needed for multiprocessing, zarr can async write/read without synchronisation issues as long as each processor is writing to a separate chunk
            
            for i_ in range(0,nSegments):
            
                #Import .tdms optical data and convert to .npy array
                print("Segment no.: ", i_+1,"/",nSegments)
                
                ImageStack,timesIntensity[:,i_*cfg.nFramesSegment[ind]:(i_*cfg.nFramesSegment[ind])+cfg.nFramesSegment[ind]],_ = dio.RAWtoNPY(ind, cfg.nFramesSegment[ind], firstFrame+(cfg.nFramesSegment[ind]*i_))
                
                
                ImageStack = StabInt(ind, ImageStack)
                tempArray.append(ImageStack)#stabilise and interpolate
    
        
        # with tables.File(cfg.FileAddressOut[ind]+'/ImageStackStabCOARSE.h5', mode = 'w') as fcoarse:
            
        #     atom = tables.Float32Atom()
        #     timesIntensity = np.zeros([2,cfg.ImageStackFrames[ind]])

        #     if cfg.filetype == '.tdms':    
        #         tempArray = fcoarse.create_earray(fcoarse.root, 'data', atom, (0, xSize, ySize)) #creates extendable array (earray) containing datatype float64
                
            
        #     elif cfg.filetype == '.tif' or cfg.filetype == '.npy':
        #         tempArray = fcoarse.create_earray(fcoarse.root, 'data', atom, (0, ySize, xSize))

        #     else:
        #         print('Filetype error')
            
        #     for i_ in range(0,nSegments):
            
        #         #Import .tdms optical data and convert to .npy array
        #         print("Segment no.: ", i_+1,"/",nSegments)
                
        #         ImageStack,timesIntensity[:,i_*cfg.nFramesSegment[ind]:(i_*cfg.nFramesSegment[ind])+cfg.nFramesSegment[ind]],_ = dio.RAWtoNPY(ind, cfg.nFramesSegment[ind], firstFrame+(cfg.nFramesSegment[ind]*i_))
                
        #         #coarse stab
        #         stack, trajectoryArrayCoarse = StabInt(ind, ImageStack)
 
                
        #         tempArray.append(stack)#stabilise and interpolate


        #Object detection routine:
        maxObjects, particleSizes, segMapStack, maxSegMap, particleFluxes, particleIntensities = ObjDetCombined(ind, xSize, ySize, '/ImageStackStab.h5')
    
        #For each of flux, size and intensity:
        #1. unfiltered - normalised, standardised
        #2. filtered - norm., stand.
        #3. 4 output arrays per input array - 12 total
    
        df = pd.DataFrame()
        shape = particleFluxes.shape
        #Column storing the max number of particles detected (raw), and the no. valid particles (filtered)
        df['NumParticlesRaw'] = np.zeros(shape[0]*shape[1])
        df['NumParticlesRaw'] = maxObjects.shape[0]
        df['NumParticlesFilt'] = shape[0]

        #Particle sizes in pixels (raw and S-G filtered)
        df['SizesPix'] = particleSizes.flatten()
        df['SizesPixFilt'] = dm.SavitskyGolayArrayFilter(df['SizesPix'][:], shape)
        
        #Normalised and standardised fluxes, sizes
        df['FluxesNorm'], df['FluxesStand'] = dm.normAndStand(particleFluxes, shape)
        df['SizesNorm'], df['SizesStand'] = dm.normAndStand(particleSizes, shape)
        df['IntensitiesNorm'], df['IntensitiesStand'] = dm.normAndStand(particleIntensities, shape)

        
        
        #Savitsky-Golay filter applied to normalised data
        df['FluxesNormFilt'] = dm.SavitskyGolayArrayFilter(df['FluxesNorm'][:], shape)
        df['SizesNormFilt'] = dm.SavitskyGolayArrayFilter(df['SizesNorm'][:], shape)
        df['IntensitiesNormFilt'] = dm.SavitskyGolayArrayFilter(df['IntensitiesNorm'][:], shape)
        
        
        #Savitsky-Golay filter applied to standardised data
        df['FluxesStandFilt'] = dm.SavitskyGolayArrayFilter(df['FluxesStand'][:], shape)
        df['SizesStandFilt'] = dm.SavitskyGolayArrayFilter(df['SizesStand'][:], shape)
        df['IntensitiesStandFilt'] = dm.SavitskyGolayArrayFilter(df['IntensitiesStand'][:], shape)
        
        #Save one vector as the frame times for future access - so that it is saved with the relevant traces
        df['frameRate'] = cfg.frameRate 

        
        #Save dataframes
        df.to_csv(cfg.FileAddressOut[ind]+'/particleTraces.csv')
        maxObjects.to_csv(cfg.FileAddressOut[ind]+'/maxObjects.csv')
        
        #Save segmentation maps (as boolean)
        np.save(cfg.FileAddressOut[ind]+'/segMapStack.npy', segMapStack.astype(bool))
        np.save(cfg.FileAddressOut[ind]+'/maxSegMap.npy', maxSegMap.astype(bool))
        

        np.save(cfg.FileAddressOut[ind]+'/timesIntensity.npy', timesIntensity)
        np.save(cfg.FileAddressOut[ind]+'/frameRate.npy', cfg.frameRate)
