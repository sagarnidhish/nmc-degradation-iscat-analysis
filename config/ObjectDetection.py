# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:18:51 2021

@author: ar2071
"""

#object detection library


import numpy as np
import sep
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from tqdm import trange
import time
import DataManipulation as dm
import config as cfg


def SingleFrameDetection(frame):
    
    bkg = sep.Background(frame, bw=60, bh=60, fw=3, fh=3) #bw, bh, fw, fh set to default vals
    
    #print(bkg.globalback)
    #print(bkg.globalrms)

    bkg_image = np.array(bkg)
    #bkg_rms = bkg.rms()
    
    
    data_sub = frame-bkg_image

    objects,seg_map = sep.extract(data_sub, 6, err=bkg.globalrms, deblend_cont = 0.3, segmentation_map=True)
    
    seg_map = BinariseImage(seg_map)

    return objects, seg_map, data_sub

def SingleFrameDetectionSEGMAP(frame, seg_map):
    
    #frame = frame*seg_map
    bkg = sep.Background(frame, bw=60, bh=60, fw=3, fh=3) #bw, bh, fw, fh set to default vals
    
    #print(bkg.globalback)
    #print(bkg.globalrms)
    
    bkg_image = np.array(bkg)
    #bkg_rms = bkg.rms()
    
    
    data_sub = frame-bkg_image
    #apply binary mask to each frame
    #data_sub = data_sub*seg_map
    
    if cfg.sizeThreshold == 0:
        objects,seg_map = sep.extract(data_sub, 6, err=bkg.globalrms, deblend_cont = 0.3, segmentation_map=True)
    else:
        objects,seg_map = sep.extract(data_sub, 6, err=bkg.globalrms, deblend_cont = 0.3, segmentation_map=True, minarea = cfg.sizeThreshold)
        
        
    seg_map = BinariseImage(seg_map)

    return objects

def EllipseHighlighting(objects, img_sub):
    
    # plot background-subtracted image
    fig, ax = plt.subplots()
    m, s = np.mean(img_sub), np.std(img_sub)
    ax.imshow(img_sub, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
    
    # plot an ellipse for each object
    for i_ in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i_], objects['y'][i_]),
                    width=6*objects['a'][i_],
                    height=6*objects['b'][i_],
                    angle=objects['theta'][i_] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
        
    return 



def BinariseImage(image):
    #Convert "binary" image into true binary image
    #Image with bkg = 0, particles != 0 --> bkg = 0, particles = 1
    image[image!=0] = 1
    
    return image

def InverseBinary(image):
    #Convert binary image into its inverse
    #bkg = 0, particles = 1 --> bkg = 1, particles = 0
    image[image!=0] = 1
    image[image==0] = 100
    image[image==1] = 0
    image[image==100] = 1
    
    return image


def ObjectDetectionSetup(ImageStack_stab,ind):
    
    
    print("Object detection set-up:")
    time.sleep(0.2) #required wait
     
    #segMapStack = np.memmap(cfg.FileAddressOut[ind]+'/segMapStack.npy', mode = 'w+',dtype='bool',shape = (int(ImageStack_stab.shape[0])+1, ImageStack_stab.shape[1], ImageStack_stab.shape[2]))
    segMapStack = np.zeros(ImageStack_stab.shape)
    #initialise variable
    maxParticles = 0
    

    #Loop to find the maximum number of objects detected given the selection criteria
    for i_ in trange(0,ImageStack_stab.shape[0]):
        
        #find background for frame
        
        bkg = sep.Background(ImageStack_stab[i_,:,:], bw=60, bh=60, fw=3, fh=3) #explore additional args
    
        #background quantities
        bkg_image = np.array(bkg)
        #bkg_rms = bkg.rms()
    
        #subtract background from frame
        img_sub = ImageStack_stab[i_,:,:]-bkg_image
    
    
        
        if cfg.sizeThreshold == 0:
            #object detection - output particle parameters and segmentation map
            objects,seg_map = sep.extract(img_sub, 6, err=bkg.globalrms, deblend_cont = 0.3, segmentation_map=True)
        else:
            objects,seg_map = sep.extract(img_sub, 6, err=bkg.globalrms, deblend_cont = 0.3, segmentation_map=True, minarea = cfg.sizeThreshold)

        #save all segmentation maps
        segMapStack[int(i_),:,:] = seg_map
        

        if len(objects) >= maxParticles:
            
            maxParticles = len(objects)
            maxSegMap = seg_map
            tempmaxObjects = objects
            
        else:
            pass

    
    #convert segmentation maps to binary
    maxSegMap = BinariseImage(maxSegMap)
    segMapStack = BinariseImage(segMapStack)
    
    #Flush memmap changes to disk
    #segMapStack.flush()
    
    
    maxObjects = pd.DataFrame(tempmaxObjects)
    
    return maxObjects, segMapStack, maxSegMap


def ObjectDetectionStack(ImageStack_stab, maxObjects, segMapStack, maxSegMap):
    
    print("Full stack object detection:")
    time.sleep(0.4)
    
    #initialise array to hold particle size (pixels) over time
    particleSizesTemp = np.zeros([len(maxObjects),int(ImageStack_stab.shape[0])+1])
    particleIntensitiesTemp = np.zeros([len(maxObjects),int(ImageStack_stab.shape[0])+1])
    particleFluxesTemp = np.zeros([len(maxObjects),int(ImageStack_stab.shape[0])+1])

    particleSizesTemp[:,:] = np.nan
    particleIntensitiesTemp[:,:] = np.nan
    particleFluxesTemp[:,:] = np.nan
    
    for i_ in trange(0, ImageStack_stab.shape[0]):
        #Single frame object detection with binary mask applied pre-detection
        objects = SingleFrameDetectionSEGMAP(ImageStack_stab[i_,:,:], segMapStack[int(i_),:,:])
        currObjects = pd.DataFrame(objects)
        
        #Particle linking by matching currObjects barycentres to maxObjects barycentres - nan inserted where no particle is matched
        for k_ in range(0,len(currObjects)):

            for l_ in range(0,len(maxObjects)):

                if currObjects['x'][k_] > maxObjects['x'][l_]-cfg.eta and currObjects['x'][k_] < maxObjects['x'][l_]+cfg.eta and currObjects['y'][k_] > maxObjects['y'][l_]-cfg.eta and currObjects['y'][k_] < maxObjects['y'][l_]+cfg.eta:
                    
                    particleIntensitiesTemp[l_,int(i_)] = currObjects['flux'][k_]/currObjects['npix'][k_]
                    particleSizesTemp[l_, int(i_)] = currObjects['npix'][k_]
                    particleFluxesTemp[l_, int(i_)] = currObjects['flux'][k_]
                   
                    
                else:       
                    pass
    
    #Clean particles with too many NaN values
    particleIntensities = dm.nanParticleFilter(particleIntensitiesTemp)
    particleSizes = dm.nanParticleFilter(particleSizesTemp)
    particleFluxes = dm.nanParticleFilter(particleFluxesTemp)
    
    
    return particleSizes, particleFluxes, particleIntensities