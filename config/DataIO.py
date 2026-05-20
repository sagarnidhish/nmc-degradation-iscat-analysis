# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:21:08 2021

@author: ar2071
"""

#data export library
import numpy as np
from nptdms import TdmsFile
import os
import tifffile as tf
import psutil

import tkinter as tk
from tkinter import ttk, Tk, IntVar, Checkbutton, Entry, Button, Label
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import filedialog
import pandas as pd
import tables


import config as cfg
import DataManipulation as dm

def echemPathSetup():
    
    cfg.close_flag = 0
    
    root = Tk()
    root.title('SELECT ECHEM INPUT DIRECTORY')
    root.withdraw()
    cfg.EChemDirectory = filedialog.askdirectory()
    print(cfg.EChemDirectory)
    
    cfg.close_flag = 1

def selectFiles():
    global tempfilenames
    filetypes = (('All files', '*.*'), ('Text files', '.txt'))

    tempfilenames = fd.askopenfilenames(
        title='Open files',
        initialdir='/',
        filetypes=filetypes)

    showinfo(
        title='Selected Files',
        message=tempfilenames
    )


def dataPathSetup():
    
    #Set close flag to 0 - indicating a tkinter window is currently open/active
    #cfg.close_flag = 0
    
    #Initiliase the "stabilise stack?" variable to one (Yes) as default
    #cfg.stabCheck = 1
    
    root = tk.Tk()
    root.title('SELECT INPUT DATASETS')
    root.resizable(False, False)
    root.geometry('300x150')
    root.attributes('-topmost', True)
    
    open_button = ttk.Button(
        root,
        text='Open Files',
        command=selectFiles
    )
    
    open_button.pack(expand=True)
    
    #The c1 checkbox returns var1 to check if user wants the input data to be stabilised, if var1 = 0 (No), var1 = 1 (Yes)
    var1 = IntVar()
    c1 = Checkbutton(root, text = 'Stabilise Data', variable = var1, onvalue = 0, offvalue = 1)
    c1.pack()
    
    #Frame rate input label
    Label(root, text="Acquisition Frame Rate:").pack()
    
    frameRate = tk.DoubleVar()
    frameRateEntry = Entry(root, textvariable=frameRate)
    frameRateEntry.pack(expand = True)
    frameRateEntry.focus()
    
    
    root.mainloop()

    #Initialise empty lists
    cfg.OpticalDataInPath = ['']*len(tempfilenames)
    cfg.OpticalDataInName = ['']*len(tempfilenames)
    cfg.FileAddressOut = ['']*len(tempfilenames)
    cfg.EChemDirectoryOut = ['']*len(tempfilenames)
    
    #Extract image stack filename and absolute path
    for i_ in range(0,len(tempfilenames)):
        cfg.OpticalDataInPath[i_] = tempfilenames[i_]
        cfg.OpticalDataInName[i_] = os.path.basename(os.path.normpath(cfg.OpticalDataInPath[i_]))
        path = os.path.dirname(os.path.abspath(cfg.OpticalDataInPath[i_]))
        
        #Create output directory if one does not exist
        cfg.FileAddressOut[i_] = path+'/'+'Output - '+cfg.OpticalDataInName[i_]
        #cfg.EChemDirectoryOut[i_] = cfg.FileAddressOut[i_]+'/EChem'
        
        if os.path.isdir(cfg.FileAddressOut[i_]) == False:
            os.mkdir(cfg.FileAddressOut[i_])
            #os.mkdir(cfg.EChemDirectoryOut[i_])
            os.mkdir(cfg.FileAddressOut[i_]+'/Plots')
            os.mkdir(cfg.FileAddressOut[i_]+'/Images')
            os.mkdir(cfg.FileAddressOut[i_]+'/Traces')
            #os.mkdir(cfg.FileAddressOut[i_]+'/EChem')
        
        else:
            pass

    #Assign the "stabilise stack?" variable to the config namespace for global access
    cfg.stabCheck = var1.get()
    
    #Assign framerate variable to the config namespace
    cfg.frameRate = frameRate.get()
    #cfg.close_flag = 1
            


def openfile(entryBox):
    global filename
    filename = fd.askopenfilename()
    entryBox.delete(0, tk.END)
    entryBox.insert(tk.END, filename)
    

#Hard coded info for files directy produced by Run iScams software
def readTDMS(file, nFrames=-1, firstFrame = -1):


    with TdmsFile.open(file) as tdms_file:
        channel = tdms_file["img"]["cam1"]
        properties = channel.properties                          #set properties
        xSize = int(properties["Image size"])                    #get x size,  after drift correction this would be 'Width', before 'Image size'
        ySize = int(properties["Image size 2"])                  #get y size,  after drift correction this would be 'Heigth', before 'Image size 2'
        framerate = (properties["Effective frame rate"])                  #after drift correction this would be 'FPS', before 'Effective frame rate'
        frames = int(len(channel[:])/xSize/ySize)
        
        if nFrames == -1 and firstFrame == -1:
            data = channel[:] #read full data stack at once
        else:
            data = channel[firstFrame*xSize*ySize:(firstFrame+nFrames)*xSize*ySize] #read data partially

        
        
    return data, xSize, ySize, framerate, frames

#Hard coded info for files directy produced by labview 'drift correction' software
def readTDMS_stab(file, nFrames=-1, firstFrame = -1):
    '''reads TDMS file for TAM
    author NG, last change 17/04/20'''

    tdms_file = TdmsFile(file)  
    group = tdms_file["group"]     #after drift correction this would be 'group', before it is 'img'
    channel = group["0"]      # after drift correct '0', before 'cam1'
    
    properties = channel.properties                          #set properties
    xSize = int(properties["Width"])                    #get x size,  after drift correction this would be 'Width', before 'Image size'
    ySize = int(properties["Heigth"])                  #get y size,  after drift correction this would be 'Heigth', before 'Image size 2'
    framerate = (properties["FPS"])                  #after drift correction this would be 'FPS', before 'Effective frame rate'
    frames = int(len(channel[:])/xSize/ySize)
    
    data = channel[:]
    return data, xSize, ySize, framerate, frames

def readFileDims(ind):
    filetype = DataFileType(cfg.OpticalDataInPath[ind])
    
    #Determine whether the partition has sufficient space for stabilised file:
    fileSize = os.path.getsize(cfg.OpticalDataInPath[ind])
    partitions = psutil.disk_partitions()
    for partition in partitions:
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            freeStorage = int(partition_usage.free)
        except PermissionError:
            #exception can occur when disk isnt ready
            continue
    
    if fileSize*2 < freeStorage:
    
        if filetype == '.tdms':
            with TdmsFile.open(cfg.OpticalDataInPath[ind]) as tdms_file:
                channel = tdms_file["img"]["cam1"]
                properties = channel.properties
                xSize = int(properties["Image size"])                    
                ySize = int(properties["Image size 2"])
                frames = int(len(channel[:])/xSize/ySize)

                    
        elif filetype == '.tif':
            
            #Open file metadata
            tif = tf.TiffFile(cfg.OpticalDataInPath[ind])
            #find pixel resolution of image
            xSize = tif.pages[0].shape[1]
            ySize = tif.pages[0].shape[0]
            
            #no. frames
            frames = len(tif.pages)

        elif filetype == '.npy':
            
            with open(cfg.OpticalDataInPath[ind], 'rb') as f:
                major, minor = np.lib.format.read_magic(f)
                shape, _, dtype = np.lib.format.read_array_header_1_0(f)
                
                xSize = shape[2]
                ySize = shape[1]

                
        else:
            print('Filetype not supported')
    else:
        print('Storage required for stabilised data: ', get_size(fileSize*2))
        print('Storage Available: ', get_size(freeStorage))
        raise OSError
        
        
    return frames, xSize, ySize, filetype

def RAWtoNPY(ind, nFrames=-1, firstFrame=-1):

    filetype = DataFileType(cfg.OpticalDataInPath[ind])
    
    FrameRate = None #remove once implemented for .tif 
    
    if filetype == '.tdms':
        if nFrames == -1 and firstFrame == -1:
            dataset = readTDMS(cfg.OpticalDataInPath[ind])
            TotFrames = dataset[4]
        else:
            dataset = readTDMS(cfg.OpticalDataInPath[ind], nFrames, firstFrame)
            TotFrames = nFrames
            
        #Name useful parameters related to the dataset
        xSize = dataset[1]
        ySize = dataset[2]
        FrameRate = dataset[3]
        
        
        print('Frame rate ',FrameRate)
        print('Frame number ',TotFrames)
        
        #For the ImageStack, the first index gives the frame number of the image (time direction), the second index gives the row number of the pixel in each image, and the third index gives the column number in each image 
        Intensity = []
        ImageStack = []
        
        for frame in range(0,TotFrames):
            imag = dataset[0][frame*xSize*ySize:(frame+1)*xSize*ySize]
            Intensity.append(np.sum(imag))
            ImageStack.append(imag)
        
        Intensity = np.array(Intensity)/(xSize*ySize)
        ImageStack = np.array(ImageStack)
        ImageStack = np.reshape(ImageStack,(TotFrames,xSize,ySize)) 
        
        #Constrcut a list of time points corresponding to the frames
        times = np.arange(0,TotFrames,1)/FrameRate
        
        #Construct an array listing the time points and mean intensity values togther
        timesIntensity = np.vstack((times,Intensity))
        
        
    elif filetype == '.tif':
        timesIntensity = np.nan
        if nFrames == -1 and firstFrame == -1:
            ImageStack = tf.imread(cfg.OpticalDataInPath[ind])
            #ImageStack = np.reshape(ImageStack, [dataset.shape[0],dataset.shape[2],dataset.shape[1]])
        else:
            print('firstFrame',firstFrame)
            print('nframes', nFrames)
            print('test',range(firstFrame,firstFrame+nFrames))
            ImageStack = tf.imread(cfg.OpticalDataInPath[ind], key = range(firstFrame,firstFrame+nFrames))
            #ImageStack = np.reshape(ImageStack, [ImageStack.shape[0],ImageStack.shape[2],ImageStack.shape[1]])
            
    elif filetype == '.npy':
        ImageStack = np.load(cfg.OpticalDataInPath[ind])

        xSize = ImageStack.shape[2]
        ySize = ImageStack.shape[1]
        TotFrames = ImageStack.shape[0]
        ######
        FrameRate = np.nan #READ FRAMERATE FROM THE METADATA .TXT FILE ATTACHED TO TIF STACKS
        ######
    
        print('Frame number ', TotFrames)
        
        #ImageStack = tf.imread(fileaddress+filename, key = range(firstFrame,firstFrame+nFrames))
        #####
        timesIntensity = np.nan #Implement timesintensity for .tif?
        #####
        
    else:
        print('Filetype not supported')
        
    #Construct the ImageStack array, and also a list of mean intensity values:

    
    return ImageStack, timesIntensity, FrameRate



def TIFtoNPY(DataIn, nFrames=-1, firstFrame=-1):
    

    
    if nFrames == -1 and firstFrame == -1:
        dataset = tf.imread(DataIn)
        

    else:
        dataset = tf.imread(DataIn, key = range(firstFrame,firstFrame+nFrames))
        
    xSize = dataset.shape[2]
    ySize = dataset.shape[1]
    totFrames = dataset.shape[0]
    
    
    print('Frame number ', totFrames)
    
    
    
    
    return dataset, xSize, ySize



#Converts labview's TDMS (.tdms) output to numpy array stack (.npy)
def TDMStoNPY(DataIn, nFrames=-1, firstFrame=-1):
 
    if nFrames == -1 and firstFrame == -1:
        dataset = readTDMS(DataIn)
        TotFrames = dataset[4]
    else:
        dataset = readTDMS(DataIn, nFrames, firstFrame)
        TotFrames = nFrames
        
    #Name useful parameters related to the dataset
    xSize = dataset[1]
    ySize = dataset[2]
    FrameRate = dataset[3]
    
    
    print('Frame rate ',FrameRate)
    print('Frame number ',TotFrames)
    
    
    #Construct the ImageStack array, and also a list of mean intensity values:
    #For the ImageStack, the first index gives the frame number of the image (time direction), the second index gives the row number of the pixel in each image, and the third index gives the column number in each image 
    Intensity = []
    ImageStack = []
    for frame in range(0,TotFrames):
        imag = dataset[0][frame*xSize*ySize:(frame+1)*xSize*ySize]
        Intensity.append(np.sum(imag))
        ImageStack.append(imag)
    Intensity = np.array(Intensity)/(xSize*ySize)
    ImageStack = np.array(ImageStack)
    ImageStack = np.reshape(ImageStack,(TotFrames,xSize,ySize)) 
    
    #Constrcut a list of time points corresponding to the frames
    times = np.arange(0,TotFrames,1)/FrameRate
    
    #Construct an array listing the time points and mean intensityvalues togther
    timesIntensity = np.vstack((times,Intensity))
    

    return ImageStack, timesIntensity


def DataFileType(dataAddress):

    dataAddress = [dataAddress]
    
    for fp in dataAddress:
        # Split the extension from the path and normalise it to lowercase.
        ext = os.path.splitext(fp)[-1].lower()

    return ext

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
        
def loadDataFrame(filename):    
    #Load previously stored particle traces dataframe
    cols = list(pd.read_csv(cfg.FileAddressOut[0]+'/'+filename, nrows =1))
    print('Data series loaded from',filename,'dataframe: \n',cols[1:],'\n\n')
    
    particleTraces = pd.read_csv(cfg.FileAddressOut[0]+'/'+filename, usecols =[i for i in cols if i != 'Unnamed: 0'])
    
    return particleTraces


def exportTrace(particleTraces, traceNames, particleIndex = None, frameRate = None, save = False):
    
    for i_ in range(0,len(traceNames)):
        
        trace = dm.unflatten(particleTraces, traceNames[i_], particleIndex)
        
        if frameRate != None:
            traceTime = np.zeros([trace.shape[0],2])
            
            for k_ in range(0,len(trace)):
                
                traceTime[k_,0] = trace[k_]
                traceTime[k_,1] = k_*frameRate
                
            
        else:
            traceTime = trace
        
        if save == True:
            if particleIndex != None:
                
                filename = cfg.FileAddressOut[0]+'/Traces/'+traceNames[i_]+'_{0}'.format(particleIndex)
                np.savetxt(filename+'.csv', traceTime, delimiter=",")
                
            else:
                #export all particles
                np.savetxt(cfg.FileAddressOut[0]+'/Traces/'+traceNames[i_]+'.csv', traceTime, delimiter=",")
        else:
            #do not save trace(s) to file
            pass
        
        return trace
        
    
def readH5(fileIn, border = 0, minusFrames = 0):
    
    border = np.abs(border)
    
    if minusFrames == 0:
        if border == 0:
            with tables.File(cfg.FileAddressOut[0] + fileIn, 'r') as h5f:
                stack = h5f.root.data[:, :, :]
                size = h5f.root.data[:, :, :].shape 
        elif border != 0:
            with tables.File(cfg.FileAddressOut[0] + fileIn, 'r') as h5f:
                stack = h5f.root.data[:, border:-border, border:-border]
                size = h5f.root.data[:, border:-border, border:-border].shape         
        else:
            ValueError
        
    elif minusFrames != 0:
        
        minusFrames = -1*np.abs(minusFrames)
        
        if border == 0:
            with tables.File(cfg.FileAddressOut[0] + fileIn, 'r') as h5f:
                stack = h5f.root.data[0:minusFrames, :, :]
                size = h5f.root.data[0:minusFrames, :, :].shape 
        elif border != 0:
            with tables.File(cfg.FileAddressOut[0] + fileIn, 'r') as h5f:
                stack = h5f.root.data[0:minusFrames, border:-border, border:-border]
                size = h5f.root.data[0:minusFrames, border:-border, border:-border].shape         
        else:
            ValueError        
        
        
    return stack, size


