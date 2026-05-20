# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:53:41 2021

@author: ar2071


TESTING SCRIPT
"""
#%%
import sys
import os

sys.path
sys.path.append("C://Users/ar2071/OneDrive - University of Cambridge/PhD/Python/iSCAT Data Analysis/Unified Current")

# Imports
import DataManipulation as dm
import EChem as echem
import DataIO as dio
import multiprocessing_lib as multi 

import tifffile as tf
import tables
import multiprocessing as mp
from icecream import ic #Debugging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import time as t
from tqdm import trange
#%% Fine gaussian drift correct
import PIL
from PIL import Image

#define 2D gaussian shape
def twoD_Gaussian(xy_tuple, amplitude, xo, yo, sigma_x,sigma_y,theta,offset):# amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x,y) = xy_tuple   
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    G = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return G.ravel()


#Load stack
# ImageStack = np.array(Image.open('X:/NMC 3C - substack/sub_stack_1-1322-4.tif'))

#ImageStack,timesIntensity,frameRate = dio.RAWtoNPY(0)
ImageStack,_ = dio.readH5('/ImageStackStab.h5')
#%%
frameRate = 0.5 #ARB. PLACEHOLDER
xSize = ImageStack.shape[2]
ySize = ImageStack.shape[1]
totFrames = ImageStack.shape[0]

spot1 = ImageStack[0,705:730,1610:1630]
spot2 = ImageStack[1,705:730,1610:1630]
multi.areaSelection(spot1)

#%%
import matplotlib.pyplot as plt
import scipy.optimize

def MAKEtwoD_Gaussian(xy_tuple, amplitude, xo, yo, sigma_x,sigma_y,theta,offset):# amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x,y) = xy_tuple   
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    G = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return G




def gaussianStab(ImageStack):
    #Enter the x and y coordinates of the centre of your chosen blob here
    x_Jcent = int(1620)
    y_Jcent = int(717)
    #Enter the number of pixles to take on each side of the center when fitting to the blob
    Jhalfwidth = int(5)
    
    ImageJitter = ImageStack[:,y_Jcent-Jhalfwidth:y_Jcent+Jhalfwidth,x_Jcent-Jhalfwidth:x_Jcent+Jhalfwidth]
    
    x_pixels = np.arange(len(ImageJitter[0,0,:]))
    y_pixels = np.arange(len(ImageJitter[0,:,0]))
    x,y =  np.meshgrid(x_pixels, y_pixels)
    xy_tuple = (x,y)
    
    #This is an image of the chosen blob in the first frame. This is the image that a 2D gaussian will be fitted to in the first frame (it shouldnt contain any other features ideally)
    # plt.figure()
    # plt.imshow(ImageJitter[0,:,:])
    # plt.title('1st jit image')
    # plt.show()
    
    
    
    #Initial guesses for fitting parameters, to fit 2D gaussian to particle
    amplitude0 = 200
    xo0 = 5
    yo0 = 5
    sigma_x0 = 2
    sigma_y0 = 2
    theta0 = -0.4 
    #Background
    offset = 300 
    initial_guess = (amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0, offset)
    
    #Intialise lists to store x and y 'jitter' values for each frame            
    x_jitter = []
    y_jitter = []
    
    popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, xy_tuple, ImageJitter[0,:,:].ravel(), p0 = initial_guess,maxfev=10000)
    
    print('popt',popt.shape)
    print(popt)
    print('pcov',pcov.shape)
    print(pcov)
    
    gauss = MAKEtwoD_Gaussian(xy_tuple, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
    
    plt.figure()
    plt.imshow(gauss)
    plt.show()
    
    #pyplot.figure()
    for frame in np.arange(len(ImageStack[:,0,0])):
        print('Fitting frame ', frame)
    #    print(x_Jcent,y_Jcent, 'x_Jcent,y_Jcent')
        ImageJitter = ImageStack[:,y_Jcent-Jhalfwidth:y_Jcent+Jhalfwidth,x_Jcent-Jhalfwidth:x_Jcent+Jhalfwidth]
    #    print(numpy.shape(ImageJitter))

                
        popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, xy_tuple, ImageJitter[frame,:,:].ravel(), p0 = initial_guess,maxfev=10000)
        
        
        x_change = popt[1]-Jhalfwidth
        y_change = popt[2]-Jhalfwidth
        
        x_poptpos = x_Jcent - Jhalfwidth + popt[1]
        y_poptpos = y_Jcent - Jhalfwidth + popt[2]
        x_jitter.append(x_poptpos)
        y_jitter.append(y_poptpos)
    #    print(x_poptpos,y_poptpos, ' jitter')
        
    #    pyplot.figure()
    #    pyplot.imshow(ImageJitter[frame,:,:])
    #    pyplot.plot(popt[1],popt[2],'ko')
    #    pyplot.show()
        
    #    print(popt[1],popt[2])
        x_Jcent = x_Jcent + int(np.rint(x_change))
        y_Jcent = y_Jcent + int(np.rint(y_change))
    #    print(x_Jcent,y_Jcent)
    #    print(x_change,y_change, ' changes')
    #    print(int(numpy.rint(x_change)),int(numpy.rint(y_change)), ' changes')
        
    #    pyplot.imshow(ImageJitter[frame,:,:])
    #    pyplot.show()
    
    
    #Check that the centre point of the 2D gaussian is still being fitted correctly in a chosen frame
    # plt.figure()
    # plt.imshow(ImageStack[200,:,:],vmax=800)
    # plt.plot(x_jitter[200],y_jitter[500],'ro')
    # plt.show()
    
    #Show the pixel values of the centre point across all frames
    plt.figure(figsize=(6,4))
    plt.plot(times,x_jitter,label='x direction',linewidth=1)
    plt.plot(times,y_jitter,label='y direction',linewidth=1)
    #pyplot.title(filestring)
    plt.xlabel('Time / s')
    plt.ylabel('Translation of image / pix')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #Subtract the initial pixel values of the centrepoint to get just the amount that the image moves by
    x_motion = x_jitter-x_jitter[0]
    y_motion = y_jitter-y_jitter[0]
    
    trajectory = [x_motion, y_motion]
    
    #Show the movemnet of the centre point across all frames
    plt.figure(figsize=(6,4))
    plt.plot(times,x_motion,label='x direction',linewidth=1)
    plt.plot(times,y_motion,label='y direction',linewidth=1)
    #pyplot.title(filestring)
    plt.xlabel('Time / s')
    plt.ylabel('Translation of image / pix')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return trajectory

traj = gaussianStab(ImageStack)



#%% FIXING BIGTIFF IMPORT
import numpy as np
import PIL
from PIL import Image

#img = Image.open(cfg.OpticalDataInPath[0])
img = Image.open('X:/NMC 3C - substack/sub_stack_1-1322-4.tif')
arr = np.array(img)
#%% CV2 VERSION?
import cv2

stack = cv2.imreadmulti('X:/NMC 3C - substack/sub_stack_1-1322-4.tif', flags = cv2.IMREAD_ANYCOLOR)
#%%
frames = stack[1]
frame = frames[0]
#%%multipagetiff package
import multipagetiff as mpt

stack = mpt.read_stack(cfg.OpticalDataInPath[0])


#%% Particle heterogeneity - traces
nStd = 1
traceName = 'IntensitiesNormFilt'


#Calculate mean trace and standard deviation of entire dataset
mTrace, stdvec = dm.meanTrace(particleTraces, traceName)


#Iterate over every particle:
allTraces = dm.unflatten(particleTraces, traceName)
dist = np.zeros(allTraces.shape)

for i in range(0, allTraces.shape[0]):
    
    #Unflatten a single particle trace
    traceI = dm.unflatten(particleTraces, traceName, particleIndex = i)
    
    #Find distance of select trace from mean (as a multiple of std. dev.) per time point
    dist[i,:] = (traceI-mTrace)/stdvec


plt.figure()
plt.plot(mTrace+nStd*stdvec, linestyle = '--', color = 'b')
plt.plot(mTrace-nStd*stdvec, linestyle = '--', color = 'b')
plt.plot(mTrace, color = 'k')
plt.plot(dm.unflatten(particleTraces, traceName, particleIndex=2))
plt.show()

#Plot how many standard deviations each trace is from the mean
plt.figure()
for i in range(0,dist.shape[0]):
    plt.plot(dist[i,:])
plt.show()

#%% Visualising heterogeneity on the image stack
import matplotlib.cm as cm
import matplotlib 
#Find max and min "distance" from mean of any given trace

if np.abs(np.amax(dist)) >= np.abs(np.amin(dist)):
    
    maxDev = np.abs(np.amax(dist))
    
elif np.abs(np.amin(dist)) >= np.abs(np.amax(dist)):

    maxDev = np.abs(np.amin(dist))
else:
    print('error')

dist = dist/maxDev #normalised distance of each trace at every time point from the mean, in terms of the std. dev.


#%%
'''
THERE IS A MISMATCH OF INDICES BETWEEN THE MAXOBJECTS (135 PARTICLES) AND ANY FRAME WHICH DOES NOT HAVE THE MAX NO. OBJECTS.
'''

def spatialStdDevStack(particleTraces, maxObjects, traceName = 'IntensitiesNormFilt', fileIn = '/segMapStack.npy'):

    #Calculate mean trace and standard deviation of entire dataset
    mTrace, stdvec = dm.meanTrace(particleTraces, traceName)
    
    
    #Iterate over every particle:
    allTraces = dm.unflatten(particleTraces, traceName)
    dist = np.zeros(allTraces.shape)
    
    for i in range(0, allTraces.shape[0]):
        
        #Unflatten a single particle trace
        traceI = dm.unflatten(particleTraces, traceName, particleIndex = i)
        
        #Find distance of select trace from mean (as a multiple of std. dev.) per time point
        dist[i,:] = (traceI-mTrace)/stdvec
    
    

    
    #Visualising heterogeneity on the image stack
    import matplotlib.cm as cm
    import matplotlib 
    #Find max and min "distance" from mean of any given trace
    
    if np.abs(np.amax(dist)) >= np.abs(np.amin(dist)):
        
        maxDev = np.abs(np.amax(dist))
        
    elif np.abs(np.amin(dist)) >= np.abs(np.amax(dist)):
    
        maxDev = np.abs(np.amin(dist))
    else:
        print('error')
    
    dist = dist/maxDev #normalised distance of each trace at every time point from the mean, in terms of the std. dev.
    print(dist.shape)

    #Apply std. dev. data to frames:
    segMapStack = np.load(cfg.FileAddressOut[0]+fileIn)
    
    '''
    DATA SEGMENTATION, AS USED IN OTHER STACK GENERATION FUNCTIONS, REQUIRES THE INPUT FILE TO BE HDF5!!
    '''
    

    for i in range(0,segMapStack.shape[0]):
        
        #Temp. frame holding std. dev. values per particle to multiply with binary map
        tempFrame = np.ones([segMapStack.shape[1],segMapStack.shape[2]])
        
        #Size ordered particles (large to small) minimise overlap when multiplying std. dev. values
        sizeOrderedIndices = maxObjects.sort_values('npix', ascending = False).index
        
        for j in range(0,maxObjects.shape[0]):
            
            xcoords = [maxObjects['xmin'][sizeOrderedIndices[j]],maxObjects['xmax'][sizeOrderedIndices[j]]]
            ycoords = [maxObjects['ymin'][sizeOrderedIndices[j]],maxObjects['ymax'][sizeOrderedIndices[j]]]
    
            tempFrame[xcoords[0]:xcoords[1], ycoords[0]:ycoords[1]] = dist[sizeOrderedIndices[j],i]
            
        
        segMapStack[i,:,:] = segMapStack[i,:,:]*tempFrame
            
    return segMapStack
            
    
    
SDsegMapStack = spatialStdDevStack(particleTraces, maxObjects)


#%%
from matplotlib import rcParams
#Setting default figure formatting for papers etc.
rcParams['font.family'] = 'arial'
rcParams['font.size'] = '8'
rcParams['figure.figsize'] = (3.15*2, 1.5*1.94) 
rcParams['lines.linewidth'] = 1.2
    


fig, ax = plt.subplots()

twin1 = ax.twinx()


p1, = ax.plot(echem_raw[:,0],echem_raw[:,1], "b-", label="Voltage",alpha=0.5)
p2, = twin1.plot(echem_raw[:,0],echem_raw[:,2]*1e3, "r-", label="Current", alpha =0.5)


ax.set_xlabel("Time / hours")
ax.set_ylabel("Voltage / V")
twin1.set_ylabel("Current / mA")

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())


tkw = dict(size=3, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2])
plt.savefig(cfg.FileAddressOut[0]+'/Plots'+'/echem' + '.png', dpi = 600, facecolor = 'w', edgecolor = 'w', transparent = False, bbox_inches = 'tight')
plt.savefig(cfg.FileAddressOut[0]+'/Plots/'+ 'echem' + '.svg')

plt.show()

#SOURCE: https://matplotlib.org/3.4.3/gallery/ticks_and_spines/multiple_yaxis_with_spines.html

#%%size histograms of min and max sizes during cycle
import matplotlib.pyplot as plt
from matplotlib import rcParams

#Setting default figure formatting for papers etc.
rcParams['font.family'] = 'arial'
rcParams['font.size'] = '8'
rcParams['figure.figsize'] = (3.15*1.5, 1.5*1.94) 
rcParams['lines.linewidth'] = 1.2
    
    
#absolute sizes in pixels:
sizeArr = dm.unflatten(particleTraces, ['SizesPixFilt'])

sizeMax = np.zeros(sizeArr.shape[0])
sizeMin = np.zeros(sizeArr.shape[0])
print(sizeArr.shape)
print(sizeMax.shape)
for i in range(0,len(sizeMax)):
    sizeMax[i] = np.amax(sizeArr[i,:])*1e-6*(cfg.pixelSizenm)**2
    sizeMin[i] = np.amin(sizeArr[i,:])*1e-6*(cfg.pixelSizenm)**2
    
plt.figure()

plt.hist(sizeMax, bins = 100, alpha=0.5, cumulative = True, label='Max. particle size', color = ['#ff0000'])
plt.hist(sizeMin, bins = 100, alpha=0.5,cumulative = True, label='Min. particle size', color = ['#007FFF'])

plt.xlabel('Active Particle Size /'r'$\mu m^2$')
plt.ylabel('Counts')

plt.legend(loc='lower right')
plt.savefig(cfg.FileAddressOut[0]+'/Plots'+'/sizeHistogram' + '.png', dpi = 600, facecolor = 'w', edgecolor = 'w', transparent = False, bbox_inches = 'tight')

plt.show()
    
#%%size vs intensity scatter plot

import matplotlib.pyplot as plt
from matplotlib import rcParams

#Setting default figure formatting for papers etc.
rcParams['font.family'] = 'arial'
rcParams['font.size'] = '8'
rcParams['figure.figsize'] = (3.15*1.5, 1.5*1.94) 
rcParams['lines.linewidth'] = 1.2
    
    
#absolute sizes in pixels:
sizeArr = dm.unflatten(particleTraces, ['SizesPixFilt'])

sizeMax = np.zeros(sizeArr.shape[0])
sizeMin = np.zeros(sizeArr.shape[0])


for i in range(0,len(sizeMax)):
    sizeMax[i] = np.amax(sizeArr[i,:])*1e-6*(cfg.pixelSizenm)**2
    sizeMin[i] = np.amin(sizeArr[i,:])*1e-6*(cfg.pixelSizenm)**2
    
    
intArr = dm.unflatten(particleTraces, ['IntensitiesNormFilt'])

intMin = np.zeros(intArr.shape[0])
intMax = np.zeros(intArr.shape[0])

for i in range(0, len(intMax)):
    intMax[i] = np.amax(intArr[i,:])
    intMin[i] = intArr[i,0]


a,b = np.polyfit(sizeMax, intMax, 1)

plt.figure()

plt.scatter(sizeMax, intMax, label='Max. Intensity, Max. Size', color = ['#ff0000'], alpha=0.5)
plt.scatter(sizeMin, intMin, label='Min. Intensity, Min. Size', color = ['#007FFF'],alpha=0.5)
plt.plot(sizeMax,a*sizeMax+b, color ='#000000')

plt.xlabel('Active Particle Size /'r'$\mu m^2$')
plt.ylabel('Normalised Intensity /arb. units')

plt.legend()
plt.savefig(cfg.FileAddressOut[0]+'/Plots'+'/IntensityVsSize' + '.png', dpi = 600, facecolor = 'w', edgecolor = 'w', transparent = False, bbox_inches = 'tight')

plt.show()


#%%
from scipy import signal

arr1 = np.zeros([100,100])
arr2 = np.zeros([10,10])

arr1[50,50] = 1
arr2[5,5] = 1.34

corr = signal.correlate2d(arr1,arr2)

#%%
import numpy as np
from scipy.signal import fftconvolve


def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out

#read in raw frames:
import DataIO as dio

ImageStackFrames, xSize, ySize, filetype = dio.readFileDims(r'C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\05_LCo-C22-1cyc-2C-blue_0\cam1/event0.tdms')
ImageStack, _ = dio.RAWtoNPY(r'C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\05_LCo-C22-1cyc-2C-blue_0\cam1/event0.tdms', ImageStackFrames, 0)

out = np.zeros([ImageStackFrames-1, 79,39])

for i in range(1,ImageStackFrames):
    out[i-1] = normxcorr2(ImageStack[i, 180:220,30:50],ImageStack[i-1,180:220,30:50])
#%%
import napari
viewer = napari.view_image(out)
napari.run()
    
#%%
import napari
ImageStack = np.load(r"C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\NMC 3C\Output/noBkgstack.h5")
viewer = napari.view_image(ImageStack)
napari.run()

#%%calculating the centre of a 2D intensity distribution radially - translated from matlab reference: https://www.nature.com/articles/nmeth.2071#Sec11

def radialcentre(image):
    
    dims = np.shape(image)
    Nx = dims[1]
    Ny = dims[2]
    
    xm_onerow = np.linspace(-(Nx-1)/2.0+0.5,(Nx-1)/2.0-0.5, Nx*2+1)
    xm = xm_onerow(ones(Ny-1, 1), :);
    
    return xc, yc, sigma


#%%


def show_entry_fields():
    print("Active mass: %s\nLithiums per TM: %s\nMolar mass: %s\nCathode or Anode: %s" % (e1.get(), e2.get(), e3.get(), e4.get()))
    activemass = e1.get() 
    
master = tk.Tk()
master.title("Enter echem parameters")
master.geometry("300x120")

tk.Label(master, text="Active mass").grid(row=0)
tk.Label(master, text="Lithiums per TM").grid(row=1)
tk.Label(master, text="Molar mass").grid(row=2)
tk.Label(master, text="Cathode or anode").grid(row=3)

e1 = tk.Entry(master)
e2 = tk.Entry(master)
e3 = tk.Entry(master)
e4 = tk.Entry(master)


e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)


#tk.Button(master, text='Quit', command=master.quit).grid(row=4, column=0, sticky=tk.W, pady=4)

tk.Button(master, text='Show', command=show_entry_fields).grid(row=4, column=1, sticky=tk.W, pady=4)

                                                     


tk.mainloop()

#%%more "pythonic" version of above

fields = 'Active mass', 'Lithiums per tm', 'Molar mass', 'Cathode or anode'

def fetch(entries):
    for entry in entries:
        field = entry[0]
        text  = entry[1].get()
        print('%s: "%s"' % (field, text)) 

def makeform(root, fields):
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=15, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries


root = tk.Tk()
ents = makeform(root, fields)
root.bind('<Return>', (lambda event, e=ents: fetch(e)))   

b1 = tk.Button(root, text='Show', command=(lambda e=ents: fetch(e)))
b1.pack(side=tk.LEFT, padx=5, pady=5)

b2 = tk.Button(root, text='Quit', command=root.quit)
b2.pack(side=tk.LEFT, padx=5, pady=5)


root.mainloop()
#%%
import tkinter as tk
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 300,  relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='Calculate the Square Root')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)

label2 = tk.Label(root, text='Type your Number:')
label2.config(font=('helvetica', 10))
canvas1.create_window(200, 100, window=label2)

entry1 = tk.Entry (root) 
canvas1.create_window(200, 140, window=entry1)

def getSquareRoot():
    
    x1 = entry1.get()
    
    label3 = tk.Label(root, text= 'The Square Root of ' + x1 + ' is:',font=('helvetica', 10))
    canvas1.create_window(200, 210, window=label3)
    
    label4 = tk.Label(root, text= float(x1)**0.5,font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 230, window=label4)
    
button1 = tk.Button(text='Get the Square Root', command=getSquareRoot, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
canvas1.create_window(200, 180, window=button1)

root.mainloop()
#%%add entry fields using button within tkinter
import tkinter as tk
from tkinter import Tk
#------------------------------------

def addBox():
    print('ADD')

    ent = Entry(root)
    ent.pack()

    all_entries.append( ent )

#------------------------------------

def showEntries():

    for number, ent in enumerate(all_entries):
        print(number, ent.get())

#------------------------------------

all_entries = []
    
root = Tk()
    
showButton = Button(root, text='Show all text', command=showEntries)
showButton.pack()
    
addboxButton = Button(root, text='<Add Time Input>', fg="Red", command=addBox)
addboxButton.pack()
    
root.mainloop()


#%%
dio.batchPathSetup()
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tables
filename = 'ImageStackStab.h5'


# The parametrized function to be plotted
# def f(t, amplitude, frequency):
#     return amplitude * np.sin(2 * np.pi * frequency * t)

def f_alt(FileAddressOut,filename, pIndex, fIndex):
    
    pIndex = int(pIndex)
    fIndex = int(fIndex)
    
    with tables.File(FileAddressOut+"/"+filename, 'r') as h5f:
    
        frame = h5f.root.data[fIndex, maxObjects['y'][pIndex]-padding*maxObjects['a'][pIndex]:maxObjects['y'][pIndex]+padding*maxObjects['a'][pIndex], maxObjects['x'][pIndex]-padding*maxObjects['a'][pIndex]:maxObjects['x'][pIndex]+padding*maxObjects['a'][pIndex]]

    
    return frame

t = np.linspace(0, 1, 1000)

# Define initial parameters
# init_amplitude = 5
# init_frequency = 3

init_pIndex = 1
init_fIndex = 0

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line = plt.pcolormesh(f_alt(FileAddressOut, filename, init_pIndex, init_fIndex))
#ax.set_xlabel('Time [s]')

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axP = plt.axes([0.25, 0.1, 0.65, 0.03])
particle_slider = Slider(
    ax=axP,
    label='particle index',
    valmin=0,
    valmax=particleIntensities.shape[0],
    valinit=init_pIndex,
)

# Make a vertically oriented slider to control the amplitude
axF = plt.axes([0.1, 0.25, 0.0225, 0.63])
frame_slider = Slider(
    ax=axF,
    label="frame index",
    valmin=0,
    valmax=330,
    valinit=init_fIndex,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    line = plt.pcolormesh(f_alt(FileAddressOut, filename, particle_slider.val, frame_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
particle_slider.on_changed(update)
frame_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
#resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
#button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    particle_slider.reset()
    frame_slider.reset()
#button.on_clicked(reset)

plt.show()
    
    
#%%bokkeh line plot
from bokeh.plotting import figure, output_file, show

output_file("line.html")

p = figure(width=1200, height=800)
x = np.array(np.linspace(0,306,307))

# add a line renderer

test = dm.unflatten(particleTraces, 'FluxesNorm')

for i_ in range(0,len(test)):
    p.line(x, test[i_], line_width=2)

show(p)


#%%Bokked plots showing 1. frame from stack, 2. selected trace, 3. selected trace - output/save as .html

from bokeh.plotting import figure, show
from bokeh.layouts import row
from bokeh.palettes import inferno# select a palette
import itertools# itertools handles the cycling 
from bokeh.models import PolySelectTool
from bokeh.models import WheelZoomTool
#p1: image plot
p1 = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")], title = "iSCAT image - click on a particle to select it", toolbar_location = "below")

p1.x_range.range_padding = p1.y_range.range_padding = 0

# must give a vector of image data for image parameter
p1.image(image=[segMapStack[0,:,:]], x=0, y=0, dw=segMapStack.shape[2], dh=segMapStack.shape[1], palette="Inferno256", level="image")
p1.grid.grid_line_width = 0.5
p1.add_tools(PolySelectTool())

#p2: stacked line plot - raw intensities
p2 = figure(width=700, height=500, toolbar_location = "below")
x = np.array(np.linspace(0,306,307))

#generate colour palette
#colors = itertools.cycle(inferno(len(particleIntensities)))# create a color iterator 

#Line renderer
for i_ in range(0,len(test)):
    p2.line(x, test[i_], line_width=2)


#p3: stacked line plot - normalised intensities (1 to inf)
p3 = figure(width=700, height=500, toolbar_location = "below")
    
test1 = dm.unflatten(particleTraces, 'FluxesNormFilt')

#Line renderer
for i_ in range(0,len(test1)):
    p3.line(x, test1[i_], line_width=2)
    #p3.line(x, particleIntensitiesNorm[i_,:], line_width=2, color=next(colors))
show(row(p1,p2,p3))

#%% WORKING SURFACE PLOT OF SINGLE FRAME
import scipy.signal

with tables.File(FileAddressOut+"/"+"noBkgStack.h5", 'r') as h5f:
    
    data = h5f.root.data[682,:,:]
    #size = h5f.root.data.shape

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy
# downscaling has a "smoothing" effect
lena = data

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.gray,
        linewidth=0)

# show it
plt.show()

#%%no background diff stack
dm.noBkgStack(FileAddressOut, segMapStack, fileIn = '/diffStack.h5')
#%%Shrinking cores
import Plotting as p
p.stackViewer('ImageStackStab.h5', FileAddressOut, maxObjects, 50)
#%%
#Bin the differential pixel intensities along x and y axes:
#arbitrary multiplier to pad the frame
particleIndex = 1
xmin = maxObjects['xmin'][particleIndex]*0.9
xmax = maxObjects['xmax'][particleIndex]*1.1
        
ymin = maxObjects['ymin'][particleIndex]*0.9
ymax = maxObjects['ymax'][particleIndex]*1.1
    
with tables.File(FileAddressOut+"/"+'noBkgStack.h5', 'r') as h5f:
    frame = h5f.root.data[:, ymin:ymax, xmin:xmax]
    size = h5f.root.data[:, ymin:ymax, xmin:xmax].shape         

xhist = np.zeros([size[0] , size[1]])
yhist = np.zeros([size[0], size[2]])

xstd = np.zeros([size[0]])
ystd = np.zeros([size[0]])

for i_ in range(0,size[0]):
    for j_ in range(0,size[1]): #iterate over x   

        xhist[i_,j_] = np.sum(frame[i_,j_,:])/np.count_nonzero(frame[i_,j_,:])

    for k_ in range(0,size[2]): #iterate over y

        yhist[i_,k_] = np.sum(frame[i_,:,k_])/np.count_nonzero(frame[i_,:,k_])
        
    #Calculate standard deviations in each axis of pixel intensities per frame:
    xstd[i_] = np.nanstd(xhist[i_,:])
    ystd[i_] = np.nanstd(yhist[i_,:])

#remove NaN values that result from empty image frames (only ever at the start and end of image stack)
xstd = xstd[~np.isnan(xstd)]
ystd = ystd[~np.isnan(ystd)]

#%%
from scipy.signal import savgol_filter
xstd[:] = savgol_filter(xstd[:], window_length = 91, polyorder = 2, deriv = 1)
ystd[:] = savgol_filter(ystd[:], window_length = 91, polyorder = 2, deriv = 1)
#%%
xstd[:] = savgol_filter(xstd[:], window_length = 17, polyorder = 2)
ystd[:] = savgol_filter(ystd[:], window_length = 17, polyorder = 2)

#%%
# xfft = np.fft.fft(xstd[:])
# yfft = np.fft.fft(ystd[:])

# plt.figure()
# plt.plot(xfft)
# plt.plot(yfft)
# plt.show()

#%%
import numpy as np
from scipy.signal import butter,filtfilt
# Filter requirements.
T = 1         # Sample Period
fs = 5588       # sample rate, Hz
cutoff = fs/120      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Filter the data, and plot both the original and filtered signals.
xbutter = butter_lowpass_filter(xstd, cutoff, fs, order)
ybutter = butter_lowpass_filter(ystd, cutoff, fs, order)

plt.figure()
plt.plot(xbutter)
plt.plot(ybutter)
fig.show()


#extract first local min/max - fit gaussian in this range to calibrate accuracy vs. noise
xmin = scipy.signal.argrelextrema(xbutter,np.less)[0][0]
xmax = scipy.signal.argrelextrema(xbutter,np.greater)[0][0]

ymin = scipy.signal.argrelextrema(ybutter,np.less)[0][0]
ymax = scipy.signal.argrelextrema(ybutter,np.greater)[0][0]


#%%
plt.figure()
plt.plot(xbutter, label = 'X')
plt.plot(ybutter, label = 'Y')
plt.legend()
plt.ylabel('Standard Deviation per axis per frame')
plt.xlabel('Frame Number')
plt.show()

#%%polynomial regression


x = np.linspace(0,5587,5588)
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()


#%%
import seaborn as sns
fig, axes = plt.subplots(1,2)

sns.heatmap(ax = axes[0], data = xhist.T)
sns.heatmap(ax = axes[1], data = yhist.T)

#%%SIMULTANEOUS INTERACTIVE PLOTTING IN NAPARI######
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
import Plotting as p
import napari

# create image
x = np.linspace(0, 5, 256)
y = np.linspace(0, 5, 256)[:, np.newaxis]
img = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

# add it to the viewer
viewer = napari.view_image(img, colormap='viridis')
layer = viewer.layers[-1]

# create mpl figure with subplots
mpl_fig = plt.figure()

ax = mpl_fig.add_subplot(111)
(line,) = ax.plot(layer.data[123])  # linescan through the middle of the image


# add the figure to the viewer as a FigureCanvas widget
viewer.window.add_dock_widget(FigureCanvas(mpl_fig))


# connect a callback that updates the line plot when
# the user clicks on the image
@layer.mouse_drag_callbacks.append
def profile_lines_drag(layer, event):
    try:
        line.set_ydata(layer.data[int(event.position[0])])
        print(event.position)
        line.figure.canvas.draw()
    except IndexError:
        pass


napari.run()

#%%

#unflatten traces
trace = dm.unflatten(particleTraces, ['IntensitiesNormFilt'])

#all plotting
for j in range(len(trace)):
    plt.plot(trace[j], alpha = 0.3)
mTrace, stdvec = dm.meanTrace(particleTraces, ['IntensitiesNormFilt'])
plt.plot(mTrace, color='k', lw=2.5)
plt.plot(mTrace+1*stdvec, '--k', lw=1.5)
plt.plot(mTrace-1*stdvec, '--k', lw=1.5)

img = stack[150,:,:]
viewer = napari.view_image(img, colormap='viridis')
layer = viewer.layers[-1]

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)

viewer.window.add_dock_widget(FigureCanvas(mpl_fig))

# connect a callback that updates the line plot when
# the user clicks on the image
@layer.mouse_drag_callbacks.append
def profile_lines_drag(layer, event):
    try:
        line.set_ydata()
        print(event.position)
        #print(layer.data[123])
        line.figure.canvas.draw()
    except IndexError:
        pass


napari.run()
#%%Load capacity(t)

cap05 = np.load(r'C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\echem_test/05_Vvs_cap.npy')

plt.figure()
plt.plot(cap05[0],cap05[1])
plt.show()
print(len(cap05[0])/12000)
np.savetxt(r'C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\echem_test/05_VvsCap.csv', cap05[0] ,delimiter = ',')

#%% big tiff

import libtiff

#tif = TIFFfile(r'C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\NMC 3C/Full_stack_NMCa7_1C.tif')
tif = libtiff.TIFF.open(r'C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\NMC 3C/Full_stack_NMCa7_1C.tif')
width = tif.GetField("ImageWidth")
height = tif.GetField("ImageLength")
bits = tif.GetField('BitsPerSample')
sample_format = tif.GetField('SampleFormat')

#%%config test
import config as cfg

#%%
a, b = dio.echemPathSetup()
#%%
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import config as cfg


def select_files():
    filetypes = (('All files', '*.*'), ('Text files', '.txt'))

    cfg.filenames = fd.askopenfilenames(
        title='Open files',
        initialdir='/',
        filetypes=filetypes)

    showinfo(
        title='Selected Files',
        message=cfg.filenames
    )
    print(cfg.filenames)



#%%
def window():
    root = tk.Tk()
    root.title('Tkinter File Dialog')
    root.resizable(False, False)
    root.geometry('300x150')
    root.attributes('-topmost', True)
    
    open_button = ttk.Button(
        root,
        text='Open Files',
        command=select_files
    )
    
    open_button.pack(expand=True)
    
    
    root.mainloop()
    
window()
#%%
import os
dummy = np.ones([10000,10000])

vec = np.zeros([10000,10000])
cev = np.zeros(vec.shape)

for i in range(0,10000):
    vec[i,i] += i    
    cev[9999-i,9999-i] += i
os.remove(r'X:/rw.h5')

#%%
import tables
import h5py
import numpy as np

with h5py.File(r'X:/rw.h5', mode = 'a') as fOut:
            #defines the datatype as float32 to save space in stabilised image stack
            # atom = tables.Float32Atom()
 
            # tempArray = fOut.create_earray(fOut.root, 'vec', atom, (0,100,100)) #creates extendable array (earray) containing datatype float64
            # tempArray1 = fOut.create_earray(fOut.root, 'cev', atom, (0,100,100))
            
            fOut.create_dataset('vec',data = vec)
            fOut.create_dataset('cev',data = cev)
            
            

            # try:
            #     tempArray.append(vec)#stabilise and interpolate
            #     tempArray1.append(cev)
            # except:
            #     print('error')
            #     fOut.close()
#%%
            
with h5py.File(r'X:/rw.h5', mode = 'w') as fIn:
    del fIn['vec'][:,:]
    b[0,0] = fIn['cev'][0,0]+99 #selective write
    

    
    
    
    
#%%
stack = np.load(r'X:\LCO Red-Blue\05_LCo-C22-1cyc-2C-blue_0\cam1/SynthStackPerturbed.npy')
import multiprocessing_lib as m

arr = m.fineStab(0, stack)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(arr)
plt.show()
#%%
plt.figure()
plt.plot(trajectory[:,0])
plt.plot(trajectory[:,1])
plt.show()
                