# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:17:21 2022

@author: ar2071
"""
#Configuration file containing global variables



###User Input Options:

#Number of frames to analyse - every Nth frame:
nFramesAnalysis = 20

#EChem Variables:
ActiveMass = 0.812 #Mass of acive electrode material in mg
LiX = 1 #Lithiums per transition metal, e.g. 1 for LCO, 17 for NWO, XX for NMC
MolarMass = 97.87 #Molar mass of electrode material
CatOrAn = True #True = cathode, False = Anode


###DEFAULT VALUES - USER CAN IGNORE:

#Filtering particles by minimum size:
pixelSizenm = 95 #pixel size (spatial resolution) in nanometres
PSF_4sigma = 1000 #500nm approximately for most set-ups
sizeThreshold = int((PSF_4sigma/pixelSizenm)**2)

#Default parameters - user can ignore:
eta = 10 #used in od.ObjectDetectionStack as a particle centre of mass matching condition
nanThreshold = 0.05 #remove particles from dataset with no. NaNs greater than this value

#Benchmarking mode sets size threshold particle filtering to 0
BENCHMARK_MODE = False
