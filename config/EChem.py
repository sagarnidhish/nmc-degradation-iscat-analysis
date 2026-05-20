# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:30:25 2021

@author: ajm314 - modified by ar2071 11/21
"""

'Read EChem from GAMRY PWR charge discharge output'
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import os
import config as cfg

#%%

# determine number of lines
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

#get data from OCV period before a cycle
def get_OCV_before(filenumber):
    for filename in os.listdir(cfg.EChemDirectory):
        if (filename.find(filenumber) >= 0) and (filename.find('OCV_before') >= 0):
            print(filename)

            # open file, read lines, close file
            f = open(cfg.EChemDirectory+filename, "r")
            lines = list(f)
            f.close()
            
            firstline = 48
            # crop lines to include only data entries last line is stop line
            data_string = lines[firstline:file_len(cfg.EChemDirectory+filename)]
            # parse each line and split at tabs, then truncate to data 
            stringlist = []
            for line in data_string:
                stringlist.append(line.split("\t")[1:5])
            #Convert to array
            data_array = np.array(stringlist, dtype=np.float32)
            
            index = data_array[:,0]
            time = data_array[:,1]
            voltage = data_array[:,2]
            
            return (index, time, voltage)

#Get data from OCv period after a cycle
def get_OCV_after(filenumber):
    for filename in os.listdir(cfg.EChemDirectory):
        if (filename.find(filenumber) >= 0) and (filename.find('OCV_after') >= 0):
#            print(filename)

            # open file, read lines, close file
            f = open(cfg.EChemDirectory+filename, "r")
            lines = list(f)
            f.close()
            
            firstline = 48
            # crop lines to include only data entries last line is stop line
            data_string = lines[firstline:file_len(cfg.EChemDirectory+filename)]
            # parse each line and split at tabs, then truncate to data 
            stringlist = []
            for line in data_string:
                stringlist.append(line.split("\t")[1:5])
            #Convert to array
            data_array = np.array(stringlist, dtype=np.float32)
            
            index = data_array[:,0]
            time = data_array[:,1]
            voltage = data_array[:,2]
            
            return (index, time, voltage)

#get all discharge data from inside a 'loop' folder        
def get_Discharge(filenumber):
    alldata = []
    for filename in os.listdir(cfg.EChemDirectory):
        if (filename.find(filenumber) == 0) and (filename.find('_Charge_Discharge') >= 0):
#            print(filename)
            
            for item in os.listdir(cfg.EChemDirectory+filename+"/CHARGE_DISCHARGE/"):
                
                if item.find('Discharge') >= 0:
                    print(item)

                    # open file, read lines, close file
                    f = open(cfg.EChemDirectory+filename+"/CHARGE_DISCHARGE/"+item, "r")
                    lines = list(f)
                    f.close()
                    
                    firstline = 56
                    # crop lines to include only data entries last line is stop line
                    data_string = lines[firstline:file_len(cfg.EChemDirectory+filename+"/CHARGE_DISCHARGE/"+item)-1]
                    
                    # parse each line and split at tabs, then truncate to data 
                    stringlist = []
                    for line in data_string:
                        stringlist.append(line.split("\t")[1:11])
                    #Convert to array
                    data_array = np.array(stringlist, dtype=np.float32)
                    
                    index = data_array[:,0]
                    time = data_array[:,1]
                    voltage = data_array[:,2]
                    current = data_array[:,3]*1000 #mA
                    
                    alldata.append((index, time, voltage, current))
    return alldata

#get all charge data from inside a 'loop' folder 
def get_Charge(filenumber):
    alldata = []
    for filename in os.listdir(cfg.EChemDirectory):
        if (filename.find(filenumber) == 0) and (filename.find('_Charge_Discharge') >= 0):
#            print(filename)
            
            for item in os.listdir(cfg.EChemDirectory+filename+"/CHARGE_DISCHARGE/"):
                
                if item.find('_charge') >= 0:
#                    print(item)

                    # open file, read lines, close file
                    f = open(cfg.EChemDirectory+filename+"/CHARGE_DISCHARGE/"+item, "r")
                    lines = list(f)
                    f.close()
                    
                    firstline = 57
                    # crop lines to include only data entries last line is stop line
                    data_string = lines[firstline:file_len(cfg.EChemDirectory+filename+"/CHARGE_DISCHARGE/"+item)-1]
                    
                    # parse each line and split at tabs, then truncate to data 
                    stringlist = []
                    for line in data_string:
                        stringlist.append(line.split("\t")[1:11])
                    #Convert to array
                    data_array = np.array(stringlist, dtype=np.float32)
                    
                    index = data_array[:,0]
                    time = data_array[:,1]
                    voltage = data_array[:,2]
                    current = data_array[:,3]*1000 #mA
                    
                    alldata.append((index, time, voltage, current))
    return alldata

#Get any OCV data that is from inside the loop
def get_OCV_middle(filenumber):
    alldata = []
    for filename in os.listdir(cfg.EChemDirectory):
        if (filename.find(filenumber) >= 0) and (filename.find('_Charge_Discharge') >= 0):
#            print(filename)
            for item in os.listdir(cfg.EChemDirectory+filename):
#                print(item)
                if item.find('OTHER')>= 0:
        
                    for item in os.listdir(cfg.EChemDirectory+filename+"/OTHER/"):
                        
                        if item.find('Discharge') < 0:
#                            print(item)
        
                            # open file, read lines, close file
                            f = open(cfg.EChemDirectory+filename+"/OTHER/"+item, "r")
                            lines = list(f)
                            f.close()
                            
                            firstline = 48
                            # crop lines to include only data entries last line is stop line
                            data_string = lines[firstline:file_len(cfg.EChemDirectory+filename+"/OTHER/"+item)-1]
                            
                            # parse each line and split at tabs, then truncate to data 
                            stringlist = []
                            for line in data_string:
                                stringlist.append(line.split("\t")[1:5])
                            #Convert to array
                            data_array = np.array(stringlist, dtype=np.float32)
                            
                            index = data_array[:,0]
                            time = data_array[:,1]
                            voltage = data_array[:,2]
                            
                            alldata.append((index, time, voltage))
    return alldata

#Stick together the collected parts of the data in (hopefully the correct order). (Uses 'if' cases to do this, not super orbust)
def append_sections_anode(filenumber):
    OCV_before_data = get_OCV_before(filenumber)
    OCV_after_data = get_OCV_after(filenumber)
    Discharge_data = get_Discharge(filenumber)
    Charge_data = get_Charge(filenumber)
    OCV_middle_data = get_OCV_middle(filenumber)
    
    times = []
    voltage = []
    current = []
    
    if len(OCV_middle_data) == 0:
        if len(Discharge_data) != len(Charge_data):
            print('Incorrect structure')
        else:
            #OCV_before data
            time = np.array(OCV_before_data[1])
            times = np.append(times,time)
            voltage = np.append(voltage,OCV_before_data[2])
            current = np.append(current,np.zeros(len(OCV_before_data[2])))
            carriedtime = time[-1]
            
            for cycle in np.arange(len(Discharge_data)):
                cycleindex = cycle 
                print(cycle)
                #Discharge step
                Discharge_data_step = Discharge_data[cycleindex]
                time = np.array(Discharge_data_step[1]) + carriedtime
                times = np.append(times,time)
                voltage = np.append(voltage,Discharge_data_step[2])
                current = np.append(current,Discharge_data_step[3])
                carriedtime = time[-1]
                #Charge step
                Charge_data_step = Charge_data[cycleindex]
                time = np.array(Charge_data_step[1]) + carriedtime
                times = np.append(times,time)
                voltage = np.append(voltage,Charge_data_step[2])
                current = np.append(current,Charge_data_step[3])
                carriedtime = time[-1]
            
            #OCV_after data
            time = np.array(OCV_after_data[1]) + carriedtime
            times = np.append(times,time)
            voltage = np.append(voltage,OCV_after_data[2])
            current = np.append(current,np.zeros(len(OCV_after_data[2])))
            carriedtime = time[-1]
            
    if len(OCV_middle_data) == 1:
        if (len(Discharge_data) != 1) or (len(Charge_data) != 1):
            print('Incorrect structure')
        carriedtime = 0
        stepindex = 0
        for step in [OCV_before_data,Discharge_data[0],OCV_middle_data[0],Charge_data[0],OCV_after_data]:
            time = np.array(step[1]) + carriedtime
            times = np.append(times,time)
            voltage = np.append(voltage,step[2])
            if (stepindex == 1) or (stepindex == 3):
                current = np.append(current,step[3])
            else:
                current = np.append(current,np.zeros(len(step[2])))
            carriedtime = time[-1]
            stepindex = stepindex +1

    if len(OCV_middle_data) == 3:
        if (len(Discharge_data) != 2) or (len(Charge_data) != 2):
            print('Incorrect structure')
        carriedtime = 0
        stepindex = 0
        for step in [OCV_before_data,Discharge_data[0],OCV_middle_data[0],Discharge_data[1],OCV_middle_data[1],Charge_data[0],OCV_middle_data[2],Charge_data[1],OCV_after_data]:
            time = np.array(step[1]) + carriedtime
            times = np.append(times,time)
            voltage = np.append(voltage,step[2])
            if (stepindex == 1) or (stepindex == 3) or (stepindex == 5) or (stepindex == 7):
                current = np.append(current,step[3])
            else:
                current = np.append(current,np.zeros(len(step[2])))
            carriedtime = time[-1]
            stepindex = stepindex +1
            
    return(times, voltage, current)

#Stick together the collected parts of the data in (hopefully the correct order). (Uses 'if' cases to do this, not super orbust)
def append_sections_cathode(filenumber):
    OCV_before_data = get_OCV_before(filenumber)
    OCV_after_data = get_OCV_after(filenumber)
    Discharge_data = get_Discharge(filenumber)
    Charge_data = get_Charge(filenumber)
    OCV_middle_data = get_OCV_middle(filenumber)
    
    times = []
    voltage = []
    current = []
    
    if len(OCV_middle_data) == 0:
        if len(Discharge_data) != len(Charge_data):
            print('Incorrect structure')
        else:
            #OCV_before data
            time = np.array(OCV_before_data[1])
            times = np.append(times,time)
            voltage = np.append(voltage,OCV_before_data[2])
            current = np.append(current,np.zeros(len(OCV_before_data[2])))
            carriedtime = time[-1]
            
            for cycle in np.arange(len(Discharge_data)):
                cycleindex = cycle 
                print(cycle)
                #Charge step
                Charge_data_step = Charge_data[cycleindex]
                time = np.array(Charge_data_step[1]) + carriedtime
                times = np.append(times,time)
                voltage = np.append(voltage,Charge_data_step[2])
                current = np.append(current,Charge_data_step[3])
                carriedtime = time[-1]
                #Discharge step
                Discharge_data_step = Discharge_data[cycleindex]
                time = np.array(Discharge_data_step[1]) + carriedtime
                times = np.append(times,time)
                voltage = np.append(voltage,Discharge_data_step[2])
                current = np.append(current,Discharge_data_step[3])
                carriedtime = time[-1]
            
            #OCV_after data
            time = np.array(OCV_after_data[1]) + carriedtime
            times = np.append(times,time)
            voltage = np.append(voltage,OCV_after_data[2])
            current = np.append(current,np.zeros(len(OCV_after_data[2])))
            carriedtime = time[-1]
            
    if len(OCV_middle_data) == 1:
        if (len(Discharge_data) != 1) or (len(Charge_data) != 1):
            print('Incorrect structure')
        carriedtime = 0
        stepindex = 0
        for step in [OCV_before_data,Discharge_data[0],OCV_middle_data[0],Charge_data[0],OCV_after_data]:
            time = np.array(step[1]) + carriedtime
            times = np.append(times,time)
            voltage = np.append(voltage,step[2])
            if (stepindex == 1) or (stepindex == 3):
                current = np.append(current,step[3])
            else:
                current = np.append(current,np.zeros(len(step[2])))
            carriedtime = time[-1]
            stepindex = stepindex +1

    if len(OCV_middle_data) == 3:
        if (len(Discharge_data) != 2) or (len(Charge_data) != 2):
            print('Incorrect structure')
        carriedtime = 0
        stepindex = 0
        for step in [OCV_before_data,Discharge_data[0],OCV_middle_data[0],Discharge_data[1],OCV_middle_data[1],Charge_data[0],OCV_middle_data[2],Charge_data[1],OCV_after_data]:
            time = np.array(step[1]) + carriedtime
            times = np.append(times,time)
            voltage = np.append(voltage,step[2])
            if (stepindex == 1) or (stepindex == 3) or (stepindex == 5) or (stepindex == 7):
                current = np.append(current,step[3])
            else:
                current = np.append(current,np.zeros(len(step[2])))
            carriedtime = time[-1]
            stepindex = stepindex +1
            
    return(times, voltage, current)

#Calculate capacity and delta-x from the current values for Nb14WO (17 TMs per unit cell). NOTE THIS WILL NEED EDITTING FOR NB16!!
def calc_capacity_NWO(filenumber):
    
    times, voltage, current = append_sections_anode(filenumber)
    
    capacity = integrate.cumtrapz(current, x = times,initial=0)
    capacity = -(capacity/3600)/(cfg.activemass/1000) #convert s to hours and mg to grams
    
    delta_x = (capacity*cfg.molarmass*3.6)/(1.602*10**-19 * 6.022*10**23)
    delta_x = delta_x/17 #define delta x as the number per transition metal
    
    return times ,capacity, delta_x

def calc_capacity_LCO(filenumber):
    
    times, voltage, current = append_sections_cathode(filenumber)
    
    capacity = integrate.cumtrapz(current, x = times,initial=0)
    capacity = -(capacity/3600)/(cfg.activemass/1000) #convert s to hours and mg to grams
    
    delta_x = (capacity*cfg.molarmass*3.6)/(1.602*10**-19 * 6.022*10**23)
    delta_x = delta_x/1 #define delta x as the number per transition metal
    
    return times ,capacity, delta_x


def CalcCapacityGeneralCathode(filenumber):
    
    times, voltage, current = append_sections_cathode(filenumber)
    
    capacity = integrate.cumtrapz(current, x = times,initial=0)
    capacity = -(capacity/3600)/(cfg.ActiveMass/1000) #convert s to hours and mg to grams
    
    delta_x = (capacity*cfg.MolarMass*3.6)/(1.602*10**-19 * 6.022*10**23)
    delta_x = delta_x/cfg.LiX #define delta x as the number per transition metal
    
    return times ,capacity, delta_x 

def CalcCapacityGeneralAnode(filenumber):
    
    times, voltage, current = append_sections_anode(filenumber)
    
    capacity = integrate.cumtrapz(current, x = times,initial=0)
    capacity = (capacity/3600)/(cfg.ActiveMass/1000) #convert s to hours and mg to grams
    
    delta_x = (capacity*cfg.MolarMass*3.6)/(1.602*10**-19 * 6.022*10**23)
    delta_x = delta_x/cfg.LiX #define delta x as the number per transition metal
    
    return times ,capacity, delta_x 

#delete any Gamry files with #2 in the name
def delete_02_files(fname): #note, fname needs '/' to move down paths
    folder = fname
    files = os.listdir(folder)
    
    for name in files:
        print(name)
        if name.find("#2") > 0:
            print('delete here')
            os.remove(fname+name)
            print(fname+name, ' deleted')

def EChemProcessing():
    
    #'set up plot style (optional)'
    # Edit the font, font size, and axes width
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['lines.linewidth'] = 0.8
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams['figure.figsize'] = 4, 3
    mpl.rc('image', cmap='gray')

    # generate paths to all files
    directories = [ name for name in os.listdir(cfg.EChemDirectory) if os.path.isdir(os.path.join(cfg.EChemDirectory, name)) ]
    allfilenames = []
    for name in directories:
        allfilenames.append(name + "/CHARGE_DISCHARGE/")
        
    print(cfg.EChemDirectory)
    print(allfilenames)
    
    #Make list of all file number
    filenumbers = []
    for filename in allfilenames:
        print(filename)
        print(filename[:2])
        filenumbers.append(filename[:2])
    print(filenumbers)
    for filenumber in filenumbers:
        print(filenumber)

        if cfg.CatOrAn == True:
            data = append_sections_cathode(filenumber)
        elif cfg.CatOrAn == False:
            data = append_sections_anode(filenumber)
        else:
            raise TypeError("CatOrAn must be assigned True or False")
            

        #Uncomment this to save the assembled echemdata files as npy files
        savedata = np.vstack((data[0],data[1],data[2]))
        print(np.shape(savedata))
        np.save(r'C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\EChem numpy56/'+filenumber+'_time-volt-curr.npy',savedata)
        cfg.time_volt_curr_directory = cfg.EChemDirectoryOut+filenumber+'_time-volt-curr.npy'
        np.save(cfg.time_volt_curr_directory, savedata)
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(data[0]/60,data[1],'k')
        ax1.set_ylabel("Voltage vs Li/Li$^{+}$ / V")
        ax2.plot(data[0]/60,data[2],'b',alpha=0.5)
        ax2.set_ylabel("Current / mA")   
        ax2.yaxis.label.set_color('b')
        ax1.set_xlabel('Time / min')
        plt.title(str(filenumber))
        plt.tight_layout()
        #plt.show()
        #plt.savefig(r'C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\EChem plot56/'+filenumber+'_VIvst.png')
        plt.savefig(cfg.EChemDirectoryOut+filenumber+'_VIvst.png')
    
    #GET TIMES, CAPCAITIES AND DELTA x FOR EACH FILENUMBER
    for filenumber in filenumbers:
        print(filenumber)
        
        
        
        if cfg.CatOrAn == True:
            data = append_sections_cathode(filenumber)
            times, cap, delta_x = CalcCapacityGeneralCathode(filenumber)
        elif cfg.CatOrAn == False:
            data = append_sections_anode(filenumber)
            times, cap, delta_x = CalcCapacityGeneralAnode(filenumber)
        else:
            raise TypeError("CatOrAn must be assigned True or False")
            

        #Uncomment this to save the assembled echemdata files as npy files
        savedata = np.vstack((times,cap,delta_x))
        print(np.shape(savedata))
        np.save(cfg.EChemDirectoryOut+filenumber+'_time-cap-xperTM.npy',savedata)
        
        plt.figure()
        plt.plot(times,delta_x)
        plt.show()
        
        plt.figure()
        plt.plot(delta_x,data[1],'k')
        plt.xlabel('$\Delta x_{Li}$ per TM')
        plt.ylabel("Voltage vs Li/Li$^{+}$ / V")
        plt.tight_layout()
        plt.savefig(cfg.EChemDirectoryOut+filenumber+'_VvsXLi.png')
        plt.show()
        
        plt.figure()
        plt.plot(cap,data[1],'k')
        plt.xlabel('$\Delta x_{Li}$ per TM')
        plt.ylabel("Voltage vs Li/Li$^{+}$ / V")
        plt.tight_layout()
        plt.savefig(cfg.EChemDirectoryOut+filenumber+'_TEST.png')
        plt.show()
        
        savedata = np.vstack((cap*-1,data[1]))
        np.save(r'C:\Users\ar2071\OneDrive - University of Cambridge\PhD\iSCAT Data\EChem numpy56/'+filenumber+'_Vvs_cap.npy',savedata)
        cfg.time_volt_curr_directory = cfg.EChemDirectoryOut+filenumber+'_Vvs_cap.npy'
        np.save(cfg.time_volt_curr_directory, savedata)
        
        fig, (ax1,ax2) = plt.subplots(2,1, figsize=(4,3.1))
        ax2a = ax2.twinx()
        ax1.plot(data[0]/60,data[1],'k')
        ax1.set_ylabel("Voltage vs Li/Li$^{+}$ / V")
        ax2a.plot(data[0]/60,data[2],'b',alpha=0.5)
        ax2a.set_ylabel("Current / mA")   
        ax2a.yaxis.label.set_color('b')
        ax1.set_xticklabels([])
        ax2.plot(times/60,delta_x,'k')
        ax2.set_ylabel("$\Delta x_{Li}$ per TM")
        ax2.set_xlabel('Time / min')
        plt.tight_layout()
        plt.savefig(cfg.EChemDirectoryOut+filenumber+'_Vxvst.png')
        plt.show()    
    
    return 



#%%
#USE THIS WITH CARE!! 
## Run this cell (uncommmented) to delete any files containing #2 in the name. 
## The intention is to delete all the empty '2nd loop' files, but it will not discriminate between empty files and real 2nd loop data files

#for filename in allfilenames:
#        
#    delete_02_files(EchemDirectory+filename)

