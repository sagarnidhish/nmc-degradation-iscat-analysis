# radialcenter.m
#
# Copyright 2011-2012, Raghuveer Parthasarathy, The University of Oregon
#
##
# Disclaimer / License
#   This program is free software: you can redistribute it and/or
#     modify it under the terms of the GNU General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#   This set of programs is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   (gpl.txt) along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
#
# Calculates the center of a 2D intensity distribution.
# Method: Considers lines passing through each half-pixel point with slope
# parallel to the gradient of the intensity at that point.  Considers the
# distance of closest approach between these lines and the coordinate
# origin, and determines (analytically) the origin that minimizes the
# weighted sum of these distances-squared.
#
# Inputs
#   I  : 2D intensity distribution (i.e. a grayscale image)
#        Size need not be an odd number of pixels along each dimension
#
# Outputs
#   xc, yc : the center of radial symmetry,
#            px, from px #1 = left/topmost pixel
#            So a shape centered in the middle of a 2*N+1 x 2*N+1
#            square (e.g. from make2Dgaussian.m with x0=y0=0) will return
#            a center value at x0=y0=N+1.
#            Note that y increases with increasing row number (i.e. "downward")
#   sigma  : Rough measure of the width of the distribution (sqrt. of the
#            second moment of I - min(I));
#            Not determined by the fit -- output mainly for consistency of
#            formatting compared to my other fitting functions, and to get
#            an estimate of the particle "width"
#
# Raghuveer Parthasarathy
# The University of Oregon
# August 21, 2011 (begun)
# last modified Apr. 6, 2012 (minor change)
# Copyright 2011-2012, Raghuveer Parthasarathy

import pdb
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
#Generate test gaussian image
def makeGaussian(size, fwhm = 3, centre=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if centre is None:
        x0 = y0 = size // 2
    else:
        x0 = centre[0]
        y0 = centre[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def radialcenter(I): 
    # Number of grid points
    Ny,Nx = I.shape
  
    

    # grid coordinates are -n:n, where Nx (or Ny) = 2*n+1
    # grid midpoint coordinates are -n+0.5:n-0.5;
    # The two lines below replace
    #    xm = repmat(-(Nx-1)/2.0+0.5:(Nx-1)/2.0-0.5,Ny-1,1);
    # and are faster (by a factor of >15 !)
    # -- the idea is taken from the repmat source code
    
    xm = np.zeros([Nx-1,Ny-1])
    for i in range(0,xm.shape[1]):
        xm[:,i] = np.arange(-(Nx-1)/2.0 + 0.5,(Nx)/2.0-0.5)

    ym = np.zeros([Nx-1,Ny-1])
    for i in range(0,ym.shape[1]):
        ym[:,:] = np.arange(-(Ny-1)/2.0+0.5,(Ny)/2.0-0.5)
    
    #ym = np.transpose(ym)
    # Calculate derivatives along 45-degree shifted coordinates (u and v)
    # Note that y increases "downward" (increasing row number) -- we'll deal
    # with this when calculating "m" below.

    dIdu = I[0:Ny-1,1:Nx] - I[1:Ny,0:Nx-1]
    dIdv = I[0:Ny-1,0:Nx-1] - I[1:Ny,1:Nx]

    # Smoothing --
    h = np.ones((3,3)) / 9 #simple 3x3 smoothing filter

    fdu = convolve2d(dIdu, h, mode = 'same')
    fdv = convolve2d(dIdv, h, mode = 'same')
    


    dImag2 = np.multiply(fdu,fdu) + np.multiply(fdv,fdv)
        
    # Slope of the gradient .  Note that we need a 45 degree rotation of
    # the u,v components to express the slope in the x-y coordinate system.
    # The negative sign "flips" the array to account for y increasing
    # "downward"
    m = np.divide(-(dImag2 + fdu),(fdu - dImag2))
    

    
    
    
    # print('m thing', fdv-fdu)
    # *Very* rarely, m might be NaN if (fdv + fdu) and (fdv - fdu) are both
    # zero.  In this case, replace with the un-smoothed gradient.

    
    if np.count_nonzero(np.isnan(m)) > 0:
        unsmoothm = (dIdv + dIdu) / (dIdu - dIdv)
        m[np.isnan(m)] = unsmoothm[np.isnan(m)]
    
    # If it's still NaN, replace with zero. (Very unlikely.)
    if np.count_nonzero(np.isnan(m)) > 0:
        m[np.isnan(m)] = 0

    # Almost as rarely, an element of m can be infinite if the smoothed u and v
    # derivatives are identical.  To avoid NaNs later, replace these with some
    # large number -- 10x the largest non-infinite slope.  The sign of the
    # infinity doesn't matter
    
    m = np.transpose(m)
    # Shorthand "b", which also happens to be the
    # y intercept of the line of slope m that goes through each grid midpoint
    b = ym - np.multiply(m,xm)

    

    
    # Weighting: weight by square of gradient magnitude and inverse
    # distance to gradient intensity centroid.
    sdI2 = sum(sum(dImag2))
    
    dImag2 = np.transpose(dImag2)
    
    xcentroid = sum(sum(np.multiply(dImag2,xm))) / sdI2
    ycentroid = sum(sum(np.multiply(dImag2,ym))) / sdI2
    w = dImag2 / np.sqrt(np.multiply((xm - xcentroid),(xm - xcentroid)) + np.multiply((ym - ycentroid),(ym - ycentroid)))
    

    def lsradialcenterfit(m, b, w): 
        # least squares solution to determine the radial symmetry center
        
        # inputs m, b, w are defined on a grid
        # w are the weights for each point
        wm2p1 = w / (np.multiply(m,m) + 1)
        sw = sum(sum(wm2p1))
        smmw = sum(sum(np.multiply(np.multiply(m,m),wm2p1)))
        smw = sum(sum(np.multiply(m,wm2p1)))
        smbw = sum(sum(np.multiply(np.multiply(m,b),wm2p1)))
        sbw = sum(sum(np.multiply(b,wm2p1)))
        det = smw * smw - smmw * sw
        xc = (smbw * sw - smw * sbw) / det
        
        yc = (smbw * smw - smmw * sbw) / det
        
        return xc,yc

    # least-squares minimization to determine the translated coordinate
    # system origin (xc, yc) such that lines y = mx+b have
    # the minimal total distance^2 to the origin:
    # See function lsradialcenterfit (below)
    xc,yc = lsradialcenterfit(m,b,w)
    
    ##
    # Return output relative to upper left coordinate
    xc = xc + (Nx + 1) / 2.0
    yc = yc + (Ny + 1) / 2.0

    # A rough measure of the particle width.
    # Not at all connected to center determination, but may be useful for tracking applications;
    # could eliminate for (very slightly) greater speed
    Isub = I - np.amin(I)
    px,py = np.meshgrid(np.arange(1,Nx+1),np.arange(1,Ny+1))
    #px py are correct
    
    
    xoffset = px - xc
    yoffset = py - yc
    
    #offsets are correct
    
    
    r2 = np.multiply(xoffset,xoffset) + np.multiply(yoffset,yoffset)
    sigma = np.sqrt(sum(sum(np.multiply(Isub,r2))) / sum(Isub)) / 2
    
    return xc, yc
    
    

####testing stuff:
# from numpy import genfromtxt
# frame = genfromtxt(r'X:/testim.csv', delimiter=',')
# #stack = np.load(r'X:\LCO Red-Blue\05_LCo-C22-1cyc-2C-blue_0\cam1/SynthStackPerturbed.npy')


# xcentre = 23.499
# ycentre = 23.01

# frame = np.ones([140,140])*1e-7
# gauss = makeGaussian(100,10, centre = [xcentre, ycentre])
# frame[0:100,0:100] = gauss 


# xc, yc = radialcenter(frame)







# print(xcentre-xc, ycentre-yc)

# plt.figure()
# plt.imshow(frame)
# plt.plot(xc,yc,'xb')
# plt.show()

