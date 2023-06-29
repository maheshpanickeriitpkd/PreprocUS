# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:46:48 2023
Adoped from https://github.com/alimuldal/phasepack
@author: Mahesh Panicker (mahesh@iitpkd.ac.in)
Arpan Tripathi (tripathiarpan20@gmail.com)
"""

#%% Import necessary libraries
import numpy as np
import cv2
from scipy.fftpack import fftshift, ifftshift
from scipy.fftpack import fft2, ifft2
from phasepack.filtergrid import filtergrid
from phasepack.tools import lowpassfilter as _lowpassfilter
from skimage.transform import radon,iradon

#%% Normalize the image between 0 and 1
def normalise(img):
  return (img - img.min())/(img.max() - img.min())


#%% Calculate integrated back scattered (IBS) energy
def integrated_backscatter_energy(img): #img is grayscale image
  ibs= np.cumsum(img ** 2,0)
  return ibs

#%% Calculate shadow energy
def indices(i, rows):
  ret = np.zeros((rows-i+1,))
  for i in range(ret.shape[0]):
    ret[i] = ret[i] + i
  return ret


def shadow(img):
  rows = img.shape[0]
  cols = img.shape[1]
  stdImg = round(rows/4)
  sh = np.zeros_like(img)

  for j in range(cols):
    for i in range(rows):
        gaussWin= np.exp(-((indices(i+1,rows))**2)/(2*(stdImg**2)))
        sh[i,j] = np.sum(np.multiply(img[i:rows,j], np.transpose(gaussWin)) / np.sum(gaussWin))        
  return sh

#%% Calculate local phase, local energy and phase symmetry
def filtergrid(rows, cols):

    # Set up u1 and u2 matrices with ranges normalised to +/- 0.5
    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         sparse=True)

    # Quadrant shift to put 0 frequency at the top left corner
    u1 = ifftshift(u1)
    u2 = ifftshift(u2)

    # Compute frequency values as a radius from centre (but quadrant shifted)
    radius = np.sqrt(u1 * u1 + u2 * u2)

    return radius, u1, u2

def lowpassfilter(size, cutoff, n):
    """
    Constructs a low-pass Butterworth filter:
        f = 1 / (1 + (w/cutoff)^2n)
    usage:  f = lowpassfilter(sze, cutoff, n)
    where:  size    is a tuple specifying the size of filter to construct
            [rows cols].
        cutoff  is the cutoff frequency of the filter 0 - 0.5
        n   is the order of the filter, the higher n is the sharper
            the transition is. (n must be an integer >= 1). Note
            that n is doubled so that it is always an even integer.
    The frequency origin of the returned filter is at the corners.
    """

    if cutoff < 0. or cutoff > 0.5:
        raise Exception('cutoff must be between 0 and 0.5')
    elif n % 1:
        raise Exception('n must be an integer >= 1')
    if len(size) == 1:
        rows = cols = size
    else:
        rows, cols = size

    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radius = np.sqrt(x * x + y * y)

    return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))


def rayleighmode(data, nbins=50):
    """
    Computes mode of a vector/matrix of data that is assumed to come from a
    Rayleigh distribution.
    usage:  rmode = rayleighmode(data, nbins)
    where:  data    data assumed to come from a Rayleigh distribution
            nbins   optional number of bins to use when forming histogram
                    of the data to determine the mode.
    Mode is computed by forming a histogram of the data over 50 bins and then
    finding the maximum value in the histogram. Mean and standard deviation
    can then be calculated from the mode as they are related by fixed
    constants.
        mean = mode * sqrt(pi/2)
        std dev = mode * sqrt((4-pi)/2)
    See:
        <http://mathworld.wolfram.com/RayleighDistribution.html>
        <http://en.wikipedia.org/wiki/Rayleigh_distribution>
    """
    n, edges = np.histogram(data, nbins)
    ind = np.argmax(n)
    return (edges[ind] + edges[ind + 1]) / 2.


def analyticEstimator(img, nscale=5, minWaveLength=10, mult=2.1, sigmaOnf=0.55, k=2.,\
                 polarity=0, noiseMethod=-1):

    if img.dtype not in ['float32', 'float64']:
        img = np.float64(img)
        imgdtype = 'float64'
    else:
        imgdtype = img.dtype

    if img.ndim == 3:
        img = img.mean(2)
    rows, cols = img.shape

    epsilon = 1E-4  # used to prevent /0.
    IM = fft2(img)  # Fourier transformed image

    zeromat = np.zeros((rows, cols), dtype=imgdtype)

    # Matrix for accumulating weighted phase congruency values (energy).
    totalEnergy = zeromat.copy()

    # Matrix for accumulating filter response amplitude values.
    sumAn = zeromat.copy()

    radius, u1, u2 = filtergrid(rows, cols)

    # Get rid of the 0 radius value at the 0 frequency point (at top-left
    # corner after fftshift) so that taking the log of the radius will not
    # cause trouble.
    radius[0, 0] = 1.

    H = (1j * u1 - u2) / radius


    lp = lowpassfilter([rows, cols], .4, 10)
    # Radius .4, 'sharpness' 10
    logGaborDenom = 2. * np.log(sigmaOnf) ** 2.

    for ss in range(nscale):
        wavelength = minWaveLength * mult ** ss
        fo = 1. / wavelength  # Centre frequency of filter

        logRadOverFo = np.log(radius / fo)
        logGabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        logGabor *= lp      # Apply the low-pass filter
        logGabor[0, 0] = 0.  # Undo the radius fudge

        IMF = IM * logGabor   # Frequency bandpassed image
        f = np.real(ifft2(IMF))  # Spatially bandpassed image

        # Bandpassed monogenic filtering, real part of h contains convolution
        # result with h1, imaginary part contains convolution result with h2.
        h = ifft2(IMF * H)

        # Squared amplitude of the h1 and h2 filters
        hAmp2 = h.real * h.real + h.imag * h.imag

        # Magnitude of energy
        sumAn += np.sqrt(f * f + hAmp2)

        # At the smallest scale estimate noise characteristics from the
        # distribution of the filter amplitude responses stored in sumAn. tau
        # is the Rayleigh parameter that is used to describe the distribution.
        if ss == 0:
            # Use median to estimate noise statistics
            if noiseMethod == -1:
                tau = np.median(sumAn.flatten()) / np.sqrt(np.log(4))

            # Use the mode to estimate noise statistics
            elif noiseMethod == -2:
                tau = rayleighmode(sumAn.flatten())

        # Calculate the phase symmetry measure

        # look for 'white' and 'black' spots
        if polarity == 0:
            totalEnergy += np.abs(f) - np.sqrt(hAmp2)

        # just look for 'white' spots
        elif polarity == 1:
            totalEnergy += f - np.sqrt(hAmp2)

        # just look for 'black' spots
        elif polarity == -1:
            totalEnergy += -f - np.sqrt(hAmp2)


    if noiseMethod >= 0:
        T = noiseMethod


    else:
        totalTau = tau * (1. - (1. / mult) ** nscale) / (1. - (1. / mult))

        # Calculate mean and std dev from tau using fixed relationship
        # between these parameters and tau. See
        # <http://mathworld.wolfram.com/RayleighDistribution.html>
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2.)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2.)

        # Noise threshold, must be >= epsilon
        T = np.maximum(EstNoiseEnergyMean + k * EstNoiseEnergySigma,
                       epsilon)
    # print(totalEnergy,'!!!!!!!!!\n')
    phaseSym = np.maximum(totalEnergy - T, 0)
    # print(phaseSym,'||||||||||||\n')
    phaseSym /= sumAn + epsilon

    #print(type(f), f.shape, f)
    #print(type(hAmp2), hAmp2.shape, hAmp2)

    LP = (1 - np.arctan2(np.sqrt(hAmp2),f))
    FS = phaseSym  
    LE = (hAmp2 + f*f)

    return LP, FS, LE
#%% Estimate Pleura Probability Map
def pleura_prob_map(img, minwl = 1):
  sh = normalise(shadow(img))*255
  ibs = normalise(integrated_backscatter_energy(img))*255
  shibs = normalise(ibs*sh)*255
  # shibs = shibs * (shibs >= shibs.mean())
  LP,FS,LE = analyticEstimator(img, minWaveLength = minwl)
  #ROI after pleura
  ImgbpROI = normalise( normalise(LP) * 255 * (255-shibs))
  #ROI before pleura
  ImgapROI = normalise( normalise(LP) *255* shibs) 
  return ImgapROI,ImgbpROI,sh,ibs,LP

#%% Radon transformation for keeping only horizontal edges/sructures
def radonHorzTrans(img,n = 18,filter='hann'):
    #filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
    theta_in = np.linspace(0., 180., num=180)
    r = radon(img, theta_in)
    
    desired_theta_idx = np.intersect1d(np.where(theta_in>90-n), np.where(theta_in<90+n))
    theta_out = theta_in[desired_theta_idx]    
    
    ir=iradon(r[:,desired_theta_idx], theta=theta_out, filter_name=filter)
    ir = cv2.resize(ir,(np.shape(img)[1],np.shape(img)[0]))
    return ir

#%% Radon transformation for keeping only vertical edges/sructures
def radonVertTrans(img,n = 72,filter='hann'):
    #filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']   
    theta_in = np.linspace(0., 180., num=180)
    r = radon(img, theta_in)
    
    indices_less_than_n = np.where(theta_in < n)[0]
    indices_greater_than_180_minus_n = np.where(theta_in > 180 - n)[0]   
    desired_theta_idx = np.concatenate((indices_less_than_n, indices_greater_than_180_minus_n))    
    theta_out = theta_in[desired_theta_idx]
    ir=iradon(r[:,desired_theta_idx], theta=theta_out, filter_name=filter)
    ir = cv2.resize(ir,(np.shape(img)[1],np.shape(img)[0]))
    return ir
