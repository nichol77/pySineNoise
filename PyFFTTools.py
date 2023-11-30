"""
File: PyFFTTools.py
Author: Ryan Nichol <r.nichol@ucl.ac.uk
Date: 2023-09-08

Description: A collection of very simple functions that can be used to process the output of fft in to various useful formats
"""

import numpy as np
from scipy.signal import butter,filtfilt,lfilter,lfilter_zi,sosfilt
import scipy

class PyFFTTools:
    
    def getNfromRFFT(yrfft):
        """Function to determine the timedomain length from an rfft
        Args:
            yrfft: The array of complex numbers from the rfft (this will be just the positive)
 
        Returns:
            The length N in the time domain

        """
        if np.isreal(yrfft[-1]):
            return 2*(len(yrfft)-1)
        else:
            return 2*(len(yrfft)-1)+1
        
    
    def convertRFFTToMag(yrfft):
        """Function to convert the result of an rfft to magnitude
        Args:
            yrfft: The array of complex numbers from the rfft (this will be just the positive)
 
        Returns:
            The array of magnitudes (of length N/2) where N is the length of yf

        """
        N=PyFFTTools.getNfromRFFT(yrfft)
        Nfft=len(yrfft)
        norm=np.sqrt(2)*np.ones(Nfft)/N
        norm[0]=norm[0]/np.sqrt(2)
        if N % 2 == 0:
            norm[-1]=norm[-1]/np.sqrt(2)
        return norm * np.abs(yrfft)
        
    def convertRFFTToPower(yf,R=50):
        """Function to convert the result of an fft to magnitude
    
        #Note this function may not give the correct values for the DC and Nyquist bins. Might fix this in a future version

        Args:
            yf: The array of complex numbers from the fft (this will be both the positive and negative frequencies)
            R: The impediance in Ohms
 
        Returns:
            The array of powers (of length N/2) where N is the length of yf

        """
        mag=PyFFTTools.convertRFFTToMag(yf)
        return (mag**2)/R  # The 2/N is the normalisation which results in 1V rms sine over a 1Ohm resistor having 1W

        
    
    def convertFFTToMag(yf):
        """Function to convert the result of an fft to magnitude
        Args:
            yf: The array of complex numbers from the fft (this will be both the positive and negative frequencies)
 
        Returns:
            The array of magnitudes (of length N/2) where N is the length of yf

        """
        N=yf.shape[0] #The length of yf
        Nfft=(N//2)+1
        norm=np.sqrt(2)*np.ones(Nfft)/N
        norm[0]=norm[0]/np.sqrt(2)
        if N % 2 == 0:
            norm[-1]=norm[-1]/np.sqrt(2)
        return norm * np.abs(yf[0:Nfft])  # sqrt(2)/N is the normalisation which results in 1V rms sine over a 1Ohm resistor having 1W

    def convertFFTToPower(yf,R=50):
        """Function to convert the result of an fft to magnitude
    
        #Note this function may not give the correct values for the DC and Nyquist bins. Might fix this in a future version

        Args:
            yf: The array of complex numbers from the fft (this will be both the positive and negative frequencies)
            R: The impediance in Ohms
 
        Returns:
            The array of powers (of length N/2) where N is the length of yf

        """
        mag=PyFFTTools.convertFFTToMag(yf)
        return (mag**2)/R  # The 2/N is the normalisation which results in 1V rms sine over a 1Ohm resistor having 1W

    def convertPowerToNormalisedMag(P,R=50):
        """Function to convert an array of powers into the normalised magnitude
    

        Args:
            P: The array of powers (in W) of length N/2
            R: The impediance in Ohms
 
        Returns:
            The array of normalised bin magnitudes

        """
        return np.sqrt(P*R)
    
    def convertPowerToBinMag(P,R=50):
        """Function to convert an array of powers into the normalised magnitude
    

        Args:
            P: The array of powers (in W) of length N/2
            R: The impediance in Ohms
 
        Returns:
            The array of bin magnitudes

        """
        N=2*P.shape[0]
        return np.sqrt(P*R*(N**2)/2)
    
    def convertPowerTodBm(P):
        """Function to convert Watts to dBm
        Args:
            P: The input power (or array of powers) in Watts
 
        Returns:
            The power (or array of powers) converted into dB relative to 1mW (dBm)
        """
        mW=1e-3#mW
        return 10*np.log10(P/mW)
    
    
    def convertdBmtoPower(P):
        """Function to convert power in dBm to power in Watts
        Args:
            P: The input power (or array of powers) in dB relative to 1mW (dBm)
 
        Returns:
            The power (or array of powers) converted into Watts
        """  
        mW=1e-3#mW
        return np.power(10,P/10)*mW

    def getRMS(y):
        """Function to convert Watts to dBm
        Args:
            y: The input array (e.g. could be an array of voltages)
 
        Returns:
            The root mean square of the values
        """
        return np.sqrt(np.mean(y**2))

    
    def convertFFTTodBm(cVals,R=50):
        """Function to convert the output of an FFT (assumed to be the fft of a voltage array in volts)
        Args:
            cVals: The array of N complex numbers from the fft (this will be both the positive and negative frequencies)
            R: The impediance in Ohms
 
        Returns:
            The array of N/2 power values per frequency bin in dBm
        """
        return PyFFTTools.convertPowerTodBm(PyFFTTools.convertFFTToPower(cVals,R))
    
    def convertRFFTTodBm(cVals,R=50):
        """Function to convert the output of an RFFT (assumed to be the fft of a voltage array in volts)
        Args:
            cVals: The array of N/2 complex numbers from the fft (this will be both the positive and negative frequencies)
            R: The impediance in Ohms
 
        Returns:
            The array of N/2 power values per frequency bin in dBm
        """
        return PyFFTTools.convertPowerTodBm(PyFFTTools.convertRFFTToPower(cVals,R))
    
        
    def getFrequencyBins(N,dt):
        """Function to generate a linear array of the N/2 positive frequency bins
        Args:
            N: The number of samples in time domain 
            dt: The time between samples (in some time unit)
 
        Returns:
            The array of frequency bins (in inverse units of dt)
        """
        Nfft=(N//2)+1
        return np.linspace(0.0, 1.0/(2.0*dt), Nfft)  #The N/2 frequency values from 0 to 1/(2*dt)
        
        
    def getNoiseRMS(T,R,df):
        """Function to determine the noise RMS from thermal noise in some bandwidth
        Args:
            T: The temperature in K
            R: The impediance in Ohms
            df: The bandwidth in Hz
 
        Returns:
            The sqrt of 4 * k * T * R * df
        """
        return np.sqrt(4 * scipy.constants.k * T *R *df)
    
    def convertVrmsTodBm(Vrms,R=50):
        """Function to convert Vrms to dbM
        Args:
            Vrms: The RMS voltage
            R: The impediance in Ohms
           
 
        Returns:
            Thew power in dBm
        """
        return PyFFTTools.convertPowerTodBm(Vrms**2/R)
        
    def convertFrequencyBinning(vnaGain,xfFFT):
        """Function to convert a gain curve from one set of freqeuncy bins to another
        Args:
            vnaGain: The gain measurement from the VNA a 2D array of frequency and gain
            xfFFT: The frequency binning of the FFT we are trying to match
           
 
        Returns:
            The gain curve in the FFT binning
        """
        #Here we are going to convert from the Network Analyser binning to the fft binning
        dfVNA=(vnaGain[1,0]-vnaGain[0,0])
        #print("dfVNA",dfVNA,"dfFFT",df)
        f0=vnaGain[0,0]
        ind=np.array((xfFFT)/dfVNA,dtype=int)
        #print("vnaGain",vnaGain[0:10])
        #print("ind",ind[0:10])
        #print("xfFFT",xfFFT[0:10])
        
        #print(ind)
        ind[ind>=len(vnaGain[:,0])]=len(vnaGain[:,0])-1
        return vnaGain[ind,1]
        
    def generateThermalNoiseFFTOrig(xf,gainLogMag,T=292,R=50):
        """Function to generate the complex thermal noise in the frequency domain
        Args:
            xf: The frequency binning of the FFT we are going to generate
            gainLogMag: The gain measurement as a 1D array of Log Magnitude
            T: The temperature in K
            R: The impedance in Ohms
           
 
        Returns:
            The complex FFT of simulated thermal noise with the given temperature and gain
        """
        noiseLevel=np.ones(xf.shape)*PyFFTTools.convertPowerTodBm(PyFFTTools.getNoiseRMS(T,R,(xf[1]-xf[0]))**2/R)
        noiseLevelNoise=noiseLevel+gainLogMag 
        binMags=PyFFTTools.convertPowerToBinMag(PyFFTTools.convertdBmtoPower(noiseLevelNoise))
        binStd=binMags/np.sqrt(2)
        realVals=np.random.normal(loc=0,scale=binStd)
        imagVals=np.random.normal(loc=0,scale=binStd)
        realVals=np.concatenate([realVals,np.flip(realVals)])
        imagVals=np.concatenate([np.array([0+1j])*imagVals,np.array([0-1j])*np.flip(imagVals)])
        complexVals=realVals+imagVals
        return complexVals
    
    def generateThermalNoiseRFFT(xf,gainLogMag,T=292,R=50):
        """Function to generate the complex thermal noise in the frequency domain
        Args:
            xf: The frequency binning of the FFT we are going to generate
            gainLogMag: The gain measurement as a 1D array of Log Magnitude
            T: The temperature in K
            R: The impedance in Ohms
           
 
        Returns:
            The complex Real FFT of simulated thermal noise with the given temperature and gain
        """
        #noiseLevelPower=np.ones(xf.shape)*PyFFTTools.getNoiseRMS(T,R,(xf[1]-xf[0]))**2/R
        #binMags=PyFFTTools.convertPowerToBinMag(PyFFTTools.convertdBmtoPower(noiseLevelPower))
        noiseLevel=np.ones(xf.shape)*PyFFTTools.convertPowerTodBm(PyFFTTools.getNoiseRMS(T,R,(xf[1]-xf[0]))**2/R)
        noiseLevelNoise=noiseLevel+gainLogMag 
        binMags=PyFFTTools.convertPowerToBinMag(PyFFTTools.convertdBmtoPower(noiseLevelNoise))
        binStd=binMags/np.sqrt(2)
        realVals=np.random.normal(loc=0,scale=binStd)
        imagVals=np.array([0+1j])*np.random.normal(loc=0,scale=binStd)
        #realVals=np.concatenate([realVals,np.array([0+0j])])
        #imagVals=np.concatenate([np.array([0+1j])*imagVals,np.array([0+0j])])
        complexVals=realVals+imagVals
        return complexVals
    
    def getThermalNoisePower(xf,gainLogMag,T=292,R=50):
        """Function to generate the thermal noise power level in the frequency domain
        Args:
            xf: The frequency binning of the FFT we are going to generate
            gainLogMag: The gain measurement as a 1D array of Log Magnitude
            T: The temperature in K
            R: The impedance in Ohms
           
 
        Returns:
           An array of the thermal noise power with the given temperature and gain
        """
        noiseLevel=np.ones(xf.shape)*PyFFTTools.convertPowerTodBm(PyFFTTools.getNoiseRMS(T,R,(xf[1]-xf[0]))**2/R)
        noiseLevelNoise=noiseLevel+gainLogMag 
        return PyFFTTools.convertdBmtoPower(noiseLevelNoise)
    
    def convertRFFTtoFFT(rfft):
        """Function to convert the N//2 + 1 numbers from an np.fft.rfft to the N numbers from np.fft.fft
        Args:
            rfft: The N//2 + 1 numbers from an np.fft.rfft
           
 
        Returns:
            The N numbers we would have got from np.fft.fft
        """
        return np.concatenate([rfft,np.flip(np.conjugate(rfft[np.iscomplex(rfft)]))])
    
    
        
    def convertFFTtoRFFT(fft):
        """Function to convert the N numbers from np.fft.ff to the N//2 + 1 numbers from an np.fft.rfft
        Args:
            rfft: The N numbers we got from np.fft.fft
           
 
        Returns:
             The N//2 + 1 numbers we would have got from an np.fft.rfft
        """
        Nfft=(len(fft)//2)+1
        return fft[0:Nfft]
        
        
    #Wrapper for a lowpass butterworth filter
    #Not very efficient at the moment
    def butter_lowpass_filter(data, cutoff, fs, order):
        # Filter requirements.
        #T = 5.0         # Sample Period
        #fs = 30.0       # sample rate, Hz
        #cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        #order = 2       # sin wave can be approx represented as quadratic
        #n = int(T * fs) # total number of samples
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
     #Wrapper for a lowpass butterworth filter
    #Not very efficient at the moment
    def butter_lowpass_lfilter(data, cutoff, fs, order):
        # Filter requirements.
        #T = 5.0         # Sample Period
        #fs = 30.0       # sample rate, Hz
        #cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        #order = 2       # sin wave can be approx represented as quadratic
        #n = int(T * fs) # total number of samples
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        zi = lfilter_zi(b, a)
        z, _ = lfilter(b, a, data, zi=zi*data[0])
        return z
    
     #Wrapper for a lowpass butterworth filter
    #Not very efficient at the moment
    def butter_lowpass_sosfilter(data, cutoff, fs, order):
        # Filter requirements.
        #T = 5.0         # Sample Period
        #fs = 30.0       # sample rate, Hz
        #cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        #order = 2       # sin wave can be approx represented as quadratic
        #n = int(T * fs) # total number of samples
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        sos = butter(order, normal_cutoff, btype='low', analog=False,output='sos')
        return sosfilt(sos, data)
    