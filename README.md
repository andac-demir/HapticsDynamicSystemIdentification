
# System Identification in Frequency Domain


```python
import os
import pywt
import itertools
from itertools import zip_longest
import numpy as np
from numpy.linalg import norm
import math
import pandas as pd
import bamboolib # gui for pandas df
from scipy.fftpack import fft
from scipy.signal import welch
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.tsa.api import VAR
from tqdm import tqdm_notebook
from IPython.core.debugger import set_trace
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter

from scipy.io import loadmat
import seaborn as sns
from math import floor
import tkinter as tk
import decimal
from random import shuffle
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import concurrent.futures # for parallel processing
import glob
```

# Loading The Dataset (Trial by Trial)


```python
# Get the participant data from Dataset directory
global t0, fs, dt, num_eeg_ch, num_emg_ch, num_force_ch, num_ch, num_conds
t0 = 0
fs = 1200
dt = 1.0/fs
num_eeg_ch = 14
num_emg_ch = 4
num_force_ch = 3
num_ch = num_eeg_ch + num_emg_ch + num_force_ch
num_conds = 18

# Get a list of files to process
data_files = list(map(loadmat, glob.glob("Dataset/*.mat")))
num_participants = len(data_files)

columns=['EEG-1','EEG-2','EEG-3','EEG-4','EEG-5',
         'EEG-6','EEG-7','EEG-8','EEG-9','EEG-10',
         'EEG-11','EEG-12','EEG-13','EEG-14','EMG-1',
         'EMG-2','EMG-3','EMG-4','Force-x', 
         'Force-y','Force-z']

glob.glob("Dataset/*.mat")
```




    ['Dataset/WC1200Hz.mat',
     'Dataset/P11200Hz.mat',
     'Dataset/JT1200Hz.mat',
     'Dataset/BT1200Hz.mat',
     'Dataset/YA1200Hz.mat',
     'Dataset/YZ1200Hz.mat',
     'Dataset/HCH1200Hz.mat',
     'Dataset/CC1200Hz.mat',
     'Dataset/ED1200Hz.mat',
     'Dataset/YK1200Hz.mat',
     'Dataset/KH1200Hz.mat']




```python
def extract_data(file):
    data_splitBy_trials = [] # list of data frames, each df corresponding to a trial

    for cond in tqdm_notebook(range(num_conds), ascii=True):
        num_trials = file['EEGSeg_Ch'][0,0][0,cond].shape[0]
        # if a trial has long enough samples
        if file['EEGSeg_Ch'][0,0][0,cond].shape[1] >= 400:
            for trial in range(num_trials): 
                data = pd.DataFrame(columns=columns)
                for ch in range(num_eeg_ch):
                    data.iloc[:,ch] = file['EEGSeg_Ch'][0,ch][0,cond][trial,:]
                for ch in range(num_emg_ch):
                    data.iloc[:,ch+num_eeg_ch] = file['EMGSeg_Ch'][0,ch][0,cond][trial,:]
                for ch in range(num_force_ch):
                    data.iloc[:,ch+num_eeg_ch+num_emg_ch] = file['ForceSeg_Ch'][0,ch][0,cond][trial,:]

                # mean subtraction in each trial from the eeg and emg columns for removing dc drift
                data.iloc[:,:18] -= data.iloc[:,:18].mean()
                # convert volts to microvolts for EEG and EMG channels
                data.iloc[:,:18] *= 1e6
                # convert volts to milivolts for force channels
                data.iloc[:,18:] *= 1e3  
                data_splitBy_trials.append(data)
                
    train_data, test_data = train_test_split(data_splitBy_trials)
    return train_data, test_data


def train_test_split(data, train_ratio=0.5):
    shuffle(data) # shuffle list, in-place operator
    num_train = int(train_ratio*len(data))
    train_data, test_data = data[:num_train], data[num_train:]
    return train_data, test_data


train_data = [] # train data is a list (all participants each trial) of dataframes
test_data = [] # test data is a list (each participant) of list (each trial) of dataframes 

print('Reading and processing in parallel.')

# Create a pool of processes. By default, one is created for each CPU in your machine.
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Process the list of files, but split the work across the process pool to use all CPUs.
    # executor.map() function takes in the helper function to call and the list of data to process with it.
    # executor.map() function returns results in  the same order as the list of data given to the process.
    for train, test in tqdm_notebook(executor.map(extract_data, data_files), total=len(data_files)):
        train_data.extend(train)
        test_data.append(test)
        
total_trials = len(train_data)
for test in test_data:
    total_trials += len(test)
print('Total number of trials of all the conditions in the file: %i' %total_trials) 
train_data[0].head(10) # first 10 samples of the first trial
```

    Reading and processing in parallel.
    
    Total number of trials of all the conditions in the file: 17648
    
<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EEG-1</th>
      <th>EEG-2</th>
      <th>EEG-3</th>
      <th>EEG-4</th>
      <th>EEG-5</th>
      <th>EEG-6</th>
      <th>EEG-7</th>
      <th>EEG-8</th>
      <th>EEG-9</th>
      <th>EEG-10</th>
      <th>...</th>
      <th>EEG-12</th>
      <th>EEG-13</th>
      <th>EEG-14</th>
      <th>EMG-1</th>
      <th>EMG-2</th>
      <th>EMG-3</th>
      <th>EMG-4</th>
      <th>Force-x</th>
      <th>Force-y</th>
      <th>Force-z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.353050</td>
      <td>-6.342973</td>
      <td>6.819101</td>
      <td>-2.594969</td>
      <td>7.674979</td>
      <td>12.857037</td>
      <td>6.150998</td>
      <td>6.870037</td>
      <td>0.286839</td>
      <td>-1.858157</td>
      <td>...</td>
      <td>12.602734</td>
      <td>6.593984</td>
      <td>4.943229</td>
      <td>-490.881225</td>
      <td>-1416.124088</td>
      <td>12.110258</td>
      <td>-822.497425</td>
      <td>-143.306509</td>
      <td>-142.857805</td>
      <td>107.210733</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.049549</td>
      <td>-5.716996</td>
      <td>6.335072</td>
      <td>-2.228633</td>
      <td>8.675963</td>
      <td>12.129119</td>
      <td>5.363917</td>
      <td>7.693931</td>
      <td>0.764127</td>
      <td>-1.339923</td>
      <td>...</td>
      <td>13.400500</td>
      <td>7.402246</td>
      <td>5.753743</td>
      <td>-270.386181</td>
      <td>-1754.931790</td>
      <td>9.632009</td>
      <td>-1244.423804</td>
      <td>-143.149018</td>
      <td>-142.699182</td>
      <td>107.372403</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.856026</td>
      <td>-5.097317</td>
      <td>5.981134</td>
      <td>-1.943278</td>
      <td>9.534561</td>
      <td>11.588279</td>
      <td>4.802978</td>
      <td>8.387277</td>
      <td>1.110266</td>
      <td>-0.958579</td>
      <td>...</td>
      <td>14.097848</td>
      <td>8.106449</td>
      <td>6.384907</td>
      <td>-141.042472</td>
      <td>-1761.562806</td>
      <td>6.286931</td>
      <td>-1243.261514</td>
      <td>-142.994955</td>
      <td>-142.544284</td>
      <td>107.539088</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.812003</td>
      <td>-4.479552</td>
      <td>5.808489</td>
      <td>-1.732781</td>
      <td>10.231863</td>
      <td>11.299196</td>
      <td>4.545568</td>
      <td>8.944809</td>
      <td>1.343468</td>
      <td>-0.703989</td>
      <td>...</td>
      <td>14.691936</td>
      <td>8.708110</td>
      <td>6.831515</td>
      <td>-106.098784</td>
      <td>-1790.783984</td>
      <td>1.174901</td>
      <td>-1245.794711</td>
      <td>-141.831577</td>
      <td>-141.370296</td>
      <td>108.844407</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.955704</td>
      <td>-3.852129</td>
      <td>5.863095</td>
      <td>-1.574301</td>
      <td>10.768366</td>
      <td>11.315863</td>
      <td>4.654234</td>
      <td>9.379075</td>
      <td>1.501485</td>
      <td>-0.546178</td>
      <td>...</td>
      <td>15.195866</td>
      <td>9.223468</td>
      <td>7.112822</td>
      <td>-74.663853</td>
      <td>-1782.945973</td>
      <td>-0.607883</td>
      <td>-1269.338546</td>
      <td>-137.539312</td>
      <td>-137.079418</td>
      <td>113.612279</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.319859</td>
      <td>-3.198186</td>
      <td>6.180563</td>
      <td>-1.430718</td>
      <td>11.162797</td>
      <td>11.675831</td>
      <td>5.169790</td>
      <td>9.718120</td>
      <td>1.636877</td>
      <td>-0.438804</td>
      <td>...</td>
      <td>15.635344</td>
      <td>9.680359</td>
      <td>7.269382</td>
      <td>-77.112765</td>
      <td>-1776.046735</td>
      <td>-1.627216</td>
      <td>-1331.983027</td>
      <td>-129.153624</td>
      <td>-128.712311</td>
      <td>122.906871</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.927914</td>
      <td>-2.498382</td>
      <td>6.781974</td>
      <td>-1.254640</td>
      <td>11.448938</td>
      <td>12.395821</td>
      <td>6.106377</td>
      <td>10.001562</td>
      <td>1.810520</td>
      <td>-0.324553</td>
      <td>...</td>
      <td>16.043936</td>
      <td>10.113869</td>
      <td>7.357503</td>
      <td>-117.799455</td>
      <td>-1820.213777</td>
      <td>3.970964</td>
      <td>-1406.503734</td>
      <td>-119.542882</td>
      <td>-119.089819</td>
      <td>133.681715</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6.790958</td>
      <td>-1.733881</td>
      <td>7.671055</td>
      <td>-0.993781</td>
      <td>11.671012</td>
      <td>13.469359</td>
      <td>7.449102</td>
      <td>10.275510</td>
      <td>2.084202</td>
      <td>-0.141676</td>
      <td>...</td>
      <td>16.457591</td>
      <td>10.561314</td>
      <td>7.442016</td>
      <td>-73.740446</td>
      <td>-1896.775943</td>
      <td>5.970048</td>
      <td>-1489.205179</td>
      <td>-112.978227</td>
      <td>-112.514667</td>
      <td>141.081572</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.905754</td>
      <td>-0.889143</td>
      <td>8.833048</td>
      <td>-0.596917</td>
      <td>11.878159</td>
      <td>14.866661</td>
      <td>9.154603</td>
      <td>10.586955</td>
      <td>2.512989</td>
      <td>0.168920</td>
      <td>...</td>
      <td>16.908969</td>
      <td>11.057185</td>
      <td>7.588219</td>
      <td>80.008804</td>
      <td>-1986.421329</td>
      <td>6.111144</td>
      <td>-1598.266778</td>
      <td>-110.970661</td>
      <td>-110.510454</td>
      <td>143.338755</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.254008</td>
      <td>0.045467</td>
      <td>10.235321</td>
      <td>-0.019850</td>
      <td>12.118512</td>
      <td>16.536747</td>
      <td>11.154366</td>
      <td>10.978111</td>
      <td>3.138313</td>
      <td>0.656843</td>
      <td>...</td>
      <td>17.422336</td>
      <td>11.628380</td>
      <td>7.853707</td>
      <td>4.907882</td>
      <td>-2074.367982</td>
      <td>7.516975</td>
      <td>-1748.276768</td>
      <td>-113.734700</td>
      <td>-113.275625</td>
      <td>140.257925</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 21 columns</p>
</div>




```python
trial_id = 40 # choose one trial
num_samples = train_data[trial_id].shape[0]
time = (np.arange(0, num_samples) * dt + t0) * 1e3
# pick one channel from eeg, emg and force
eeg_signal_1 = train_data[trial_id].iloc[:,0] # trial id, eeg_ch1
emg_signal_1 = train_data[trial_id].iloc[:,14] # trial id, emg_ch1
force_signal_3 = train_data[trial_id].iloc[:,20] # trial id, force_ch3
```

# Visualization of The Dataset


```python
def get_ave_values(xvalues, yvalues, n):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave

def plot_signal_plus_average(ax, time, signal, average_over):
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(average_over))
    ax.set_xlim([time[0], time[-1]])
    ax.legend(loc='upper right')

fig, axes = plt.subplots(3, figsize=(12,12))
fig.subplots_adjust(hspace=0.3)
plot_signal_plus_average(axes[0], time, eeg_signal_1, average_over=3)

axes[0].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[0].set_title('EEG Channel-1', fontsize=16)
plot_signal_plus_average(axes[1], time, emg_signal_1, average_over=3)

axes[1].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[1].set_title('EMG Channel-1', fontsize=16)
plot_signal_plus_average(axes[2], time, force_signal_3, average_over=3)

axes[2].set_ylabel('Amplitude (mV)', fontsize=16)
axes[2].set_xlabel('Time (sec)', fontsize=16)
axes[2].set_title('Force Channel-3', fontsize=16)

plt.show()
```


![png](spectralAnalysis_files/spectralAnalysis_7_0.png)


# Fourier Transform of The Data


```python
def get_fft_values(y_values, T, N, f_s):
    N2 = 2 ** (int(np.log2(N)) + 1) # round up to next highest power of 2
    f_values = np.linspace(0.0, 1.0/(2.0*T), N2//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N2 * np.abs(fft_values_[0:N2//2])
    return f_values, fft_values

def plot_fft_plus_power(axes, time, signal, N, fs=fs, dt=dt,
                        plot_direction='horizontal', yticks=None, ylim=None):
    variance = np.std(signal)**2
    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2

    if plot_direction == 'horizontal':
        axes[0].plot(f_values, fft_values, 'r-', label='Fourier Transform')
        axes[1].plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
    elif plot_direction == 'vertical':
        scales = 1./f_values
        scales_log = np.log2(scales)
        axes[0].plot(fft_values, scales_log, 'r-', label='Fourier Transform')
        axes[1].plot(fft_power, scales_log, 'k--', linewidth=1, label='FFT Power Spectrum')
        axes[0].set_yticks(np.log2(yticks))
        axes[0].set_yticklabels(yticks)
        axes[0].invert_yaxis()
        axes[0].set_ylim(ylim[0], -1)
    axes[0].legend()
    axes[1].legend()

fig, axes = plt.subplots(6, figsize=(12,30))
fig.subplots_adjust(hspace=0.3)
axes[0].set_title('Fourier Transform - EEG Channel 1', fontsize=16)
axes[1].set_title('FFT Power Spectrum - EEG Channel 1', fontsize=16)
axes[0].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[1].set_ylabel('Amplitude', fontsize=16)
plot_fft_plus_power([axes[0], axes[1]], time, eeg_signal_1, len(eeg_signal_1))

axes[2].set_title('Fourier Transform - EMG Channel 1', fontsize=16)
axes[3].set_title('FFT Power Spectrum - EMG Channel 1', fontsize=16)
axes[2].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[3].set_ylabel('Amplitude', fontsize=16)
plot_fft_plus_power([axes[2], axes[3]], time, emg_signal_1, len(emg_signal_1))

axes[4].set_title('Fourier Transform - Force Channel 3', fontsize=16)
axes[5].set_title('FFT Power Spectrum - Force Channel 3', fontsize=16)
axes[4].set_ylabel('Amplitude (mV)', fontsize=16)
axes[5].set_ylabel('Amplitude', fontsize=16)
plot_fft_plus_power([axes[4], axes[5]], time, force_signal_3, len(force_signal_3))
axes[5].set_xlabel('Frequency [Hz]', fontsize=16)

plt.show()
```


![png](spectralAnalysis_files/spectralAnalysis_9_0.png)


# Estimation of The Power Spectral Density Using Welch’s Method


```python
# Welch’s method computes an estimate of the power spectral density by dividing the data
# into overlapping segments, computing a modified periodogram for each segment 
# and averaging the periodograms.

# For the default ‘hanning’ window an overlap of 50% is a reasonable trade off 
# between accurately estimating the signal power, while not over counting any of the data. 
# Narrower windows may require a larger overlap.
for counter, x in enumerate([eeg_signal_1, emg_signal_1, force_signal_3]):
    f, Pxx_spec = welch(x, fs, 'hanning', len(x), scaling='spectrum')
    plt.figure(figsize=(12,6))    
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.xlabel('Frequency [Hz]')
    if counter == 0:
        plt.ylabel('EEG Channel 1 - Linear spectrum [\u03bcV RMS]', fontsize=16)
    elif counter == 1:
        plt.ylabel('EMG Channel 1 - Linear spectrum [\u03bcV RMS]', fontsize=16)
    else:
        plt.ylabel('Force Channel z - Linear spectrum [mV RMS]', fontsize=16)
    plt.show()
    
# The peak height in the power spectrum is an estimate of the RMS amplitude.
```


![png](spectralAnalysis_files/spectralAnalysis_11_0.png)



![png](spectralAnalysis_files/spectralAnalysis_11_1.png)



![png](spectralAnalysis_files/spectralAnalysis_11_2.png)


# Discrete Wavelet Transform Coefficients


```python
# cA: approximation coefficients (low frquency components)
# cD: detail coefficients (high frequency components)
(cA_eeg1, cD_eeg1) = pywt.dwt(eeg_signal_1, 'db4')
(cA_emg1, cD_emg1) = pywt.dwt(emg_signal_1, 'db4')
(cA_force3, cD_force3) = pywt.dwt(force_signal_3, 'db4')

list_of_coeffs = [cA_eeg1, cD_eeg1, cA_emg1, cD_emg1, cA_force3, cD_force3]

# Plot a histogram and kernel density estimate of wavelet coefficients
def hist_dwtCoeffs(list_of_coeffs):
    sns.set(style="white", palette="muted", color_codes=True)
    fig, axes = plt.subplots(3, 2, figsize=(32,32), sharex=False)
    for i, coeffs in enumerate(list_of_coeffs):
        sns.distplot(coeffs, bins=40, ax=axes[floor(i/2), i%2])
        axes[floor(i/2), i%2].tick_params(axis='both', which='major', labelsize=30)

    axes[0, 0].set_title('Approximation Coeff. EEG-1', fontsize=30) 
    axes[0, 1].set_title('Detail Coeff. EEG-1', fontsize=30)        
    axes[1, 0].set_title('Approximation Coeff. EMG-1', fontsize=30) 
    axes[1, 1].set_title('Detail Coeff. EMG-1', fontsize=30)    
    axes[2, 0].set_title('Approximation Coeff. Force-z', fontsize=30) 
    axes[2, 1].set_title('Detail Coeff. Force-z', fontsize=30)  
    axes[2, 0].set_xlim([-350, 20])

    plt.tight_layout()
    
hist_dwtCoeffs(list_of_coeffs)
```


![png](spectralAnalysis_files/spectralAnalysis_13_0.png)


Length of approximation and detail coefficients is 3 samples longer than half the length 
of the given signal. If the signal length is odd, half the signal length is rounded up.

# Parsimonity of Transform

### a) Reconstruction Without Compression


```python
def get_reconstruction_score(original_sig, reconstructed_sig):
    return 100 * (norm(reconstructed_sig) ** 2 / norm(original_sig) ** 2)
    
scores = []
reconstructed_sigs = []
# Plot the reconstructed signals and show reconstruction scores
fig, axes = plt.subplots(3, figsize=(12,12), sharex=False)
for i, signal in enumerate([eeg_signal_1, emg_signal_1, force_signal_3]):
    axes[i].plot(time, signal, label='original')
    (cA, cD) = pywt.dwt(signal, 'db4')  
    reconstructed_sigs.append(pywt.idwt(cA, cD, 'db4'))  
    scores.append(get_reconstruction_score(signal, reconstructed_sigs[-1][:-1]))

for i, recon in enumerate(reconstructed_sigs):
    axes[i].plot(time, recon[:-1], label='reconstructed')
    axes[i].tick_params(axis='both', which='major', labelsize=16)
    axes[i].legend(loc='upper right')
    
axes[0].set_title('Reconstruction EEG-1, score: %.1f' %scores[0]+'%', fontsize=16) 
axes[0].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[1].set_title('Reconstruction EMG-1, score: %.1f' %scores[1]+'%', fontsize=16)     
axes[1].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[2].set_title('Reconstruction Force-z, score: %.1f' %scores[2]+'%', fontsize=16) 
axes[2].set_ylabel('Amplitude (mV)', fontsize=16)
axes[2].set_xlabel('Time (sec)', fontsize=16)

plt.tight_layout()
```


![png](spectralAnalysis_files/spectralAnalysis_17_0.png)


### b) Wavelet Data Compression by Thresholding


```python
'''
Soft thresholding sets the wavelet coefficients with absolute value below the threshold to 0 and
other coefficients towards the origin by the magnitude of threshold.

Hard thresholding just sets the wavelet coefficients with absolute value below the threshold to 0.
'''
thresholded_list_of_coeffs = []

def threshold_coeffs(coeffs, threshold=1.0, soft=True): # set threshold 0.2 for emg channels.
    if soft == True:
        return [0 if abs(coef)<threshold else coef-np.sign(coef)*threshold for coef in coeffs]
    else:
        return [0 if abs(coef)<threshold else coef for coef in coeffs]

for coeffs in list_of_coeffs:
    thresholded_list_of_coeffs.append(threshold_coeffs(coeffs))

hist_dwtCoeffs(thresholded_list_of_coeffs)

# Plot the reconstructed signals and show reconstruction scores
scores, reconstructed_sigs = [], []
fig, axes = plt.subplots(3, figsize=(12,12), sharex=False)
for i, signal in enumerate([eeg_signal_1, emg_signal_1, force_signal_3]):
    axes[i].plot(time, signal, label='original')
    cA = thresholded_list_of_coeffs[2*i]
    cD = thresholded_list_of_coeffs[2*i+1]
    reconstructed_sigs.append(pywt.idwt(cA, cD, 'db4'))  
    scores.append(get_reconstruction_score(signal, reconstructed_sigs[-1][:-1]))

for i, recon in enumerate(reconstructed_sigs):
    axes[i].plot(time, recon[:-1], label='reconstructed')
    axes[i].tick_params(axis='both', which='major', labelsize=16)
    axes[i].legend(loc='upper right')
    
axes[0].set_title('Reconstruction EEG-1, score: %.1f' %scores[0]+'%', fontsize=16) 
axes[0].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[1].set_title('Reconstruction EMG-1, score: %.1f' %scores[1]+'%', fontsize=16) 
axes[1].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[2].set_title('Reconstruction Force-z, score: %.1f' %scores[2]+'%', fontsize=16) 
axes[2].set_ylabel('Amplitude (mV)', fontsize=16)
axes[2].set_xlabel('Time (sec)', fontsize=16)

plt.tight_layout()
```


![png](spectralAnalysis_files/spectralAnalysis_19_0.png)



![png](spectralAnalysis_files/spectralAnalysis_19_1.png)


### c) Wavelet Data Compression by Parseval's Theorem


```python
def retain_energy(cA, cD, original_signal, retained_ratio=.99):
    energy = norm(original_signal) ** 2 # energy of the original signal
    c = cA + cD 
    sorted_c = sorted(c, key=abs)[::-1] # dwt coeffs sorted in a decreasing order of their magnitude
    sum, i = 0, 0
    # evaluate the energy of the compressed signal until a certain amount of energy 
    # from the original signal is retained.
    while sum <= energy * retained_ratio:
        sum += sorted_c[i] ** 2
        i += 1
    th = abs(sorted_c[i])
    cA = threshold_coeffs(cA, threshold=th, soft=False)
    cD = threshold_coeffs(cD, threshold=th, soft=False)
    return cA, cD

retained_list_of_coeffs = []

# split the list of coefficients into [cA, cD] for each signal
split_list = [list_of_coeffs[i:i+2] for i in range(0, len(list_of_coeffs), 2)]
for coeffs, signal in zip(split_list, [eeg_signal_1, emg_signal_1, force_signal_3]):
    retained_list_of_coeffs.append(retain_energy(coeffs[0], coeffs[1], signal)[0])
    retained_list_of_coeffs.append(retain_energy(coeffs[0], coeffs[1], signal)[1])

hist_dwtCoeffs(retained_list_of_coeffs)

# Plot the reconstructed signals and show reconstruction scores
scores, reconstructed_sigs = [], []
fig, axes = plt.subplots(3, figsize=(12,12), sharex=False)
for i, signal in enumerate([eeg_signal_1, emg_signal_1, force_signal_3]):
    axes[i].plot(time, signal, label='original')
    cA = retained_list_of_coeffs[2*i]
    cD = retained_list_of_coeffs[2*i+1]
    reconstructed_sigs.append(pywt.idwt(cA, cD, 'db4'))  
    scores.append(get_reconstruction_score(signal, reconstructed_sigs[-1][:-1]))

for i, recon in enumerate(reconstructed_sigs):
    axes[i].plot(time, recon[:-1], label='reconstructed')
    axes[i].tick_params(axis='both', which='major', labelsize=16)
    axes[i].legend(loc='upper right')
    
axes[0].set_title('Reconstruction EEG-1, score: %.1f' %scores[0]+'%', fontsize=16) 
axes[0].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[1].set_title('Reconstruction EMG-1, score: %.1f' %scores[1]+'%', fontsize=16) 
axes[1].set_ylabel('Amplitude (\u03bcV)', fontsize=16)
axes[2].set_title('Reconstruction Force-3, score: %.1f' %scores[2]+'%', fontsize=16) 
axes[2].set_ylabel('Amplitude (mV)', fontsize=16)
axes[2].set_xlabel('Time (sec)', fontsize=16)

plt.tight_layout()

```


![png](spectralAnalysis_files/spectralAnalysis_21_0.png)



![png](spectralAnalysis_files/spectralAnalysis_21_1.png)


# Vector Auto-Regression Based on DWT Coefficients


```python
wl_columns=['cA_EEG-1','cD_EEG-1','cA_EEG-2','cD_EEG-2','cA_EEG-3',
            'cD_EEG-3','cA_EEG-4','cD_EEG-4','cA_EEG-5','cD_EEG-5',
            'cA_EEG-6','cD_EEG-6','cA_EEG-7','cD_EEG-7','cA_EEG-8',
            'cD_EEG-8','cA_EEG-9','cD_EEG-9','cA_EEG-10','cD_EEG-10', 
            'cA_EEG-11','cD_EEG-11','cA_EEG-12','cD_EEG-12', 'cA_EEG-13',
            'cD_EEG-13','cA_EEG-14','cD_EEG-14','cA_EMG-1','cD_EMG-1',
            'cA_EMG-2','cD_EMG-2','cA_EMG-3','cD_EMG-3','cA_EMG-4','cD_EMG-4',
            'cA_Force-x','cD_Force-x', 'cA_Force-y','cD_Force-y', 
            'cA_Force-z','cD_Force-z']

# take the discrete wavelet coefficients of train and test data for each trial
def get_wavelet(data):
    wavelet_coef = pd.DataFrame(columns=wl_columns)
    th = 14*[1] + 4*[0.2] + 3*[1] # soft threshold list for all channels
    wavelet_list = []
    for i in tqdm_notebook(range(len(data)), ascii=True):  # for each trial
        for j in range(num_ch):
            cA, cD = pywt.dwt(data[i].iloc[:,j].values, 'db4')
            # Apply soft-thresholding on DWT coeffs
            wavelet_coef[wl_columns[2*j]] = threshold_coeffs(cA, threshold=th[j], soft=True)
            wavelet_coef[wl_columns[2*j+1]] = threshold_coeffs(cD, threshold=th[j], soft=True)
        wavelet_list.append(wavelet_coef)
        # delete the df wavelet_coef for the next trial
        wavelet_coef = wavelet_coef.drop(wavelet_coef.index[:])
    return wavelet_list

train_wavelet = get_wavelet(train_data)
print('Wavelet transform ready for train data')
test_wavelet = []
for i, participant in enumerate(test_data):
    test_wavelet.append(get_wavelet(participant))
    print('Wavelet transform ready for participant %s' %str(i+1))
```  

### Model Lag Order Selection 


```python
# saves bic scores for all model orders:
def save_bic(dataframe, filename, use_saved=True):
    if use_saved==True: return
    lag_bic_scores = np.zeros((1,20))
    for P in tqdm_notebook(range(20), ascii=True):
        model = VAR(dataframe)
        results = model.fit(P+1)
        bic_score = results.bic
        lag_bic_scores[0, P] = bic_score
        np.save(filename, lag_bic_scores)

train_wavelet_df = pd.concat(train_wavelet)

# use only approximation coeefficients for eeg avoid columns with 0's,
# in order to yield a positive-semidefinite matrix
eeg_ind_wl = list(range(0, 28, 2))
emg_ind_wl = list(range(28, 36, 1))
force_ind_wl = list(range(36, 42, 1))

# Only EEG (first 28 columns)   
filename = 'VARResults/onlyEEG_lagOrder.npy'
save_bic(train_wavelet_df.iloc[:,eeg_ind_wl], filename)
print("Lag order selection completed for EEG only.")

# Only EMG (28-35 columns)
filename = 'VARResults/onlyEMG_lagOrder.npy'
save_bic(train_wavelet_df.iloc[:,emg_ind_wl], filename)
print("Lag order selection completed for EMG only.")
 
# Only Force (35-41 columns)
filename = 'VARResults/onlyForce_lagOrder.npy'
save_bic(train_wavelet_df.iloc[:,force_ind_wl], filename)
print("Lag order selection completed for Force only.")

# EEG & EMG
filename = 'VARResults/EEG_EMG_lagOrder.npy'
save_bic(train_wavelet_df.iloc[:,eeg_ind_wl+emg_ind_wl], filename)
print("Lag order selection completed for EEG and EMG.")

# EEG & Force
filename = 'VARResults/EEG_Force_lagOrder.npy'
save_bic(train_wavelet_df.iloc[:,eeg_ind_wl+force_ind_wl], filename)
print("Lag order selection completed for EEG and Force.")

# EMG & Force
filename = 'VARResults/EMG_Force_lagOrder.npy'
save_bic(train_wavelet_df.iloc[:,emg_ind_wl+force_ind_wl], filename)
print("Lag order selection completed for EMG and Force.")

# all
filename = 'VARResults/all_lagOrder.npy'
save_bic(train_wavelet_df.iloc[:,eeg_ind_wl+emg_ind_wl+force_ind_wl], filename)
print("Lag order selection completed for all.")
```
  

### Model Lag Order Results


```python
all = np.load('VARResults/all_lagOrder.npy')
only_eeg = np.load('VARResults/onlyEEG_lagOrder.npy')
only_emg = np.load('VARResults/onlyEMG_lagOrder.npy')
only_force = np.load('VARResults/onlyForce_lagOrder.npy')
eeg_emg = np.load('VARResults/EEG_EMG_lagOrder.npy')
eeg_force = np.load('VARResults/EEG_Force_lagOrder.npy')
emg_force = np.load('VARResults/EMG_Force_lagOrder.npy')

BIC = pd.DataFrame({'all': all[0,:],
                    'only_eeg': only_eeg[0,:],
                    'only_emg': only_emg[0,:],
                    'only_force': only_force[0,:],
                    'eeg_emg': eeg_emg[0,:],
                    'eeg_force': eeg_force[0,:],
                    'emg_force': emg_force[0,:]})
model_orders = 1 + np.arange(20)

sns.set(style="white", palette="muted", color_codes=True, font_scale=1.6)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 2, figsize=(10,12), sharex=False)
sns.despine(left=True)

sns.pointplot(x=model_orders, y=BIC['all'], color='r', ax=axes[0,0], scale=0.4).set_title('EEG, EMG and Force')
sns.pointplot(x=model_orders, y=BIC['eeg_force'], color='b', ax=axes[0,1], scale=0.4).set_title('EEG and Force')
sns.pointplot(x=model_orders, y=BIC['eeg_emg'], color='m', ax=axes[1,0], scale=0.4).set_title('EEG and EMG')
sns.pointplot(x=model_orders, y=BIC['emg_force'], color='g', ax=axes[1,1], scale=0.4).set_title('EMG and Force')
sns.pointplot(x=model_orders, y=BIC['only_eeg'], color='g', ax=axes[2,0], scale=0.4).set_title('Only EEG')
sns.pointplot(x=model_orders, y=BIC['only_force'], color='r', ax=axes[2,1], scale=0.4).set_title('Only Force')

for ax in axes.flatten():
    ax.set_ylabel('')
    ax.set_xlabel('')

# The rest is for some cosmetics of the plots...
# reduce the density of x-axis
for ax in axes.flatten():
    for ind, label in enumerate(ax.get_xticklabels()):
        if (ind+1) % 5 == 0: #every 5th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

# draw vertical lines:
for i in range(4,24,5):
    axes.flatten()[0].vlines(x=i, ymin=BIC['all'].min()-1, ymax=BIC['all'].max(), color='black', 
                             alpha=1, linewidth=1, linestyles='dotted')
    axes.flatten()[1].vlines(x=i, ymin=BIC['eeg_force'].min()-1, ymax=BIC['eeg_force'].max(), color='black', 
                             alpha=1, linewidth=1, linestyles='dotted')
    axes.flatten()[2].vlines(x=i, ymin=BIC['eeg_emg'].min()-0.6, ymax=BIC['eeg_emg'].max(), color='black', 
                             alpha=1, linewidth=1, linestyles='dotted')
    axes.flatten()[3].vlines(x=i, ymin=BIC['emg_force'].min()-0.1, ymax=BIC['emg_force'].max(), color='black', 
                             alpha=1, linewidth=1, linestyles='dotted')
    axes.flatten()[4].vlines(x=i, ymin=BIC['only_eeg'].min()-1.2, ymax=BIC['only_eeg'].max(), color='black', 
                             alpha=1, linewidth=1, linestyles='dotted')
    axes.flatten()[5].vlines(x=i, ymin=BIC['only_force'].min()-0.1, ymax=BIC['only_force'].max(), 
                             color='black', alpha=1, linewidth=1, linestyles='dotted')

# draw horizontal lines:
for i in range(int(BIC['all'].min()), int(BIC['all'].max()+1), 2):
    axes.flatten()[0].hlines(y=i, xmin=0, xmax=19, color='black', alpha=1, linewidth=1, linestyles='dotted')
for i in range(int(BIC['eeg_force'].min()), int(BIC['eeg_force'].max()+1)):
    axes.flatten()[1].hlines(y=i, xmin=0, xmax=19, color='black', alpha=1, linewidth=1, linestyles='dotted')
for i in range(int(BIC['eeg_emg'].min()), int(BIC['eeg_emg'].max()+1)):
    axes.flatten()[2].hlines(y=i, xmin=0, xmax=19, color='black', alpha=1, linewidth=1, linestyles='dotted')
for i in np.arange(BIC['emg_force'].min(), BIC['emg_force'].max(), 0.5):
    axes.flatten()[3].hlines(y=i, xmin=0, xmax=19, color='black', alpha=1, linewidth=1, linestyles='dotted')
for i in range(int(BIC['only_eeg'].min()), int(BIC['only_eeg'].max()+1)):
    axes.flatten()[4].hlines(y=i, xmin=0, xmax=19, color='black', alpha=1, linewidth=1, linestyles='dotted')
for i in np.arange(BIC['only_force'].min(), BIC['only_force'].max(), 0.5):
    axes.flatten()[5].hlines(y=i, xmin=0, xmax=19, color='black', alpha=1, linewidth=1, linestyles='dotted')

# display and save
f.text(0.5, 0.0, 'Model Order (P)', ha='center')
f.text(0.01, 0.5, 'BIC Score', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig("VARResults/lagOrders.pdf", bbox_inches="tight")
```


![png](spectralAnalysis_files/spectralAnalysis_27_0.png)


### Model Training and Multi-Step Ahead Prediction


```python
def show_entry_fields():
    global nPreds, nAhead
    nPreds, nAhead = int(e1.get()), int(e2.get())
    print("nPreds: %s\nnAhead: %s" % (nPreds, nAhead))
    root.destroy()

root = tk.Tk()

window_height = 80
window_width = 350

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))

root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
root.title('Prediction Parameters')

# total number of samples to be predicted one-by-one
tk.Label(root, text="num_predictions (max. 200)").grid(row=0)
# number of samples to be predicted at results.forecast case
tk.Label(root, text="num_ahead").grid(row=1)

e1 = tk.Entry(root)
e2 = tk.Entry(root)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)

tk.Button(root, text='Submit', command=show_entry_fields).grid(row=3, 
                                                               column=1, 
                                                               sticky=tk.W, 
                                                               pady=4)
tk.mainloop()
```

    nPreds: 200
    nAhead: 1
    


```python
P = 5 # optimal lag order based on BIC plots
use_saved = false 
# Use these figures by default:
# nPreds = 200 number of samples to be predicted one-by-one
# nAhead = 1 number of samples to be predicted at results.forecast case

def fit_model(train, test, cols, lag_order=P, nPreds=nPreds, nAhead=nAhead):
    assert nAhead <= nPreds, ("number of samples to be predicted at each prediction must be"
                              "less than total number of samples to be predicted")
    model = VAR(train)
    results = model.fit(lag_order)
    forecasted_trials = []
    for i, participant in enumerate(tqdm_notebook(test, ascii=True)):
        participant_forecast = []
        for trial in participant:
            forecasts = []
            # sliding window over last P observed samples to predict next nAhead samples
            for j in range(math.floor(nPreds/nAhead)):
                forecasts.extend(results.forecast(trial.iloc[j*nAhead:j*nAhead+lag_order,cols].values, nAhead))
            # convert the list to a pd.DataFrame
            forecasts = pd.DataFrame(forecasts, columns=[wl_columns[x] for x in cols])
            participant_forecast.append(forecasts)       
        forecasted_trials.append(participant_forecast)
    return forecasted_trials


if use_saved == False:
    # Only EEG (first 28 columns)   
    forecasted_eeg = fit_model(train_wavelet_df.iloc[:,eeg_ind_wl], 
                               test_wavelet, eeg_ind_wl)
    np.save('VARResults/forecasted_eeg.npy', forecasted_eeg)


    # Only EMG (28-35 columns)
    forecasted_emg = fit_model(train_wavelet_df.iloc[:,emg_ind_wl], 
                               test_wavelet, emg_ind_wl)
    np.save('VARResults/forecasted_emg.npy', forecasted_emg)


    # Only Force (35-41 columns)
    forecasted_force = fit_model(train_wavelet_df.iloc[:,force_ind_wl], 
                                 test_wavelet, force_ind_wl)
    np.save('VARResults/forecasted_force.npy', forecasted_force)


    # EEG & EMG
    forecasted_eegemg = fit_model(train_wavelet_df.iloc[:,eeg_ind_wl+emg_ind_wl], 
                                  test_wavelet, eeg_ind_wl+emg_ind_wl)
    np.save('VARResults/forecasted_eegemg.npy', forecasted_eegemg)


    # EEG & Force
    forecasted_eegforce = fit_model(train_wavelet_df.iloc[:,eeg_ind_wl+force_ind_wl], 
                                    test_wavelet, eeg_ind_wl+force_ind_wl)
    np.save('VARResults/forecasted_eegforce.npy', forecasted_eegforce)


    # EMG & Force
    forecasted_emgforce = fit_model(train_wavelet_df.iloc[:,emg_ind_wl+force_ind_wl], 
                                    test_wavelet, emg_ind_wl+force_ind_wl)
    np.save('VARResults/forecasted_emgforce.npy', forecasted_eegforce)


    # all
    forecasted_all = fit_model(train_wavelet_df.iloc[:,eeg_ind_wl+emg_ind_wl+force_ind_wl], 
                               test_wavelet, eeg_ind_wl+emg_ind_wl+force_ind_wl)
    np.save('VARResults/forecasted_all.npy', forecasted_all)
```


```python
if use_saved == True:
    forecasted_all = np.load('VARResults/forecasted_all.npy')
    forecasted_eeg = np.load('VARResults/forecasted_eeg.npy')
    forecasted_emg = np.load('VARResults/forecasted_emg.npy')
    forecasted_force = np.load('VARResults/forecasted_force.npy')
    forecasted_eegforce = np.load('VARResults/forecasted_eegforce.npy')
    forecasted_eegemg = np.load('VARResults/forecasted_eegemg.npy')
    forecasted_emgforce = np.load('VARResults/forecasted_emgforce.npy')

participant = 0
trial_id = 0
forecasted_all[participant][trial_id].head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cA_EEG-1</th>
      <th>cA_EEG-2</th>
      <th>cA_EEG-3</th>
      <th>cA_EEG-4</th>
      <th>cA_EEG-5</th>
      <th>cA_EEG-6</th>
      <th>cA_EEG-7</th>
      <th>cA_EEG-8</th>
      <th>cA_EEG-9</th>
      <th>cA_EEG-10</th>
      <th>...</th>
      <th>cA_EMG-3</th>
      <th>cD_EMG-3</th>
      <th>cA_EMG-4</th>
      <th>cD_EMG-4</th>
      <th>cA_Force-x</th>
      <th>cD_Force-x</th>
      <th>cA_Force-y</th>
      <th>cD_Force-y</th>
      <th>cA_Force-z</th>
      <th>cD_Force-z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-21.508919</td>
      <td>-6.562134</td>
      <td>-5.334276</td>
      <td>-12.519816</td>
      <td>-7.660932</td>
      <td>18.792467</td>
      <td>-8.004852</td>
      <td>-7.197331</td>
      <td>-7.644245</td>
      <td>-7.353781</td>
      <td>...</td>
      <td>44.817772</td>
      <td>-2.734210</td>
      <td>-4019.377539</td>
      <td>-5.542910</td>
      <td>-0.308927</td>
      <td>-0.042133</td>
      <td>-0.634253</td>
      <td>0.005554</td>
      <td>515.792485</td>
      <td>-0.013243</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-31.988362</td>
      <td>-9.679031</td>
      <td>-12.837952</td>
      <td>-14.955971</td>
      <td>-11.728261</td>
      <td>6.745457</td>
      <td>-16.777899</td>
      <td>-10.349562</td>
      <td>-10.761796</td>
      <td>-12.302932</td>
      <td>...</td>
      <td>0.151611</td>
      <td>-11.177749</td>
      <td>-3990.103306</td>
      <td>40.577191</td>
      <td>0.035784</td>
      <td>-0.025288</td>
      <td>-0.601149</td>
      <td>0.001181</td>
      <td>517.878683</td>
      <td>-0.027134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-43.024810</td>
      <td>-13.169176</td>
      <td>-23.323018</td>
      <td>-19.242330</td>
      <td>-16.214815</td>
      <td>-5.445994</td>
      <td>-31.684048</td>
      <td>-14.756533</td>
      <td>-15.055037</td>
      <td>-17.988735</td>
      <td>...</td>
      <td>79.340383</td>
      <td>19.482956</td>
      <td>-3988.189563</td>
      <td>1.550280</td>
      <td>-0.240876</td>
      <td>-0.022136</td>
      <td>-0.717007</td>
      <td>-0.000499</td>
      <td>519.958571</td>
      <td>-0.011604</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-53.045517</td>
      <td>-15.638729</td>
      <td>-34.744007</td>
      <td>-22.968621</td>
      <td>-20.691025</td>
      <td>-15.180897</td>
      <td>-46.128005</td>
      <td>-19.516757</td>
      <td>-19.205198</td>
      <td>-22.032503</td>
      <td>...</td>
      <td>109.806616</td>
      <td>2.000533</td>
      <td>-4042.333529</td>
      <td>-18.669508</td>
      <td>-0.296238</td>
      <td>0.028599</td>
      <td>-0.692788</td>
      <td>0.004719</td>
      <td>521.970803</td>
      <td>0.012616</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-62.623406</td>
      <td>-16.375638</td>
      <td>-45.902627</td>
      <td>-25.095084</td>
      <td>-24.016677</td>
      <td>-28.023513</td>
      <td>-59.231594</td>
      <td>-23.517937</td>
      <td>-20.968722</td>
      <td>-23.150697</td>
      <td>...</td>
      <td>97.732225</td>
      <td>4.857370</td>
      <td>-4035.816118</td>
      <td>31.923246</td>
      <td>-0.352592</td>
      <td>0.005702</td>
      <td>-0.689491</td>
      <td>0.001992</td>
      <td>523.947882</td>
      <td>-0.000218</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-66.332786</td>
      <td>-11.965188</td>
      <td>-49.628569</td>
      <td>-20.470753</td>
      <td>-22.133005</td>
      <td>-33.086566</td>
      <td>-63.909462</td>
      <td>-22.092892</td>
      <td>-16.550132</td>
      <td>-17.180603</td>
      <td>...</td>
      <td>56.981538</td>
      <td>-6.149926</td>
      <td>-3710.482551</td>
      <td>2.675647</td>
      <td>-0.340517</td>
      <td>0.033182</td>
      <td>-0.701706</td>
      <td>-0.001186</td>
      <td>526.119603</td>
      <td>-0.006669</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-61.068685</td>
      <td>-1.450935</td>
      <td>-43.934387</td>
      <td>-7.933298</td>
      <td>-13.179571</td>
      <td>-27.054523</td>
      <td>-57.269690</td>
      <td>-13.528102</td>
      <td>-4.588219</td>
      <td>-3.123703</td>
      <td>...</td>
      <td>-58.276771</td>
      <td>-9.134857</td>
      <td>-3900.650323</td>
      <td>-5.134375</td>
      <td>-0.198220</td>
      <td>0.030309</td>
      <td>-0.666655</td>
      <td>0.002718</td>
      <td>528.122846</td>
      <td>-0.007075</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-45.074769</td>
      <td>12.261109</td>
      <td>-26.917283</td>
      <td>11.521022</td>
      <td>2.911373</td>
      <td>-7.488500</td>
      <td>-37.428779</td>
      <td>2.210848</td>
      <td>10.992692</td>
      <td>14.255939</td>
      <td>...</td>
      <td>116.823949</td>
      <td>-2.245084</td>
      <td>-3848.823727</td>
      <td>58.929197</td>
      <td>-0.239838</td>
      <td>0.017090</td>
      <td>-0.614297</td>
      <td>0.001331</td>
      <td>530.054991</td>
      <td>-0.014366</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-20.815300</td>
      <td>32.357550</td>
      <td>-0.422202</td>
      <td>31.765939</td>
      <td>20.602819</td>
      <td>18.883646</td>
      <td>-6.804740</td>
      <td>20.424404</td>
      <td>34.490299</td>
      <td>38.988712</td>
      <td>...</td>
      <td>36.892764</td>
      <td>0.706485</td>
      <td>-3677.308831</td>
      <td>-6.021512</td>
      <td>-0.217936</td>
      <td>0.001892</td>
      <td>-0.486872</td>
      <td>0.001420</td>
      <td>531.974540</td>
      <td>0.011716</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5.100467</td>
      <td>49.816904</td>
      <td>24.272295</td>
      <td>51.309801</td>
      <td>40.115957</td>
      <td>50.436970</td>
      <td>21.073744</td>
      <td>40.313484</td>
      <td>53.463608</td>
      <td>57.741609</td>
      <td>...</td>
      <td>51.055343</td>
      <td>-4.011759</td>
      <td>-3718.408415</td>
      <td>28.577633</td>
      <td>-0.223299</td>
      <td>0.033979</td>
      <td>-0.628178</td>
      <td>0.000502</td>
      <td>534.026332</td>
      <td>-0.072954</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 28 columns</p>
</div>




```python
# Add to the forecasted dataframes detail coefficients of eeg 
# and then apply inverse discrete wavelet transform to extract the signal.
cols_to_app = ['cD_EEG-1', 'cD_EEG-2', 'cD_EEG-3', 'cD_EEG-4', 'cD_EEG-5', 'cD_EEG-6',
               'cD_EEG-7', 'cD_EEG-8', 'cD_EEG-9', 'cD_EEG-10', 'cD_EEG-11', 'cD_EEG-12',
               'cD_EEG-13', 'cD_EEG-14']

# add the detail coefficients of eeg the col after the app coefficients for each trial
for i in range(num_eeg_ch):
    [trial.insert(2*i+1, cols_to_app[i], 0) for participant in forecasted_all for trial in participant 
                                            if cols_to_app[i] not in trial] 
    [trial.insert(2*i+1, cols_to_app[i], 0) for participant in forecasted_eeg for trial in participant
                                            if cols_to_app[i] not in trial]
    [trial.insert(2*i+1, cols_to_app[i], 0) for participant in forecasted_eegemg for trial in participant 
                                            if cols_to_app[i] not in trial]
    [trial.insert(2*i+1, cols_to_app[i], 0) for participant in forecasted_eegforce for trial in participant 
                                            if cols_to_app[i] not in trial]
        
# list of dataframes, each as the complete forecasted dwt coefficients for a given modality
f_list = [forecasted_eeg, forecasted_emg, forecasted_force, forecasted_eegemg, 
          forecasted_eegforce, forecasted_emgforce, forecasted_all]

forecasted_eegemg[participant][trial_id].head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cA_EEG-1</th>
      <th>cD_EEG-1</th>
      <th>cA_EEG-2</th>
      <th>cD_EEG-2</th>
      <th>cA_EEG-3</th>
      <th>cD_EEG-3</th>
      <th>cA_EEG-4</th>
      <th>cD_EEG-4</th>
      <th>cA_EEG-5</th>
      <th>cD_EEG-5</th>
      <th>...</th>
      <th>cA_EEG-14</th>
      <th>cD_EEG-14</th>
      <th>cA_EMG-1</th>
      <th>cD_EMG-1</th>
      <th>cA_EMG-2</th>
      <th>cD_EMG-2</th>
      <th>cA_EMG-3</th>
      <th>cD_EMG-3</th>
      <th>cA_EMG-4</th>
      <th>cD_EMG-4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-21.516644</td>
      <td>0</td>
      <td>-6.563316</td>
      <td>0</td>
      <td>-5.331854</td>
      <td>0</td>
      <td>-12.515802</td>
      <td>0</td>
      <td>-7.659147</td>
      <td>0</td>
      <td>...</td>
      <td>6.445467</td>
      <td>0</td>
      <td>386.553259</td>
      <td>8.311971</td>
      <td>-4270.200771</td>
      <td>-24.083689</td>
      <td>43.364625</td>
      <td>-2.704654</td>
      <td>-4020.318876</td>
      <td>-6.190649</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-31.998137</td>
      <td>0</td>
      <td>-9.681635</td>
      <td>0</td>
      <td>-12.839294</td>
      <td>0</td>
      <td>-14.955448</td>
      <td>0</td>
      <td>-11.730483</td>
      <td>0</td>
      <td>...</td>
      <td>5.118672</td>
      <td>0</td>
      <td>390.426199</td>
      <td>10.853298</td>
      <td>-4227.469585</td>
      <td>28.194758</td>
      <td>-1.349326</td>
      <td>-11.154912</td>
      <td>-3990.998708</td>
      <td>39.926478</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-43.037096</td>
      <td>0</td>
      <td>-13.174465</td>
      <td>0</td>
      <td>-23.327694</td>
      <td>0</td>
      <td>-19.247428</td>
      <td>0</td>
      <td>-16.220325</td>
      <td>0</td>
      <td>...</td>
      <td>-0.219769</td>
      <td>0</td>
      <td>330.679609</td>
      <td>-9.176490</td>
      <td>-4218.799801</td>
      <td>25.450791</td>
      <td>77.714244</td>
      <td>19.517425</td>
      <td>-3989.156924</td>
      <td>0.897690</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-53.057837</td>
      <td>0</td>
      <td>-15.644725</td>
      <td>0</td>
      <td>-34.748626</td>
      <td>0</td>
      <td>-22.975205</td>
      <td>0</td>
      <td>-20.696500</td>
      <td>0</td>
      <td>...</td>
      <td>-2.868747</td>
      <td>0</td>
      <td>437.757869</td>
      <td>26.264314</td>
      <td>-4368.873946</td>
      <td>-57.949230</td>
      <td>108.141678</td>
      <td>2.035797</td>
      <td>-4043.304145</td>
      <td>-19.314422</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-62.635526</td>
      <td>0</td>
      <td>-16.380625</td>
      <td>0</td>
      <td>-45.907068</td>
      <td>0</td>
      <td>-25.101442</td>
      <td>0</td>
      <td>-24.021728</td>
      <td>0</td>
      <td>...</td>
      <td>-8.561150</td>
      <td>0</td>
      <td>390.207086</td>
      <td>5.845338</td>
      <td>-4332.321886</td>
      <td>25.577013</td>
      <td>96.032562</td>
      <td>4.892747</td>
      <td>-4036.842612</td>
      <td>31.271390</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
eeg_ind = list(range(0,14))
emg_ind = list(range(14,18))
force_ind = list(range(18,21))
c_list = [[columns[i] for i in eeg_ind], 
          [columns[i] for i in emg_ind], 
          [columns[i] for i in force_ind],
          [columns[i] for i in eeg_ind + emg_ind],
          [columns[i] for i in eeg_ind + force_ind], 
          [columns[i] for i in emg_ind + force_ind],
          [columns[i] for i in eeg_ind + emg_ind + force_ind]]

# apply inverse discrete wavelet transform to each trial and extract the signals:
def apply_idwt(f, c):
    all_forecasts = []
    for participant in f:
        participant_forecasts = []
        for trial in participant:
            forecast = pd.DataFrame(columns=c)
            for i in range(len(forecast.columns)):
                forecast.iloc[:,i] = pywt.idwt(trial.iloc[:,2*i], 
                                               trial.iloc[:,2*i+1], 'db4')[:-1]
            participant_forecasts.append(forecast)
        all_forecasts.append(participant_forecasts)
    return all_forecasts

# s_list is a list of list of dataframes, each one is the predicted signal for a specific modality
s_list = []
# wl_tuple is the tuple of forecasted dwl coeffs and columns is corresponding modality dataframe's columns
for f,c in zip(f_list, c_list):
    s_list.append(apply_idwt(f,c))
```


```python
# Select an input modality to plot the predictions:
results_for_eeg = 0
results_for_eegemg = 3
results_for_eegforce = 4
results_for_all = -1

def getit():
    global value
    value = v.get()
    
def printit():
    try:
        global value
        if value==0:
            print("Only EEG Selected")
            root.destroy()
        elif value==3:
            print("EEG & EMG Selected")
            root.destroy()
        elif value==4:
            print("EEG & Force Selected")
            root.destroy()
        elif value==-1:
            print("EEG, EMG & Force Selected")
            root.destroy()
        del value
    except NameError as e:
        value = 0 
        
root = tk.Tk()

window_height = 100
window_width = 500

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))

root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
root.title('Modality Selection')
v = tk.IntVar(value=0) # start with the values of one of the buttons
tk.Label(root, text="""Choose an input modality to plot the predictions""",
        justify = tk.LEFT, padx=20).pack()
rb1 = tk.Radiobutton(root, text="Only EEG", padx=20, variable=v, value=0, command=getit).pack(anchor=tk.W)
rb2 = tk.Radiobutton(root, text="EEG & EMG", padx=20, variable=v, value=3, command=getit).pack(anchor=tk.W)
rb3 = tk.Radiobutton(root, text="EEG & Force", padx=20, variable=v, value=4, command=getit).pack(anchor=tk.W)
rb4 = tk.Radiobutton(root, text="All", padx=20, variable=v, value=-1, command=getit).pack(anchor=tk.W)
b = tk.Button(root, text="Submit", command=printit).pack(anchor=tk.W)

root.mainloop()
```

    EEG, EMG & Force Selected
    

### Visualizing Predictions Over Test Data in Frequency Domain


```python
patient_id = 4
trial_id = 98

# Plot the predictions in frequency domain given thresholded DWT coeffs for 
# different combinations of eeg+emg+force:
modality = v.get()

sns.set(style="white", palette="muted", color_codes=True, font_scale=1.5)
# Set up the matplotlib figure (all forecasts subplot figure)
f, axes = plt.subplots(5, 3, figsize=(14, 24), sharex=False, sharey=False)
sns.despine()

x_gt=np.arange(0, (nPreds+P)/fs, 1/fs)
x_pred=np.arange(P/fs, (nPreds+P)/fs, 1/fs)

T = math.floor(10*(nPreds+P)/fs) + 1

if T != 1:
    z = [decimal.Decimal(i) / decimal.Decimal(10) for i in range(0, T)]
else:
    T = math.floor(100*(nPreds+P)/fs) + 1
    z = [decimal.Decimal(i) / decimal.Decimal(100) for i in range(0, T, 2)]

plt.setp(axes, xticks=z, xticklabels=z)

for j in range(1,15,1):
    sns.set()
    axes[math.floor((j-1)/3),(j-1)%3].plot(x_gt, test_wavelet[patient_id][trial_id].iloc[:nPreds+P, 2*(j-1)].values, 
                                           color='b', label="True")
    axes[math.floor((j-1)/3), (j-1)%3].plot(x_pred, f_list[modality][patient_id][trial_id].iloc[:nPreds, 2*(j-1)].values, 
                                            color='r', linestyle='--', label="Prediction")
    axes[math.floor((j-1)/3), (j-1)%3].axvline(x=P/fs, linestyle='--', color='k')
    axes[math.floor((j-1)/3), (j-1)%3].title.set_text("cA EEG-%i" %j)
    axes[math.floor((j-1)/3), (j-1)%3].set_yticks(axes[math.floor((j-1)/3), (j-1)%3].get_yticks()[1:-1:2])
    axes[0, 0].legend(frameon=False)
    plt.setp(axes[0, 0].get_legend().get_texts(), fontsize='14')

if modality in [results_for_all, results_for_eegforce]: #force coefficient predictions exist
    axes[4,2].plot(x_gt, test_wavelet[patient_id][trial_id].iloc[:nPreds+P, -2].values, 
                   color='b', label="True")
    axes[4,2].plot(x_pred, f_list[modality][patient_id][trial_id].iloc[:nPreds, -2].values, color='r', 
                   linestyle='--', label="Prediction")
    axes[4,2].axvline(x=P/fs, linestyle='--', color='k')
    axes[4,2].ticklabel_format(axis='y', style='sci', scilimits=(4,-4))
    axes[4,2].title.set_text("cA Force-z")
    axes[4,2].set_yticks(axes[4, 2].get_yticks()[1:-1:2])
    #axes[4,2].legend() 

plt.savefig("VARResults/predictions_waveletDomain.pdf", bbox_inches="tight")
```


![png](spectralAnalysis_files/spectralAnalysis_36_0.png)


### Visualizing Predictions Over Test Data in Time Domain

First P (lag order) discrete wavelet coefficients correspond to first 2*(P-3) time samples. There is no time-series prediction within this time frame.


```python
# Plot the predictions in time domain given thresholded DWT coeffs 
# for different combinations of eeg+emg+force:
 
sns.set(style="white", palette="muted", color_codes=True, font_scale=1.5)
# Set up the matplotlib figure (all forecasts subplot figure)
f, axes = plt.subplots(5, 3, figsize=(14, 24), sharex=False, sharey=False)
sns.despine()

plt.setp(axes, xticks=z, xticklabels=z)

for j in range(1,15,1):
    sns.set()
    axes[math.floor((j-1)/3),(j-1)%3].plot(x_gt, test_data[patient_id][trial_id].iloc[2*(P-3):2*(P-3)+nPreds+P, 
                                                                          j-1].values, color='b', label="True")
    axes[math.floor((j-1)/3), (j-1)%3].plot(x_pred, 
                                            s_list[modality][patient_id][trial_id].iloc[:nPreds,j-1].values, 
                                            color='r', linestyle='--', label="Prediction")
    axes[math.floor((j-1)/3), (j-1)%3].axvline(x=P/fs, linestyle='--', color='k')
    axes[math.floor((j-1)/3), (j-1)%3].title.set_text("EEG-%i" %j)
    axes[math.floor((j-1)/3), (j-1)%3].set_yticks(axes[math.floor((j-1)/3), (j-1)%3].get_yticks()[1:-1:2])
    axes[0, 0].legend(frameon=False)
    plt.setp(axes[0, 0].get_legend().get_texts(), fontsize='14')

if modality in [results_for_all, results_for_eegforce]: #force predictions exist
    axes[4,2].plot(x_gt, test_data[patient_id][trial_id].iloc[2*(P-3):2*(P-3)+nPreds+P, -1].values, 
                   color='b', label="True")
    axes[4,2].plot(x_pred, s_list[modality][patient_id][trial_id].iloc[:nPreds, -1].values, 
                   color='r', linestyle='--', label="Prediction")
    axes[4,2].axvline(x=P/fs, linestyle='--', color='k')
    axes[4,2].ticklabel_format(axis='y', style='sci', scilimits=(4,-4))
    axes[4,2].title.set_text("Force-z")
    axes[4,2].set_yticks(axes[4, 2].get_yticks()[1:-1:2])
    #axes[4,2].legend() 

plt.savefig("VARResults/predictions_timeDomain.pdf", bbox_inches="tight")
```


![png](spectralAnalysis_files/spectralAnalysis_39_0.png)


### Displaying Goodness of Fit Performance for Different Input Combinations


```python
'''
Return the mean absolute percentage error between the true signal and prediction
https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
'''
def find_mape(true, pred):
    return 100 * np.mean(np.abs((true-pred)/true))

# evaluate and save the rmse for each patient:
for participant_id in tqdm_notebook(range(num_participants), ascii=True):
    ###### RMSE for only EEG prediction ######
    eeg_errs = []
    for i, test_trial in enumerate(test_data[participant_id]):
        sum_eeg_err = 0
        for j in eeg_ind:
            # i, which is 1 sample for each trial,
            # because the reconstructed signal is 1 sample longer than the original
            sum_eeg_err += np.sqrt(mse(test_trial.iloc[2*(P-3)+P:2*(P-3)+nPreds+P, j].values, 
                                   s_list[0][participant_id][i].iloc[:nPreds, j].values))
        eeg_errs.append(sum_eeg_err / num_eeg_ch)

    # Save the RMSE as a dataframe:
    err_df = pd.DataFrame({'eeg': np.asarray(eeg_errs)})
    err_df.to_pickle("VARResults/err_bars_forEEGOnly_id%i.pkl" %participant_id)


    ###### RMSE for only FORCE prediction ######
    force_errs = []
    for i, test_trial in enumerate(test_data[participant_id]):
        force_errs.append(np.sqrt(mse(test_trial.iloc[2*(P-3)+P:2*(P-3)+nPreds+P, -1].values, 
                                      s_list[2][participant_id][i].iloc[:nPreds, -1].values)))

    # Save the RMSE as a dataframe:
    err_df = pd.DataFrame({'force': np.asarray(force_errs)})
    err_df.to_pickle("VARResults/err_bars_forForceOnly_id%i.pkl" %participant_id)


    ###### RMSE for EEG and FORCE prediction ######
    eeg_errs, force_errs = [], []
    for i, test_trial in enumerate(test_data[participant_id]):
        sum_eeg_err = 0
        for j in eeg_ind:
            # i, which is 1 sample for each trial,
            # because the reconstructed signal is 1 sample longer than the original
            sum_eeg_err += np.sqrt(mse(test_trial.iloc[2*(P-3)+P:2*(P-3)+nPreds+P, j].values, 
                                       s_list[4][participant_id][i].iloc[:nPreds, j].values))
        eeg_errs.append(sum_eeg_err / num_eeg_ch)   
        force_errs.append(np.sqrt(mse(test_trial.iloc[2*(P-3)+P:2*(P-3)+nPreds+P, -1].values, 
                                      s_list[4][participant_id][i].iloc[:nPreds, -1].values))) 

    # Save the RMSE as a dataframe:
    err_df = pd.DataFrame({'eeg': np.asarray(eeg_errs),
                           'force': np.asarray(force_errs)})
    err_df.to_pickle("VARResults/err_bars_forEEGandForce_id%i.pkl" %participant_id)


    ###### RMSE for EMG and FORCE prediction ######
    force_errs = []
    for i, test_trial in enumerate(test_data[participant_id]):
        force_errs.append(np.sqrt(mse(test_trial.iloc[2*(P-3)+P:2*(P-3)+nPreds+P, -1].values, 
                                      s_list[5][participant_id][i].iloc[:nPreds, -1].values))) 

    # Save the RMSE as a dataframe:
    err_df = pd.DataFrame({'force': np.asarray(force_errs)})
    err_df.to_pickle("VARResults/err_bars_forEMGandForce_id%i.pkl" %participant_id)


    # RMSE for EEG and EMG prediction
    eeg_errs = []
    for i, test_trial in enumerate(test_data[participant_id]):
        sum_eeg_err = 0
        for j in eeg_ind:
            sum_eeg_err += np.sqrt(mse(test_trial.iloc[2*(P-3)+P:2*(P-3)+nPreds+P, j].values, 
                                       s_list[3][participant_id][i].iloc[:nPreds, j].values))
        eeg_errs.append(sum_eeg_err / num_eeg_ch)

    # Save the RMSE as a dataframe:
    err_df = pd.DataFrame({'eeg': np.asarray(eeg_errs)})
    err_df.to_pickle("VARResults/err_bars_forEEGandEMG_id%i.pkl" %participant_id)


    # RMSE for ALL modalities (EEG, EMG and FORCE prediction)
    eeg_errs, force_errs = [], []
    for i, test_trial in enumerate(test_data[participant_id]):
        sum_eeg_err = 0
        for j in eeg_ind:
            # i, which is 1 sample for each trial,
            # because the reconstructed signal is 1 sample longer than the original
            sum_eeg_err += np.sqrt(mse(test_trial.iloc[2*(P-3)+P:2*(P-3)+nPreds+P, j].values, 
                                       s_list[-1][participant_id][i].iloc[:nPreds, j].values))
        eeg_errs.append(sum_eeg_err / num_eeg_ch)   
        force_errs.append(np.sqrt(mse(test_trial.iloc[2*(P-3)+P:2*(P-3)+nPreds+P, -1].values, 
                                      s_list[-1][participant_id][i].iloc[:nPreds, -1].values)))

    # Save the RMSE as a dataframe:
    err_df = pd.DataFrame({'eeg': np.asarray(eeg_errs),
                           'force': np.asarray(force_errs)})
    err_df.to_pickle("VARResults/err_bars_forAll_id%i.pkl" %participant_id)
```


```python
onlyEEG_errs = []
onlyForce_errs = []
EEGandForce_errs = []
EMGandForce_errs = []
EEGandEMG_errs = []
all_errs = []
force_box_plot = []
eeg_box_plot = []

for i in range(num_participants):
    onlyEEG_errs.append(pd.read_pickle("VARResults/err_bars_forEEGOnly_id%i.pkl" %i))
    onlyForce_errs.append(pd.read_pickle("VARResults/err_bars_forForceOnly_id%i.pkl" %i))
    EEGandForce_errs.append(pd.read_pickle("VARResults/err_bars_forEEGandForce_id%i.pkl" %i))
    EMGandForce_errs.append(pd.read_pickle("VARResults/err_bars_forEMGandForce_id%i.pkl" %i))
    EEGandEMG_errs.append(pd.read_pickle("VARResults/err_bars_forEEGandEMG_id%i.pkl" %i))
    all_errs.append(pd.read_pickle("VARResults/err_bars_forAll_id%i.pkl" %i))

    force_box_plot.append(pd.DataFrame({'Only Force': onlyForce_errs[-1]['force'].values, 
                                        'EEG & Force': EEGandForce_errs[-1]['force'].values, 
                                        'Force & EMG': EMGandForce_errs[-1]['force'].values, 
                                        'All': all_errs[-1]['force'].values}))
    eeg_box_plot.append(pd.DataFrame({'Only EEG': onlyEEG_errs[-1]['eeg'].values,
                                      'EEG & Force': EEGandForce_errs[-1]['eeg'].values,
                                      'EEG & EMG': EEGandEMG_errs[-1]['eeg'].values,
                                      'All': all_errs[-1]['eeg'].values}))

sns.set(style="whitegrid", font_scale=1)
# Set up the matplotlib figure
f, axes = plt.subplots(num_participants, 2, figsize=(8, 24), sharex=False)
sns.despine(left=True)

sns.boxplot(data=eeg_box_plot[i], showfliers=False, ax=axes[0,0]).set_title('RMSE: EEG Predictions')
sns.boxplot(data=force_box_plot[i], showfliers=False, ax=axes[0,1]).set_title('RMSE: Force Predictions')
axes[0,0].set_ylabel('User %s' %str(1))

for i in range(1, num_participants):
    sns.boxplot(data=eeg_box_plot[i], showfliers=False, ax=axes[i,0])
    sns.boxplot(data=force_box_plot[i], showfliers=False, ax=axes[i,1])
    axes[i,0].set_ylabel('User %s' %str(i+1))

for idx, ax in enumerate(axes.flatten()):
    if idx < 2*(num_participants-1): ax.set_xticklabels([])
    else: ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")  
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

path = "VARResults/error_boxes.pdf"
plt.savefig(path, bbox_inches="tight")
```


![png](spectralAnalysis_files/spectralAnalysis_42_0.png)


# Continuous Wavelet Transform and Scaleograms


```python
wavelet_type = 'morl'
fig, axes = plt.subplots(3, figsize=(12,20))
fig.subplots_adjust(hspace=0.3)
for i, signal in enumerate([eeg_signal_1, emg_signal_1, force_signal_3]):
    scales = np.arange(1,len(eeg_signal_1+1))
    coef_matrix, freqs = pywt.cwt(signal, scales, wavelet_type, sampling_period=dt)
    power = (abs(coef_matrix)) ** 2
    levels = [0.0078125, 0.015625 ,0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    contourlevels = np.log2(levels)
    im = axes[i].contourf(time, freqs, np.log2(power), contourlevels, 
                          extend='both',cmap=plt.cm.seismic)
    #axes[i].matshow(coef_matrix) 
    axes[i].set_ylabel('Approximate Frequency [Hz]', fontsize=16)
    axes[i].set_xlim(time.min(), time.max())
    if i == 0:
        axes[i].set_title('CWT with Time vs Frequency - EEG Channel 1', fontsize=16)
    elif i == 1:
        axes[i].set_title('CWT with Time vs Frequency - EMG Channel 1', fontsize=16)
    else:
        axes[i].set_title('CWT with Time vs Frequency - Force Channel 3', fontsize=16)
        axes[i].set_xlabel('Time (sec)', fontsize=16)
    yticks = np.arange(0, round(freqs.max()/100)*100, step=100)
    axes[i].set_yticks(yticks)

plt.show()
```


![png](spectralAnalysis_files/spectralAnalysis_44_0.png)

