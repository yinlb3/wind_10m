import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PyEMD import EMD, EEMD, CEEMDAN

# t = np.linspace(start=0, stop=1, num=200)
# s = np.cos(11 * 2 * np.pi * t * t) + 6 * t * t
filepath = r'D:\新能源\CMA_WSP2.0_WSPD_2022_point3-4\202206\20220601\20220601_sta3_Wind_Guizhou_grid13_v2.csv'
df = pd.read_csv(filepath_or_buffer=filepath, low_memory=False)
t = np.array(df.index)
c = 't2'
s = np.array(df.loc[:, c])

model = EMD()
# model = EEMD()
# model = CEEMDAN()
imfs = model(s, t)
n = imfs.shape[0] + 1

plt.subplot(n, 1, 1)
plt.plot(t, s, color='red')
# plt.title('Input signal: $S(t)=cos(22\pi t^2) + 6t^2$')
plt.title(c)
plt.xlabel('Time [s]')
for i, imf in enumerate(imfs):
    plt.subplot(n, 1, i + 2)
    plt.plot(t, imf, color='green')
    plt.title('IMF ' + str(i + 1))
    plt.xlabel('Time [s]')
plt.show()
