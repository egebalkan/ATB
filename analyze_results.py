import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import math
import sys

#%%i_beam_perpendicular_task_2
directory_2_1 = r"D:\GIT STUFF\task-clustering\results\PERFORMANCE\i_beam\perpendicular_2\test_1\performance.csv"
directory_2_2 = r"D:\GIT STUFF\task-clustering\results\PERFORMANCE\i_beam\perpendicular_2\test_2\performance_2.csv"
#fileName     = "\i_beam_perpendicular.txt"
#fname = directory + fileName
#%%
df_2_1 = pd.read_csv (directory_2_1, header = None)
df_2_2 = pd.read_csv (directory_2_2, header = None)
#print(df)
#%%
resolution_2_1   = np.array(df_2_1[1])[:31]
time_elapsed_2_1 = np.array(df_2_1[2])[:31]
reached_2_1      = np.array(df_2_1[3])[:31]
num_bases_2_1    = np.array(df_2_1[4])[:31]

resolution_2_2   = np.array(df_2_2[1])[:31]
time_elapsed_2_2 = np.array(df_2_2[2])[:31]
reached_2_2      = np.array(df_2_2[3])[:31]
num_bases_2_2    = np.array(df_2_2[4])[:31]

p_2_1= np.polyfit(resolution_2_1, time_elapsed_2_1, 4)
fit_2_1 = np.polyval(p_2_1, resolution_2_1)

p_2_2= np.polyfit(resolution_2_2, time_elapsed_2_2, 4)
fit_2_2 = np.polyval(p_2_2, resolution_2_2)
#%%

plt.scatter(resolution_2_1, time_elapsed_2_1, color = 'blue', alpha = 0.5)
plt.plot(resolution_2_1, fit_2_1, color = 'red', label = 'test #1', linewidth = 1)

#plt.scatter(resolution_2_2, time_elapsed_2_2, color = 'red', alpha = 0.5)
#plt.plot(resolution_2_2, fit_2_2, color = 'red', label = 'test #2', linewidth = 1)

plt.title("Resolution vs Time for task: i_beam_perpendicular_task_2")
plt.ylabel("Time Elapsed [sec]")
plt.xlabel("Search Space Resolution")

#%%






#%%
directory_1_1 = r"D:\GIT STUFF\task-clustering\results\PERFORMANCE\i_beam\perpendicular\test_1\performance.csv"
#directory_1_2 = r"D:\GIT STUFF\task-clustering\results\PERFORMANCE\i_beam\perpendicular_2\test_2\performance_2.csv"
#fileName     = "\i_beam_perpendicular.txt"
#fname = directory + fileName
#%%
df_1_1 = pd.read_csv (directory_1_1, header = None)
#df_1_2 = pd.read_csv (directory_1_2, header = None)
#print(df)
#%%
resolution_1_1   = np.array(df_1_1[1])
time_elapsed_1_1 = np.array(df_1_1[2])
reached_1_1      = np.array(df_1_1[3])
num_bases_1_1    = np.array(df_1_1[4])

#resolution_2   = np.array(df_2[1])[:31]
#time_elapsed_2 = np.array(df_2[2])[:31]
#reached_2      = np.array(df_2[3])[:31]
#num_bases_2    = np.array(df_2[4])[:31]

p_1_1= np.polyfit(resolution_1_1, time_elapsed_1_1, 4)
fit_1_1 = np.polyval(p_1_1, resolution_1_1)

#p_2= np.polyfit(resolution_2, time_elapsed_2, 4)
#fit_2 = np.polyval(p_2, resolution_2)
#%%

plt.scatter(resolution_1_1, time_elapsed_1_1, color = 'blue', alpha = 0.5)
plt.plot(resolution_1_1, fit_1_1, color = 'blue', label = 'test #1', linewidth = 1)

#plt.scatter(resolution_1_2, time_elapsed_1_2, color = 'red', alpha = 0.5)
#plt.plot(resolution_1_2, fit_1_2, color = 'red', label = 'test #2', linewidth = 1)

plt.title("Resolution vs Time for task: i_beam_perpendicular_task")
plt.ylabel("Time Elapsed [sec]")
plt.xlabel("Search Space Resolution")

#%%
plt.plot(resolution_1_1, fit_1_1, color = 'blue', label = '223 points', linewidth = 1)
plt.plot(resolution_2_1[:14], fit_2_1[:14], color = 'red', label = '119 points', linewidth = 1)

plt.title("Comparison based on number of task points")
plt.ylabel("Time Elapsed [sec]")
plt.xlabel("Search Space Resolution")

#%%
directory = r"D:\GIT STUFF\task-clustering\results\PERFORMANCE\i_beam\perpendicular_2\test_19\performance_2.csv"
df = pd.read_csv(directory,header = None)
res = list(df[1])
time = list(df[2])
score = list(df[7])

roots = []

for i in res:
    roots.append(np.sqrt(i))
#%%

poly = np.polyfit(roots,score,5)
fit =np.polyval(poly, roots)

poly2 = np.polyfit(roots,time, 3)
fit2 = np.polyval(poly2,roots)

#plt.scatter(res,score)
#plt.scatter(roots,time)
plt.plot(roots,fit,color = 'blue', linewidth = 1.5)
plt.plot(roots,fit2,linestyle = 'dotted', color = "red",linewidth = 2)
plt.xlabel("Seach space reoslution")
plt.ylabel("Time [s]")
plt.title("Time vs Resolution")

#%%

percent_change = []

for i in range(len(score)):
    if score[i] == 0:
        continue
    else:
        diff = 100*(score[i]-score[i-1])/score[i]
        percent_change.append( float('%.4g' % diff))  
for i in percent_change:
    if i > 0 and i< 0.5:
        print(i)
        

#%%
directory = r"D:\GIT STUFF\task-clustering\results\PERFORMANCE\i_beam\perpendicular_2\test_21\performance_2.csv"
        
df = pd.read_csv(directory,header = None)[6:17]
res = list(df[1])
time = list(df[2])
score = list(df[7])       

torq_dir = r"D:\GIT STUFF\task-clustering\results\PERFORMANCE\i_beam\perpendicular_2\test_torque_2\performance.csv"

        
df2 = pd.read_csv(torq_dir,header = None)
res2 = list(df2[1])
time2 = list(df2[2])
score2 = list(df2[7])    
        