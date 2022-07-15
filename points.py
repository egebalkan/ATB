import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd




# =============================================================================
# extension = 'csv'
# 
# all_filenames = all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# 
# 
# #combine all files in the list
# combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
# #export to csv
# combined_csv.to_csv( "combined_csv.csv")
# 
# 
# =============================================================================
x = pd.read_csv("merged.csv", usecols =[0], header=None)
y = pd.read_csv("merged.csv", usecols =[1], header=None)
z = pd.read_csv("merged.csv", usecols =[2], header=None)

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection ='3d')
ax.scatter(x,z,y)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()