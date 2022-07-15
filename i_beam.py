import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from   scipy.spatial import distance
import time
import copy
import os
from   itertools import combinations
import sys
from tqdm import tqdm
import csv
import roboticstoolbox as rtb
import telegram_send

sys.path.append(r"D:\GIT STUFF\task-clustering")
from autonomous_task_base import atb

#%%USER INPUTS
#path of the task file on your computer


#test = "torque_2"
#name = "i_beam/perpendicular_2"
#time_limit = 6 #hours
#
#folder = "D:/GIT STUFF/task-clustering/results/PERFORMANCE/"+name+"/test_"+str(test)
#if os.path.exists(folder) == True:
#    print("THIS PATH ALREADY EXISTS! RECONFIGURE YOUR PATH!")
#    sys.exit()
#else:
#    os.mkdir(folder)
#%%  NOTIFICATIONS
#since the complexity of the calculation increases exponentially, it becomes
#unreasonable to sit in front of the computer for hours while iterating through
#ever increasing search space resolutions. that is why i decided to set up a 
#telegram bot that sends a message to your telegram number when the run is finished
#and also when an error occurs that stops the iterations.  This way you can keep
#easier track of when your run starts/ends and more importantly you don't have to
#babysit a computer for hours.

#since i'm not trying to solve the issues and rather notify that there is an 
#issue, i'm simply catching all exceptions. this is why the entire while loop 
#is in a try-except block, even though it is considered bad practice.

#guide on how to setup the telegram bot:
#https://medium.com/@robertbracco1/how-to-write-a-telegram-bot-to-send-messages-with-python-bcdf45d0a580

#telegram_send.send(messages=["Test "+str(test)+" started"])

#%%
dividend = 2
performance_data = {}
product1 = os.path.abspath(r"D:\GIT STUFF\task-clustering\tasks\i_beam\perpendicular\i_perpendicular_product1.csv")
product2 = os.path.abspath(r"D:\GIT STUFF\task-clustering\tasks\i_beam\perpendicular\i_perpendicular_product2.csv")
task     = os.path.abspath(r"D:\GIT STUFF\task-clustering\tasks\i_beam\perpendicular\i_perpendicular_task_2.csv")
                             
robot_coverage = rtb.models.DH.UR10()
robot_discrete = rtb.models.DH.UR10()
agv = 690 

buffer_in  = 300
buffer_out = 1250

robot_max_reach_coverage = 1.3
robot_max_reach_discrete = 1.3
#START

total_test_time = 0
time_start = time.time()

x_product_1 = pd.read_csv(product1, usecols =[0], header=None)
y_product_1 = pd.read_csv(product1, usecols =[2], header=None)
z_product_1 = pd.read_csv(product1, usecols =[1], header=None)

df_product1 = pd.read_csv(product1, header=None)
dropped_product1 = df_product1[[0,2,1]]
point_data_product1 = dropped_product1.values.tolist() 

x_product_2 = pd.read_csv(product2, usecols =[0], header=None)
y_product_2 = pd.read_csv(product2, usecols =[2], header=None)
z_product_2 = pd.read_csv(product2, usecols =[1], header=None)

df_product2 = pd.read_csv(product2, header=None)
dropped_product2 = df_product2[[0,2,1]]
point_data_product2 = dropped_product2.values.tolist() 

df_task = pd.read_csv(task, header=None)[::dividend]
flags = df_task[3].values.tolist()
point_data_task = df_task[[0,2,1]].values.tolist()
point_data_task_2d = df_task[[0,2]].values.tolist()

point_data_discrete = []
point_data_discrete_2d = []
point_data_coverage = []

for i in range(len(flags)):
    if flags[i] == "c":
        point_data_coverage.append(point_data_task[i])
    elif flags[i] == "d":
        point_data_discrete.append(point_data_task[i])
print(len(point_data_coverage))
        #%
#% bounding boxes for that products
Z1 = np.array(point_data_product1)
Z2 = np.array(point_data_product2)
verts1 = [[Z1[0],  Z1[1], Z1[2], Z1[3]],
          [Z1[4],  Z1[5], Z1[6], Z1[7]],
          [Z1[8],  Z1[9], Z1[10],Z1[11]],
          [Z1[12], Z1[13],Z1[14],Z1[15]],
          
          [Z1[16], Z1[17],Z1[18],Z1[19]],
          [Z1[20], Z1[21],Z1[22],Z1[23]],
          [Z1[20], Z1[21],Z1[10],Z1[11]],
          [Z1[8],  Z1[9], Z1[22],Z1[23]],
          
          [Z1[12], Z1[15],Z1[2], Z1[1]],
          [Z1[13], Z1[14],Z1[3], Z1[0]],
          [Z1[21], Z1[22],Z1[9], Z1[10]],
          [Z1[20], Z1[23],Z1[8], Z1[11]],
          [Z1[0],  Z1[1],Z1[12],Z1[13]],
          [Z1[2],  Z1[3],Z1[14],Z1[15]],
          [Z1[4],  Z1[17],Z1[18],Z1[7]],
          [Z1[5],  Z1[6], Z1[19],Z1[16]],
]

verts2 = [[Z2[0], Z2[1], Z2[2], Z2[3]],
          [Z2[4], Z2[5], Z2[6], Z2[7]],
          [Z2[8], Z2[9], Z2[10],Z2[11]],
          [Z2[12],Z2[13],Z2[14],Z2[15]],
          
          [Z2[16],Z2[17],Z2[18],Z2[19]],
          [Z2[20],Z2[21],Z2[22],Z2[23]],
          [Z2[4], Z2[7], Z2[18],Z2[17]],
          [Z2[16],Z2[19],Z2[6], Z2[5]],
          [Z2[0], Z2[1], Z2[12],Z2[13]],
          [Z2[0], Z2[3], Z2[14],Z2[13]],
          [Z2[12],Z2[15],Z2[2], Z2[1]],
          [Z2[9], Z2[10],Z2[21],Z2[22]],
          [Z2[8], Z2[11],Z2[20],Z2[23]],
          [Z2[8], Z2[9], Z2[22],Z2[23]],
          [Z2[10],Z2[11],Z2[20],Z2[21]],
]

bounding_box1 = []
line1 = np.linspace(Z1[9], Z1[10],10)
line2 = np.linspace(Z1[10],Z1[21],30)
line3 = np.linspace(Z1[21],Z1[22],10)
line4 = np.linspace(Z1[22],Z1[9], 30)

bounding_box1.append(line1)
bounding_box1.append(line2)
bounding_box1.append(line3)
bounding_box1.append(line4)

bounding_box2 = []
line5 = np.linspace(Z2[9], Z2[10],10)
line6 = np.linspace(Z2[10],Z2[21],30)
line7 = np.linspace(Z2[21],Z2[22],10)
line8 = np.linspace(Z2[22],Z2[9], 30)

bounding_box2.append(line5)
bounding_box2.append(line6)
bounding_box2.append(line7)
bounding_box2.append(line8)
#%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

labels_discrete1 =[]
labels_discrete2 =[]

for i in range(len(point_data_product1)):
    labels_discrete1.append(i)
for i in range(len(point_data_product2)):
    labels_discrete2.append(i)
 
#for i in range(len(point_data_product1)):
#    ax.text(np.array(point_data_product1)[i][0],
#                 np.array(point_data_product1)[i][1],
#                 np.array(point_data_product1)[i][2],
#                 labels_discrete1[i])
#    
#for i in range(len(point_data_product2)):    
#    ax.text(np.array(point_data_product2)[i][0],
#            np.array(point_data_product2)[i][1],
#            np.array(point_data_product2)[i][2],
#            labels_discrete2[i])

ax.add_collection3d(Poly3DCollection(verts1, facecolors='gray', edgecolors = "gray",linewidths=0, alpha=.2))
ax.add_collection3d(Poly3DCollection(verts2, facecolors='gray', edgecolors = "gray",linewidths=0, alpha=.2))

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#ax.set_title("Original Hybrid Task \n Blue: coverage tasks \n Red : discrete tasks")

ax.scatter(np.array(point_data_coverage)[:,0],
           np.array(point_data_coverage)[:,1],
           np.array(point_data_coverage)[:,2],
           marker = "o",
           c = "blue"  )
ax.scatter(np.array(point_data_discrete)[:,0],
           np.array(point_data_discrete)[:,1],
           np.array(point_data_discrete)[:,2],
           marker = "X",
           c = "red")
#for i in range(len(bounding_box1)):
#    ax.plot(np.array(bounding_box1[i])[:,0],
#               np.array(bounding_box1[i])[:,1],
#               np.array(bounding_box1[i])[:,2],
#               color = "magenta")
#for i in range(len(bounding_box2)):
#    
#    ax.plot(np.array(bounding_box2[i])[:,0],
#           np.array(bounding_box2[i])[:,1],
#           np.array(bounding_box2[i])[:,2],
#           color = "magenta")
ax.text(3000,6700,0,"number of task points: "+str (len(point_data_coverage) + len(point_data_discrete)))
ax.set_xlim(-3000,5000)
ax.set_ylim(-3000,5000)
ax.set_zlim(0,5000)
ax.view_init(elev = 90, azim = -90)
plt.show()   
#%%MAIN CALCULATIONS
scores_discrete = {}
scores_coverage = {}
score = {}
start_res = 25
#try: 
resolution = start_res
#while resolution < 20: 
    #%
time_start = time.time()
print("------------------------------------------------------------")
print("\nRESOLUTION:", resolution, "X" ,resolution, "=",resolution**2)
#list of inputs saved for later       
inputs = []
inputs.append(resolution)            #0
inputs.append(buffer_in)             #1
inputs.append(buffer_out)            #2
inputs.append(point_data_discrete)   #3
inputs.append(point_data_coverage)   #4
inputs.append(agv)                   #5
inputs.append(bounding_box1)         #6
inputs.append(bounding_box2)         #7

#%CREATE SEARCH SPACE
ss_results = atb.search_space(inputs)

ss                 = ss_results[0]
inner_bound1       = ss_results[1]
inner_bound2       = ss_results[2]
common_ss_exterior = ss_results[3]
points             = ss_results[4]
z_in1 = []
for i in range(len(inner_bound1)):
    z_in1.append(0)
    
z_in2 = []
for i in range(len(inner_bound2)):
    z_in2.append(0)
#%COVERAGE TASK
coverage_task_results = atb.coverage_task(inputs, 2, ss)
pbp_coverage_coords   = coverage_task_results[0]
centroids_coverage    = coverage_task_results[1]
clusters_coverage     = coverage_task_results[2]
centroids_ss_kmeans   = coverage_task_results[3]
#%
pbp_kmeans = []
final_ss_kmeans = {}
for i in range(len(centroids_ss_kmeans)):
    final_ss_kmeans[i] = centroids_ss_kmeans[i].intersection(ss)
for i in range(len(final_ss_kmeans)-1):
    pbp_kmeans = final_ss_kmeans[i].union(final_ss_kmeans[i+1])
 #%

##%DISCRETE TASK
discrete_task_results = atb.discrete_task(inputs, 2, ss)
pbp_discrete_coords   = discrete_task_results[0]
centroids_discrete    = discrete_task_results[1]
clusters_discrete     = discrete_task_results[2]


#%%
colors = ["red","blue","orange","green","brown","magenta","gray","yellow","cyan", "purple"]
colors_discrete = ["black","fuchsia","hotpink","peru","darkslategray","gold","lime"]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

labels_discrete1 =[]
labels_discrete2 =[]

for i in range(len(point_data_product1)):
    labels_discrete1.append(i)
for i in range(len(point_data_product2)):
    labels_discrete2.append(i)
    
ax.scatter(np.array(pbp_discrete_coords)[:,0],
                np.array(pbp_discrete_coords)[:,1],
                0,
                s = 30 , alpha = 0.9, c = "magenta")
ax.scatter(np.array(pbp_coverage_coords)[:,0],
                np.array(pbp_coverage_coords)[:,1],
                0,
                s = 30 , alpha = 0.9, c = "green")
 
#for i in range(len(point_data_product1)):
#    ax.text(np.array(point_data_product1)[i][0],
#                 np.array(point_data_product1)[i][1],
#                 np.array(point_data_product1)[i][2],
#                 labels_discrete1[i])
#    
#for i in range(len(point_data_product2)):    
#    ax.text(np.array(point_data_product2)[i][0],
#            np.array(point_data_product2)[i][1],
#            np.array(point_data_product2)[i][2],
#            labels_discrete2[i])

ax.add_collection3d(Poly3DCollection(verts1, facecolors='gray', edgecolors = "gray",linewidths=0, alpha=.2))
ax.add_collection3d(Poly3DCollection(verts2, facecolors='gray', edgecolors = "gray",linewidths=0, alpha=.2))

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#ax.set_title("Original Hybrid Task \n Blue: coverage tasks \n Red : discrete tasks")

ax.scatter(np.array(point_data_coverage)[:,0],
           np.array(point_data_coverage)[:,1],
           np.array(point_data_coverage)[:,2],
           marker = "o",
           c = "blue"  )
ax.scatter(np.array(point_data_discrete)[:,0],
           np.array(point_data_discrete)[:,1],
           np.array(point_data_discrete)[:,2],
           marker = "X",
           c = "red")

for i in range(len(clusters_discrete)):
    ax.scatter(np.array(clusters_discrete[i])[:,0],
                    np.array(clusters_discrete[i])[:,1],
                    np.array(clusters_discrete[i])[:,2],
                    c= colors_discrete[i],
                    marker = "X")
    
for i in range(len(clusters_coverage)):
    ax.scatter(np.array(clusters_coverage[i])[:,0],
                    np.array(clusters_coverage[i])[:,1],
                    np.array(clusters_coverage[i])[:,2],
                    c= colors[i],
                    marker = "o")    
#for i in range(len(bounding_box1)):
#    ax.plot(np.array(bounding_box1[i])[:,0],
#               np.array(bounding_box1[i])[:,1],
#               np.array(bounding_box1[i])[:,2],
#               color = "magenta")
#for i in range(len(bounding_box2)):
#    
#    ax.plot(np.array(bounding_box2[i])[:,0],
#           np.array(bounding_box2[i])[:,1],
#           np.array(bounding_box2[i])[:,2],
#           color = "magenta")
    
ax.set_xlim(-3000,5000)
ax.set_ylim(-3000,5000)
ax.set_zlim(0,5000)
ax.view_init(elev = 90, azim = -90)
plt.show()   
#%%
#IK FOR COVERAGE TASK
#solve IK at each pbp for all of the points, if IK returns True add 1 to reachability vector, else add 0
rel_coverage = {}
diff_coverage = []
for i in range(len(pbp_coverage_coords)):
    for j in range(len(point_data_coverage)):
        diff_coverage.append(np.subtract(point_data_coverage[j],pbp_coverage_coords[i])/1000)
        rel_coverage[i] = diff_coverage
    diff_coverage = []
print("\nStarting IK calculations for coverage task.")
t_IK1 = time.time()
IK_coverage                  = atb.inverse_kinematics(rel_coverage,robot_coverage,robot_max_reach_coverage,0)
reachability_vector_coverage = IK_coverage[0]
manipulability_coverage      = IK_coverage[1]
t_IK2 = time.time()
print("IK Calculations done! It took "+ str(float('%.4g' % (t_IK2-t_IK1)))+ " seconds.\n") 
#%
rvsums = []
for i in range(len(reachability_vector_coverage)):
    rvsums.append(sum(reachability_vector_coverage[i]))
print("Number of reachability vectors:" ,len(reachability_vector_coverage))
rv = copy.deepcopy(reachability_vector_coverage)
rvave = np.mean(rvsums)
print("Average reachability: ",rvave)
#remove the worst
print("Removing reachability vectors that can't reach that many points")
for i in range(len(reachability_vector_coverage)):
    if np.sum(reachability_vector_coverage[i]) < int(rvave/2):
        del reachability_vector_coverage[i]   
#i removed a few entries from a dict which messed up the sequence of entries, in order to iterate through them
#i put all remaininng reachability vectors in a new dict
pbp_reach_coverage= {}
for i in range(len(list(reachability_vector_coverage.keys()))):
        pbp_reach_coverage[i] = reachability_vector_coverage[list(reachability_vector_coverage.keys())[i]]
print("Number of possible base positions: " ,len(pbp_reach_coverage))
pbpsums = []
for i in range(len(pbp_reach_coverage)):
    pbpsums.append(sum(pbp_reach_coverage[i]))
pbpave = np.mean(pbpsums)
print("New average reachability: " ,pbpave,"\n")
#%
#    atb.isReachable(pbp_reach_coverage)

#% DISCRETE TASK
rel_discrete = {}
diff_discrete = []
for i in range(len(pbp_discrete_coords)):
    for j in range(len(point_data_discrete)):
        diff_discrete.append(np.subtract(point_data_discrete[j],pbp_discrete_coords[i])/1000)
        rel_discrete[i] = diff_discrete
    diff_discrete = []

#solve IK at each pbp for all of the points, 
#if IK returns True add 1 to the reachability vector, else add 0
print("Starting IK calculations for discrete task.")
t_IK1 = time.time()
IK_discrete                  = atb.inverse_kinematics(rel_discrete,robot_discrete,robot_max_reach_discrete,0)
reachability_vector_discrete = IK_discrete[0]
manipulability_discrete      = IK_discrete[1]
t_IK2 = time.time()
print("\nIK Calculations done! It took "+ str(float('%.4g' % (t_IK2-t_IK1)))+ " seconds.\n")
#%
rvsums_discrete = []
for i in range(len(reachability_vector_discrete)):
    rvsums_discrete.append(sum(reachability_vector_discrete[i]))
print("Number of reachability vectors:" ,len(reachability_vector_discrete))
rv_discrete = copy.deepcopy(reachability_vector_discrete)
rvave_discrete = np.mean(rvsums_discrete)
print("Average reachability: ",rvave_discrete)
#remove empty reachabilities and those that reach less than 5 points
print("Removing reachability vectors that can't reach that many points")
for i in range(len(reachability_vector_discrete)):
    if np.sum(reachability_vector_discrete[i]) < 1:
        del reachability_vector_discrete[i]
#removing a few entries from a dict which messed up the sequence of entries,
#in order to iterate through them all remaining reachability vectors are placed in a new dict
pbp_reach_discrete = {}
for i in range(len(list(reachability_vector_discrete.keys()))):
        pbp_reach_discrete[i] = reachability_vector_discrete[list(reachability_vector_discrete.keys())[i]]
print("Number of possible base positions: " ,len(pbp_reach_discrete))
pbpsums_discrete = []
for i in range(len(pbp_reach_discrete)):
    pbpsums_discrete.append(sum(pbp_reach_discrete[i]))
pbpave_discrete = np.mean(pbpsums_discrete)
print("New average reachability: " ,pbpave_discrete, "\n")  
#%
#    atb.isReachable(pbp_reach_discrete)

print("IK calculations are done for both tasks! Moving on with the combinations search.\n")
#% combinations 
print("\nStarting the combinations search for coverage task\n")
#    t1 = time.time()
max_combi_num = 5
combinations_results       = atb.reachability_combis(pbp_reach_coverage,max_combi_num)
valid_combos_coverage      = combinations_results[0]
incomplete_combos_coverage = combinations_results[1]
pbp_combi_coverage         = combinations_results[2]
total_checks_coverage      = combinations_results[3]
#%
print("\nStarting the combinations search for discrete task\n")
max_combi_num_discrete = 5
combinations_results         = atb.reachability_combis(pbp_reach_discrete,max_combi_num_discrete)
valid_combos_discrete        = combinations_results[0]
incomplete_combos_discrete   = combinations_results[1]
pbp_combi_discrete           = combinations_results[2]
total_checks_discrete        = combinations_results[3]

total_combi_num  = total_checks_coverage+total_checks_discrete
total_valid_num  = len(valid_combos_discrete)+len(valid_combos_coverage)
print("\nCombination search is complete for both tasks!")
print("A total of",total_combi_num," combinations have been checked")
print(total_valid_num," of these are valid")
print("\n")

#%%pick best combo
if len(valid_combos_discrete) != 0:
    print("Picking the best combination")
    valid_combo_distances = []
    valid_combo_manipulability = []    
    for k in range(len(valid_combos_discrete)):
        distances   = []
        manipulability_sums = []
        distance_pairs = []
        for combi in combinations(valid_combos_discrete[k],2):
            distance_pairs.append(combi)
        for pair in distance_pairs:
            distances.append(distance.euclidean(pbp_discrete_coords[pair[0]],pbp_discrete_coords[pair[1]]))
        for i in valid_combos_discrete[k]:               
                manipulability_sums.append(manipulability_discrete[i])
                             
        valid_combo_manipulability.append(np.average(manipulability_sums))
        valid_combo_distances.append(np.average(distances))
        
        valid_combo_torques = []
#    print("\nCalculating torques")
#    for i in tqdm(range(len(valid_combos_discrete))):
#        try:
#            current_combi = []
#            for j in valid_combos_discrete[i]:
#                current_combi.append(list(reachability_vector_discrete.keys())[j]) 
#            task_allocation = atb.allocation(current_combi, reachability_vector_discrete, point_data_discrete)   
#            valid_combo_torques.append(atb.torques(current_combi, task_allocation, robot_discrete, pbp_discrete_coords))
##                except BaseException as e:
##                    excepName = type(e).__name__
##                    print(excepName)
##                    valid_combo_torques.append(9999)
##                    continue
#        except KeyError:
#            valid_combo_torques.append(9999)
#            continue
#        except UnboundLocalError as ule:
##                print(ule)
#            valid_combo_torques.append(9999)
#            continue
#        except KeyboardInterrupt as ki:
#            print(ki)
#            sys.exit()
    #score matrix
    S_discrete = []
    for i in range(len(valid_combos_discrete)):     
        S_discrete.append([
                           100/len(valid_combos_discrete[i]),
                           valid_combo_distances[i],
                           valid_combo_manipulability[i] ,    
#                           valid_combo_torques[i]
                           ])
    scores_discrete[resolution] = S_discrete
    S_discrete = np.array(S_discrete)
    #weights
    W_discrete = np.array([100, 0.1, 5])
    weighted_sum_discrete = np.matmul(S_discrete,W_discrete)
    best = valid_combos_discrete[np.argmax(weighted_sum_discrete)]
    best_score_discrete = float('%.4g' % max(weighted_sum_discrete))/100
    #valid_combos
else:
    best = pbp_combi_discrete[min(incomplete_combos_discrete, key=incomplete_combos_discrete.get)]
    best_score_discrete = 'incomplete'
#
best_score_discrete = 0
final_combi_discrete = []
for i in best:
    final_combi_discrete.append(list(reachability_vector_discrete.keys())[i])  
print("This combination of PBPs have the best score for the discrete task: ")  
print(final_combi_discrete)

task_allocation_discrete = atb.allocation(final_combi_discrete,reachability_vector_discrete,point_data_discrete)
inputs.append(task_allocation_discrete)
   
final_bases_discrete = {}
for i in range(len(final_combi_discrete)):
    final_bases_discrete[i] = pbp_discrete_coords[final_combi_discrete[i]]
#pick best combo
print("Picking the best combination\n")
if len(valid_combos_coverage) != 0:
    goal = 100
    #find distance between robots for score matrix
    valid_combo_distances = []
    valid_combo_manipulability = []
    valid_combo_proximity = []
    for k in range(len(valid_combos_coverage)):
        distances = []
        proximities = []
        manipulability_sums = []
        distance_pairs = []
        for combi in combinations(valid_combos_coverage[k],2):
            distance_pairs.append(combi)
        for pair in distance_pairs:
            distances.append(distance.euclidean(pbp_coverage_coords[pair[0]],pbp_coverage_coords[pair[1]]))
        for i in valid_combos_coverage[k]:               
                manipulability_sums.append(manipulability_coverage[i])
                
        for i in range(len(valid_combos_coverage[k])):
            for j in range(len(valid_combos_coverage[k])):
                for l in range(len(final_combi_discrete)):
                    proximities.append(distance.euclidean(pbp_coverage_coords[valid_combos_coverage[k][i]],
                                                          pbp_discrete_coords[final_combi_discrete[l]]))

        valid_combo_distances.append(np.average(distances))
        valid_combo_proximity.append(np.average(proximities))
        valid_combo_manipulability.append(np.average(manipulability_sums))
        
#    valid_combo_torques = []
#    print("\nCalculating torques")
#    for i in tqdm(range(len(valid_combos_coverage))):
#        try:
#            current_combi = []
#            for j in valid_combos_coverage[i]:
#                current_combi.append(list(reachability_vector_coverage.keys())[j]) 
#            task_allocation = atb.allocation(current_combi,reachability_vector_coverage,point_data_coverage)
#            valid_combo_torques.append(atb.torques(current_combi,task_allocation,robot_coverage,pbp_coverage_coords,0))
##                except BaseException as e:
##                    excepName = type(e).__name__
##                    print(excepName)
##                    valid_combo_torques.append(9999)
##                    continue
#        except KeyError:
#            valid_combo_torques.append(9999)
#            continue
#        except UnboundLocalError as ule:
##                print(ule)
#            valid_combo_torques.append(9999)
#            continue
#        except KeyboardInterrupt as ki:
#            print(ki)
#            sys.exit()
#            
    #score matrix
    S_coverage = []
    for i in range(len(valid_combos_coverage)):
        S_coverage.append([
                           100/len(valid_combos_coverage[i]),
                           valid_combo_distances[i],
                           valid_combo_proximity[i],
                           valid_combo_manipulability[i] , 
#                           valid_combo_torques[i]
                          ])
    scores_coverage[resolution] = S_coverage
    S_coverage = np.array(S_coverage)
    #weights
    W_coverage = np.array([100, 0.1,0.001, 5])
    weighted_sum_coverage = np.matmul(S_coverage,W_coverage)
    best = valid_combos_coverage[np.argmax(weighted_sum_coverage)]
    best_score_coverage = float('%.4g' % max(weighted_sum_coverage))/100
    #valid_combos
else:
    index_best = min(incomplete_combos_coverage, key=incomplete_combos_coverage.get)
    best = pbp_combi_coverage[index_best]
    best_score_coverage = 'incomplete'
#%
final_combi_coverage = []
for i in best:
    final_combi_coverage.append(list(reachability_vector_coverage.keys())[i])  
print("This combination of PBPs have the best score for the coverage task: ")       
print(final_combi_coverage)
print("\n")
final_bases_coverage = {}
for i in range(len(final_combi_coverage)):
    final_bases_coverage[i] = pbp_coverage_coords[final_combi_coverage[i]]
#%
print("Allocating points to the robots that can reach them\n") 
task_allocation_coverage = atb.allocation(final_combi_coverage,reachability_vector_coverage,point_data_coverage)  
inputs.append(task_allocation_coverage)  
print("Boom we are done!")

time_end = time.time()
total_time = float('%.4g' % (time_end - time_start))
total_test_time = total_test_time + total_time

all_reachable_points_coverage = []
for i in range(len(task_allocation_coverage)):
    for j in range(len(task_allocation_coverage[i])):
        all_reachable_points_coverage.append(task_allocation_coverage[i][j])
non_reachable_points_coverage = []        
non_reachable_points_coverage = [item for item in point_data_coverage
                               if item not in all_reachable_points_coverage]

perf = atb.performance(inputs,total_time,best_score_discrete,best_score_coverage)
performance_data[resolution] = perf

#%%
# graphing of the base placement and task clustering
colors = ["red","blue","orange","green","brown","indigo","gray","yellow",
          "cyan", "purple","lime", "cadetblue","darkkhaki", "magenta"]

colors_discrete = ["black","fuchsia","brown","peru","darkslategray","gold","lime"]
fig_allocation = plt.figure(9, figsize=(12, 9))
allocation_plot     = fig_allocation.add_subplot(111, projection = "3d")
agv_coverage = {}
agv_x = 250
agv_y = 250
for i in range(len(final_combi_coverage)):
    agv_coverage = [[pbp_coverage_coords[final_combi_coverage[i]][0]+agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]+agv_y,
                     0],
                    [pbp_coverage_coords[final_combi_coverage[i]][0]+agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]-agv_y,
                     0],
                    [pbp_coverage_coords[final_combi_coverage[i]][0]-agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]-agv_y,
                     0],                      
                    [pbp_coverage_coords[final_combi_coverage[i]][0]-agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]+agv_y,
                     0],
                   
                    [pbp_coverage_coords[final_combi_coverage[i]][0]+agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]+agv_y,
                     600],
                    [pbp_coverage_coords[final_combi_coverage[i]][0]+agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]-agv_y,
                     600],
                    [pbp_coverage_coords[final_combi_coverage[i]][0]-agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]-agv_y,
                     600],                      
                    [pbp_coverage_coords[final_combi_coverage[i]][0]-agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]+agv_y,
                     600],       
                  ]
    verts_agv_coverage = [[agv_coverage[0],agv_coverage[1],agv_coverage[2],agv_coverage[3]],
                          [agv_coverage[4],agv_coverage[5],agv_coverage[6],agv_coverage[7]],
                          [agv_coverage[0],agv_coverage[4],agv_coverage[7],agv_coverage[3]],
                          [agv_coverage[0],agv_coverage[1],agv_coverage[5],agv_coverage[4]],
                          [agv_coverage[1],agv_coverage[2],agv_coverage[6],agv_coverage[5]],
                          [agv_coverage[2],agv_coverage[3],agv_coverage[7],agv_coverage[6]]
                          ]
    allocation_plot.add_collection3d(Poly3DCollection(verts_agv_coverage, facecolors='cyan', edgecolors = "gray", alpha=.1,linewidths=0.05))   
agv_discrete = {}
for i in range(len(final_combi_discrete)):
    agv_discrete = [[pbp_discrete_coords[final_combi_discrete[i]][0]+agv_x,
                     pbp_discrete_coords[final_combi_discrete[i]][1]+agv_y,
                     0],
                    [pbp_discrete_coords[final_combi_discrete[i]][0]+agv_x,
                     pbp_discrete_coords[final_combi_discrete[i]][1]-agv_y,
                     0],
                    [pbp_discrete_coords[final_combi_discrete[i]][0]-agv_x,
                     pbp_discrete_coords[final_combi_discrete[i]][1]-agv_y,
                     0],                      
                    [pbp_discrete_coords[final_combi_discrete[i]][0]-agv_x,
                     pbp_discrete_coords[final_combi_discrete[i]][1]+agv_y,
                     0],
                   
                    [pbp_discrete_coords[final_combi_discrete[i]][0]+agv_x,
                     pbp_discrete_coords[final_combi_discrete[i]][1]+agv_y,
                     600],
                    [pbp_discrete_coords[final_combi_discrete[i]][0]+agv_x,
                     pbp_discrete_coords[final_combi_discrete[i]][1]-agv_y,
                     600],
                    [pbp_discrete_coords[final_combi_discrete[i]][0]-agv_x,
                     pbp_discrete_coords[final_combi_discrete[i]][1]-agv_y,
                     600],                      
                    [pbp_discrete_coords[final_combi_discrete[i]][0]-agv_x,
                     pbp_discrete_coords[final_combi_discrete[i]][1]+agv_y,
                     600],       
                  ]
    verts_agv_discrete = [[agv_discrete[0],agv_discrete[1],agv_discrete[2],agv_discrete[3]],
                          [agv_discrete[4],agv_discrete[5],agv_discrete[6],agv_discrete[7]],
                          [agv_discrete[0],agv_discrete[4],agv_discrete[7],agv_discrete[3]],
                          [agv_discrete[0],agv_discrete[1],agv_discrete[5],agv_discrete[4]],
                          [agv_discrete[1],agv_discrete[2],agv_discrete[6],agv_discrete[5]],
                          [agv_discrete[2],agv_discrete[3],agv_discrete[7],agv_discrete[6]]
                          ]
    allocation_plot.add_collection3d(Poly3DCollection(verts_agv_discrete, facecolors='lime', edgecolors = "gray", alpha=.2,linewidths=0.05))
#plot inner/outer bounds of the search space around the product
allocation_plot.plot3D(np.array(inner_bound1)[:,0],
               np.array(inner_bound1)[:,1],
               np.array(z_in1),
               c = "black",  alpha = 0.5 )
allocation_plot.plot3D(np.array(inner_bound2)[:,0],
               np.array(inner_bound2)[:,1],
               np.array(z_in2),
               c = "black",  alpha = 0.5 )
allocation_plot.plot3D(np.array(common_ss_exterior)[:,0],
            np.array(common_ss_exterior)[:,1],
            np.array(common_ss_exterior)[:,2],
            c = "black", alpha = 0.5)
#plot allocated task points
labels_coverage =[]
for i in range(len(pbp_coverage_coords)):
    labels_coverage.append(i) 
for i in range(len(task_allocation_coverage)):
    allocation_plot.scatter(np.array(task_allocation_coverage[i])[:,0],
                            np.array(task_allocation_coverage[i])[:,1],
                            np.array(task_allocation_coverage[i])[:,2],
                            c = colors[i], marker = 'H', s = 60)

    
for i in range(len(task_allocation_discrete)):
    allocation_plot.scatter(np.array(task_allocation_discrete[i])[:,0],
                            np.array(task_allocation_discrete[i])[:,1],
                            np.array(task_allocation_discrete[i])[:,2],
                            c = colors_discrete[i], marker = 'X', s = 50)     
labels_discrete = []
labels_discrete =[]
#for i in final_combi_discrete:
#    labels_discrete.append(i)  
#plot final base placements
for i in range(len(final_combi_coverage)):
    allocation_plot.scatter(pbp_coverage_coords[final_combi_coverage[i]][0],
                            pbp_coverage_coords[final_combi_coverage[i]][1],
                            pbp_coverage_coords[final_combi_coverage[i]][2],
                            c = colors[i], s= 100, marker = "o")
    allocation_plot.scatter(final_bases_coverage[i][0],
                                final_bases_coverage[i][1],
                            final_bases_coverage[i][2],
                            c = colors[i], s= 80, marker = "o")
for i in range(len(final_combi_discrete)):
    allocation_plot.scatter(final_bases_discrete[i][0],
                            final_bases_discrete[i][1],
                            final_bases_discrete[i][2],
                            c = colors[i], s= 80, marker = "x")
#    
for i in range(len(final_combi_discrete)):
    allocation_plot.scatter(pbp_discrete_coords[final_combi_discrete[i]][0],
                            pbp_discrete_coords[final_combi_discrete[i]][1],
                            pbp_discrete_coords[final_combi_discrete[i]][2],
                            c = colors_discrete[i], s= 80, marker = 'X')

allocation_plot.scatter(np.array(pbp_coverage_coords)[:,0],
                        np.array(pbp_coverage_coords)[:,1],
                        0,
                        s = 30 , alpha = 0.2, c = "green")
#
allocation_plot.scatter(np.array(pbp_discrete_coords)[:,0],
                np.array(pbp_discrete_coords)[:,1],
                0,
                s = 30 , alpha = 0.2, c = "magenta")

allocation_plot.set_zlim(0,5000)
allocation_plot.set_ylim(-3000,6000)
allocation_plot.view_init(elev = 90, azim = -90)
allocation_plot.add_collection3d(Poly3DCollection(verts1, facecolors='gray', edgecolors = "gray",linewidths=0.05, alpha=.1))
allocation_plot.add_collection3d(Poly3DCollection(verts2, facecolors='gray', edgecolors = "gray",linewidths=0.05, alpha=.1))

allocation_plot.text(4000,8300,0,"number of task points: "+str(len(point_data_coverage)))
allocation_plot.text(4000,8000,0,"total time: "+str(total_time)+" seconds")
#allocation_plot.text(4000,7700,0,"percentage reached: "+str(perf[3])+"%")
allocation_plot.text(4000,7700,0,"coverage score: "+str(perf[5]))
#allocation_plot.text(4000,7100,0,"discrete score: "+str(perf[6]))
#allocation_plot.text(4000,6800,0,"total score: "+str(perf[7]))
plt.show()
#fname =("D:/GIT STUFF/task-clustering/results/PERFORMANCE/"+name+"/test_"+str(test)+"/result_2_"+str(resolution))
#plt.savefig(fname)
#plt.close() #close figure so that it doesn't occupy space in memory for no reason. comment out when not iterating

#        resolution +=1
#        stop iteration if score converges
#score[resolution] = performance_data[resolution][7]
#if score [resolution] == 0:
#    resolution +=1
#elif perf[4] < 100:
#    resolution +=1
#elif resolution == start_res:
#    resolution +=1
#else:
#    diff = 100*(score[resolution]-score[resolution-1])/score[resolution]
#    if diff >0 and diff < 0.5:
#        message = "Algorithm converged at resolution = ",resolution ,". The search is over"
#        print(message)
#        break
#    else:
#        resolution +=1
#if total_test_time > time_limit*3600:
#    message = "Iterations exceded pre-defined time limit without converging, stopping search."
#    print(message)
#    break
##except BaseException as e:
##    excepName = type(e).__name__
##    print(excepName)
##    telegram_send.send(messages=["THERE WAS A "+excepName])
##%%
#if total_test_time >3600:
#    total_test_time = float('%.3g' % (total_test_time/3600))
#    t=' hours'
#elif total_test_time >60:
#    total_test_time = float('%.3g' % (total_test_time/60))
#    t = ' minutes'
#else:
#    total_test_time = float('%.3g' % total_test_time)
#    t = ' seconds'    
#print("\n")
#print("TESTS ARE DONE. TOTAL TIME:")
#print(str(total_test_time)+t)
#%%save results for performance analysis
#filename = r"D:/GIT STUFF/task-clustering/results/PERFORMANCE/"+name+"/test_"+str(test)+"/performance.csv"
#data_file = open(filename, "w")
#writer = csv.writer(data_file)
#for key,value in performance_data.items():
#    writer.writerow(value)
#data_file.close()
##%%SEND TELEGRAM MESSAGE WHEN DONE
##telegram_send.send(messages=["done!"])
#telegram_send.send(messages=["Test "+str(test)+" done! Total time: "+str(total_test_time)+t])
#telegram_send.send(messages=[message])
#%%
filename = r"D:/GIT STUFF/task-clustering/results/PERFORMANCE/i_beam/perpendicular_2/task_points_test_performance.csv"
df_perf = pd.read_csv(filename,header = None)
perf_list = np.array(df_perf.values.tolist())
task_points = perf_list[:,0]
task_times  = perf_list[:,1]
task_scores = perf_list[:,2]
#
plt.plot(task_points,task_times,color = "blue",linewidth = 1)
#plt.plot(task_points,task_scores,color = "blue")
plt.show()

