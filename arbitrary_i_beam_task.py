import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#import pandas as pd
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

#
#test = "random_1"
#name = "i_beam/random"
#time_limit = 6 #hours
#
#folder = "D:/GIT STUFF/task-clustering/results/PERFORMANCE/"+name+"/test_"+str(test)
#if os.path.exists(folder) == True:
#    print("THIS PATH ALREADY EXISTS! RECONFIGURE YOUR PATH!")
#    sys.exit()
#else:
#    os.mkdir(folder)
#%NOTIFICATIONS
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

#%
performance_data = {}
                     
robot_coverage = rtb.models.DH.UR10()
robot_discrete = rtb.models.DH.UR10()
agv = 690 

buffer_in =200
buffer_out = 1400

robot_max_reach_coverage = 1.3
robot_max_reach_discrete = 1.3
#START

total_test_time = 0
time_start = time.time()


#%
centers = [[-3000, -3000, 0 ],
           [0,5000    , 0 ],
           [1000, 0, 0 ],
           [3000,2000, 0]]

angles = [0,90,90,0]
flags  = ["c","c","c","c"]

num_points = [20,20,20,10]
beams = {}

for i in range(len(centers)):
    beams[i] = atb.beam_generator(centers[i],angles[i],num_points[i],flags[i])

verts = {}
for i in range(len(beams)):
    verts[i] = beams[i][0]
bounding_boxes = {}
for i in range(len(beams)):
    bounding_boxes[i] = beams[i][1]
tasks = {}

for i in range(len(beams)):
    tasks[i] = beams[i][2]
    
point_data_discrete = []
point_data_coverage = []

for i in range(len(tasks)):
    if flags[i] == "d":
        for j in range(len(tasks[i])):
            point_data_discrete.append(tasks[i][j])
    else:
        for j in range(len(tasks[i])):
            point_data_coverage.append(tasks[i][j])
        
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_zlim(0,3000)
ax.set_ylim(min(np.array(centers)[:,1])-3000,max(np.array(centers)[:,1])+2000)
ax.set_xlim(min(np.array(centers)[:,0])-3000,max(np.array(centers)[:,0])+2000)
ax.view_init(elev = 90, azim = -90)

for i in range(len(verts)):
    ax.add_collection3d(Poly3DCollection(verts[i], facecolors='gray', edgecolors = "gray",linewidths=0.1, alpha=.2))
#    for j in range(len(bounding_boxes[i])):
#        ax.plot(np.array(bounding_boxes[i][j][:,0]),
#                np.array(bounding_boxes[i][j][:,1]),
#                np.array(bounding_boxes[i][j][:,2]),
#                color = "magenta")
if len(point_data_discrete) != 0:
    ax.scatter(np.array(point_data_discrete)[:,0],
               np.array(point_data_discrete)[:,1],
               np.array(point_data_discrete)[:,2],
               color = "red")
if len(point_data_coverage) != 0:
    ax.scatter(np.array(point_data_coverage)[:,0],
               np.array(point_data_coverage)[:,1],
               np.array(point_data_coverage)[:,2],
               color = "blue")
#%%    
#%MAIN CALCULATIONS
scores_discrete = {}
scores_coverage = {}
score = {}
start_res = 20
#try: 
resolution = start_res

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

#%CREATE SEARCH SPACE
ss_results = atb.search(resolution,bounding_boxes)
ss = ss_results[0]
inners = ss_results[1]
outers = ss_results[2]
pgn = ss_results[3]
#%COVERAGE TASK
if len(point_data_coverage) != 0:
    pbp_coverage_coords   = []
    centroids_coverage    = []
    clusters_coverage     = []
    coverage_task_results = atb.coverage_task(inputs, 6, ss)
    pbp_coverage_coords   = coverage_task_results[0]
    centroids_coverage    = coverage_task_results[1]
    clusters_coverage     = coverage_task_results[2]
    centroids_ss_kmeans   = coverage_task_results[3]

#%DISCRETE TASK
if len(point_data_discrete) != 0:
    pbp_discrete_coords   = []
    centroids_discrete    = []
    clusters_discrete     = []
    
    discrete_task_results = atb.discrete_task(inputs, 4, ss)
    pbp_discrete_coords   = discrete_task_results[0]
    centroids_discrete    = discrete_task_results[1]
    clusters_discrete     = discrete_task_results[2]

#%
pbps = []
for i in range(len(ss)):
    pbps.append([ss[i].x,ss[i].y,0])

#%%
colors = ["red","blue","orange","green","brown","magenta","gray","yellow","cyan", "purple"]
colors_discrete = ["black","cyan","purple","peru","darkslategray","gold","lime"]

fig_hyb   = plt.figure(8, figsize=(16, 9))
hybx     = fig_hyb.add_subplot(111, projection = "3d")

hybx.set_zlim(0,5000)
hybx.set_ylim(-8000,8000)
hybx.set_xlim(-6000,10000)

for i in range(len(verts)):
    hybx.add_collection3d(Poly3DCollection(verts[i], facecolors='gray', edgecolors = "gray",linewidths=0.1, alpha=.2))
    for j in range(len(bounding_boxes[i])):
        hybx.plot(np.array(bounding_boxes[i][j][:,0]),
                  np.array(bounding_boxes[i][j][:,1]),
                  np.array(bounding_boxes[i][j][:,2]),
                  )
#for i in range(len(outers)):
#    hybx.plot(np.array(outers[i])[:,0],
#            np.array(outers[i])[:,1],
#            0,
#            color = "black",alpha = 0.5)
#for i in range(len(inners)):
#    hybx.plot(np.array(inners[i])[:,0],
#            np.array(inners[i])[:,1],
#            0,
#            color = "black",alpha = 0.5)


if len(point_data_discrete) != 0:   
    for i in range(len(clusters_discrete)):
        hybx.scatter(np.array(clusters_discrete[i])[:,0],
                        np.array(clusters_discrete[i])[:,1],
                        np.array(clusters_discrete[i])[:,2],
                        c= colors_discrete[i],
                        marker = "X")
        
        hybx.scatter(centroids_discrete[i][0],
                        centroids_discrete[i][1],
                        0,
                        c=colors_discrete[i],marker = 'v', alpha = 0.5)
        hybx.scatter(np.array(pbp_discrete_coords)[:,0],
                        np.array(pbp_discrete_coords)[:,1],
                        0,
                        s = 30 , alpha = 0.3, c = "magenta")
if len(point_data_coverage) != 0:    
    for i in range(len(clusters_coverage)):
        hybx.scatter(np.array(clusters_coverage[i])[:,0],
                        np.array(clusters_coverage[i])[:,1],
                        np.array(clusters_coverage[i])[:,2],
                        c= colors[i],
                        marker = "o")
        hybx.scatter(centroids_coverage[i][0],
                        centroids_coverage[i][1],
                        0,
                        c=colors[i],marker = 'v', alpha = 0.)
    
    
    hybx.scatter(np.array(pbp_coverage_coords)[:,0],
                 np.array(pbp_coverage_coords)[:,1],
                    0,
                s = 30 , alpha = 0.3, c = "green")

    
hybx.scatter(np.array(pbps)[:,0],
             np.array(pbps)[:,1],
             color = "gray", alpha = 0.3)
    

#%%

#IK FOR COVERAGE TASK
#solve IK at each pbp for all of the points, if IK returns True add 1 to reachability vector, else add 0
if len(point_data_coverage) != 0:
    rel_coverage = {}
    diff_coverage = []
    for i in range(len(pbp_coverage_coords)):
        for j in range(len(point_data_coverage)):
            diff_coverage.append(np.subtract(point_data_coverage[j],pbp_coverage_coords[i])/1000)
            rel_coverage[i] = diff_coverage
        diff_coverage = []
    print("\nStarting IK calculations for coverage task.")
    t_IK1 = time.time()
    IK_coverage                  = atb.inverse_kinematics(rel_coverage,robot_coverage,robot_max_reach_coverage)
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

#    atb.isReachable(pbp_reach_coverage)

#% IK for DISCRETE TASK
if len(point_data_discrete) != 0:
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
    IK_discrete                  = atb.inverse_kinematics(rel_discrete,robot_discrete,robot_max_reach_discrete)
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
#%% combinations 
if len(point_data_coverage) != 0:
    print("\nStarting the combinations search for coverage task\n")
    #    t1 = time.time()
    max_combi_num = 4
    combinations_results       = atb.reachability_combis(pbp_reach_coverage,max_combi_num)
    valid_combos_coverage      = combinations_results[0]
    incomplete_combos_coverage = combinations_results[1]
    pbp_combi_coverage         = combinations_results[2]
    total_checks_coverage      = combinations_results[3]
else:
    total_checks_coverage = 0
    valid_combos_coverage = []
#%
if len(point_data_discrete) != 0:
    print("\nStarting the combinations search for discrete task\n")
    max_combi_num_discrete = 4
    combinations_results         = atb.reachability_combis(pbp_reach_discrete,max_combi_num_discrete)
    valid_combos_discrete        = combinations_results[0]
    incomplete_combos_discrete   = combinations_results[1]
    pbp_combi_discrete           = combinations_results[2]
    total_checks_discrete        = combinations_results[3]
else:
    total_checks_discrete = 0
    valid_combos_discrete = []   

total_combi_num  = total_checks_coverage+total_checks_discrete
total_valid_num  = len(valid_combos_discrete)+len(valid_combos_coverage)
print("\nCombination search is complete for both tasks!")
print("A total of",total_combi_num," combinations have been checked")
print(total_valid_num," of these are valid")
print("\n")

#%pick best combo
if len(point_data_discrete) != 0:
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
        print("\nCalculating torques")
        for i in tqdm(range(len(valid_combos_discrete))):
            try:
                current_combi = []
                for j in valid_combos_discrete[i]:
                    current_combi.append(list(reachability_vector_discrete.keys())[j]) 
                task_allocation = atb.allocation(current_combi, reachability_vector_discrete, point_data_discrete)   
                valid_combo_torques.append(atb.torques(current_combi, task_allocation, robot_discrete, pbp_discrete_coords))
    #                except BaseException as e:
    #                    excepName = type(e).__name__
    #                    print(excepName)
    #                    valid_combo_torques.append(9999)
    #                    continue
            except KeyError:
                valid_combo_torques.append(0)
                continue
            except UnboundLocalError:
    #                print(ule)
                valid_combo_torques.append(0)
                continue
            except KeyboardInterrupt as ki:
                print(ki)
                sys.exit()
        #score matrix
        S_discrete = []
        for i in range(len(valid_combos_discrete)):     
            S_discrete.append([
                               100/len(valid_combos_discrete[i]),
                               valid_combo_distances[i],
                               valid_combo_manipulability[i] ,    
                               valid_combo_torques[i]
                               ])
        scores_discrete[resolution] = S_discrete
        S_discrete = np.array(S_discrete)
        #weights
        W_discrete = np.array([100, 0.1, 5,-10])
        weighted_sum_discrete = np.matmul(S_discrete,W_discrete)
        best = valid_combos_discrete[np.argmax(weighted_sum_discrete)]
        best_score_discrete = float('%.4g' % max(weighted_sum_discrete))/100
        #valid_combos
    else:
        best = pbp_combi_discrete[min(incomplete_combos_discrete, key=incomplete_combos_discrete.get)]
        best_score_discrete = 'incomplete'
    
    final_combi_discrete = []
    for i in best:
        final_combi_discrete.append(list(reachability_vector_discrete.keys())[i])  
    print("This combination of PBPs have the best score for the discrete task: ")  
    print(final_combi_discrete)
    
    task_allocation_discrete = atb.allocation(final_combi_discrete,reachability_vector_discrete,point_data_discrete)
    inputs.append(task_allocation_discrete) #6
       
    final_bases_discrete = {}
    for i in range(len(final_combi_discrete)):
        final_bases_discrete[i] = pbp_discrete_coords[final_combi_discrete[i]]
#pick best combo
if len(point_data_coverage) != 0:
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
            
        valid_combo_torques = []
        print("\nCalculating torques")
        for i in tqdm(range(len(valid_combos_coverage))):
            try:
                current_combi = []
                for j in valid_combos_coverage[i]:
                    current_combi.append(list(reachability_vector_coverage.keys())[j]) 
                task_allocation = atb.allocation(current_combi,reachability_vector_coverage,point_data_coverage)
                valid_combo_torques.append(atb.torques(current_combi,task_allocation,robot_coverage,pbp_coverage_coords))
    #                except BaseException as e:
    #                    excepName = type(e).__name__
    #                    print(excepName)
    #                    valid_combo_torques.append(9999)
    #                    continue
            except KeyError:
                valid_combo_torques.append(0)
                continue
            except UnboundLocalError:
    #                print(ule)
                valid_combo_torques.append(0)
                continue
            except KeyboardInterrupt as ki:
                print(ki)
                sys.exit()
                
        #score matrix
        S_coverage = []
        for i in range(len(valid_combos_coverage)):
            S_coverage.append([
                               100/len(valid_combos_coverage[i]),
                               valid_combo_distances[i],
                               valid_combo_proximity[i],
                               valid_combo_manipulability[i] , 
                               valid_combo_torques[i]
                              ])
        scores_coverage[resolution] = S_coverage
        S_coverage = np.array(S_coverage)
        #weights
        W_coverage = np.array([100, 0.1, 0.01, 5,-10])
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
inputs.append(task_allocation_coverage)  #7
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
#%
perf = atb.performance(inputs,total_time,best_score_discrete,best_score_coverage)
performance_data[resolution] = perf

#%%
# graphing of the base placement and task clustering
colors = ["red","blue","orange","green","brown","indigo","gray","yellow",
          "cyan", "purple","lime", "cadetblue","darkkhaki", "magenta"]

colors_discrete = ["black","gold","indigo","brown","peru","darkslategray","gold","lime"]
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

#plot allocated task points
for i in range(len(outers)):
    allocation_plot.plot3D(np.array(outers[i])[:,0],
            np.array(outers[i])[:,1],
            0,
            color = "black",alpha = 0.5)
for i in range(len(inners)):
    allocation_plot.plot3D(np.array(inners[i])[:,0],
            np.array(inners[i])[:,1],
            0,
            color = "black",alpha = 0.5)
 
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
for i in final_combi_discrete:
    labels_discrete.append(i)  
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
    
for i in range(len(final_combi_discrete)):
    allocation_plot.scatter(pbp_discrete_coords[final_combi_discrete[i]][0],
                            pbp_discrete_coords[final_combi_discrete[i]][1],
                            pbp_discrete_coords[final_combi_discrete[i]][2],
                            c = colors_discrete[i], s= 80, marker = 'X')

allocation_plot.scatter(np.array(pbp_coverage_coords)[:,0],
                        np.array(pbp_coverage_coords)[:,1],
                        0,
                        s = 30 , alpha = 0.3, c = "green")

allocation_plot.scatter(np.array(pbp_discrete_coords)[:,0],
                        np.array(pbp_discrete_coords)[:,1],
                        0,
                        s = 30 , alpha = 0.3, c = "magenta")
allocation_plot.scatter(np.array(pbps)[:,0],
                        np.array(pbps)[:,1],
                        color = "gray", alpha = 0.1)


for i in range(len(verts)):  
    allocation_plot.add_collection3d(Poly3DCollection(verts[i], facecolors='gray', edgecolors = "gray",linewidths=0.05, alpha=.1))


text_x = -0
text_y = -5000
allocation_plot.view_init(elev = 90, azim = -90)
allocation_plot.text(text_x, text_y+1500,0, "resolution: "         +str(resolution**2))
allocation_plot.text(text_x, text_y+1200,0, "total time: "         +str(total_time)+" seconds")
allocation_plot.text(text_x, text_y+900,0,  "percentage reached: " +str(perf[3])+"%")
allocation_plot.text(text_x, text_y+600,0,  "coverage score: "     +str(perf[5]))
allocation_plot.text(text_x, text_y+300,0,  "discrete score: "     +str(perf[6]))
allocation_plot.text(text_x, text_y,0,      "total score: "        +str(perf[7]))

allocation_plot.set_zlim(0,5000)
allocation_plot.set_xlim(-5000,9000)
allocation_plot.set_ylim(-5000,9000)

plt.show()
fname =("D:/GIT STUFF/task-clustering/results/PERFORMANCE/"+name+"/test_"+str(test)+"/result_2_"+str(4))
plt.savefig(fname)
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
#except BaseException as e:
#    excepName = type(e).__name__
#    print(excepName)
#    telegram_send.send(messages=["THERE WAS A "+excepName])
#%
if total_test_time >3600:
    total_test_time = float('%.3g' % (total_test_time/3600))
    t=' hours'
elif total_test_time >60:
    total_test_time = float('%.3g' % (total_test_time/60))
    t = ' minutes'
else:
    total_test_time = float('%.3g' % total_test_time)
    t = ' seconds'    
print("\n")
print("TESTS ARE DONE. TOTAL TIME:")
print(str(total_test_time)+t)
#%save results for performance analysis
filename = r"D:/GIT STUFF/task-clustering/results/PERFORMANCE/"+name+"/test_"+str(test)+"/performance.csv"
data_file = open(filename, "w")
writer = csv.writer(data_file)
for key,value in performance_data.items():
    writer.writerow(value)
data_file.close()
#%SEND TELEGRAM MESSAGE WHEN DONE
#telegram_send.send(messages=["done!"])
telegram_send.send(messages=["Test "+str(test)+" done! Total time: "+str(total_test_time)+t])
#telegram_send.send(messages=[message])
#%%


from stl import mesh
from mpl_toolkits import mplot3d

# Create a new plot
figure = plt.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
your_mesh = np.array(mesh.Mesh.from_file(r'D:\GIT STUFF\task-clustering\tasks\turbine\turbine_mesh.stl'))[::2]
axes.scatter(your_mesh[:,0],
             your_mesh[:,1],
             your_mesh[:,2],
             s = 5, c = "gray")

axes.set_xlim(0,2000)
axes.set_ylim(0,2000)
#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
#scale = your_mesh.points.flatten()
#axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
plt.show()