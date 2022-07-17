import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from   scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle
import time
import itertools
import copy
import os
from   itertools import combinations
import sys
#from tqdm import tqdm
#import csv
import roboticstoolbox as rtb
#import telegram_send
#from stl import mesh
#from mpl_toolkits import mplot3d
from   shapely.geometry import LineString
import shapely
#import collections
#from shapely.ops import split
#from   shapely.geometry import Polygon
from   shapely.geometry import Point
#from   shapely.geometry import LineString
from   shapely.geometry import MultiPoint
#from shapely.ops import cascaded_union
sys.path.append(r"D:\GIT STUFF\task-clustering")
from autonomous_task_base import atb

#%%
#your_mesh = np.array(mesh.Mesh.from_file(r'D:\GIT STUFF\task-clustering\tasks\turbine\turbine_mesh.stl'))
turb_floor = os.path.abspath(r"D:\GIT STUFF\task-clustering\tasks\turbine\turbine_floor.csv")
turb_task = os.path.abspath(r"D:\GIT STUFF\task-clustering\tasks\turbine.csv")
df_task = pd.read_csv(turb_task, header=None).iloc[:,:3]
point_data_task = np.array(df_task.values.tolist())
  
robot_coverage = rtb.models.DH.UR10()
robot_max_reach_coverage = 1.3
resolution = 12
#%%
# Create a new plot
#point_data_coverage = []

#for i in range(len(your_mesh)):
#    point_data_coverage.append([
#                                your_mesh[i][0],
#                                your_mesh[i][1],
#                                your_mesh[i][2]
#                                ])
#for i in range(len(your_mesh)):
#    point_data_coverage.append([
#
#                                your_mesh[i][3],
#                                your_mesh[i][4],
#                                your_mesh[i][5]
#                                ])
#for i in range(len(your_mesh)):
#    point_data_coverage.append([
#
#                                your_mesh[i][6],
#                                your_mesh[i][7],
#                                your_mesh[i][8]
#                                ])

#point_data_shifted = np.array(point_data_coverage)

point_data_shifted =  []
for i in range(len(point_data_task)):
    point_data_shifted.append([point_data_task[i][0]-point_data_task[502][0],
                               point_data_task[i][1]-point_data_task[502][1],
                               point_data_task[i][2]])    
point_data_task = np.array(point_data_shifted)

bottom  = []
for i in range(len(point_data_task)):
    if  point_data_task[:,2][i]>0 and point_data_task[:,2][i] <200:
        bottom.append(point_data_task[i])
bottom = np.array(bottom)

lines = []
for i in range(len(bottom)-1):
    line = np.linspace(bottom[i],bottom[i+1],10)
    lines.append(line)
    
bb =[]
bb = list(itertools.chain(*lines))
c = []
c = [Point(coord[0], coord[1]) for coord in bb]
points = MultiPoint(c)
dilated_in   = points.buffer(500)
dilated_out  = points.buffer(1500)
inner_bound  = np.array(dilated_in.exterior.coords)
outer_bound  = np.array(dilated_out.exterior.coords)

splitter = LineString([(-1900,700), (3500,-1200)])

search_space  = dilated_out.difference(dilated_in)
final_ss      = shapely.ops.split(search_space, splitter)[0]
ss_coords     = np.array(final_ss.exterior.coords)

z_out = []
for i in range(len(ss_coords)):
    z_out.append(0)
z_in = []
for i in range(len(inner_bound)):
    z_in.append(0)

min_out_x = min(np.array(final_ss.exterior.coords)[:,0])
max_out_x = max(np.array(final_ss.exterior.coords)[:,0])
min_out_y = min(np.array(final_ss.exterior.coords)[:,1])
max_out_y = max(np.array(final_ss.exterior.coords)[:,1])


lat_x = np.linspace(min_out_x, max_out_x,resolution)
lat_y = np.linspace(min_out_y, max_out_y,resolution)

points = MultiPoint(np.transpose([np.tile(lat_x,len(lat_y)), np.repeat(lat_y, len(lat_x))]))
ss = points.intersection(final_ss)

pbps = []
for i in range(len(ss)):
    pbps.append([np.array(ss[i].coords)[0][0], 
                 np.array(ss[i].coords)[0][1],
                          0])
#%
inputs = []
inputs.append(resolution)            #0
inputs.append(0)                     #1
inputs.append(1200)                  #2
inputs.append(0)                     #3
#inputs.append(point_data_coverage)  #4
inputs.append(point_data_task)       #4
inputs.append(510)                   #5
inputs.append(0)                     #6
coverage_task_results = atb.coverage_task(inputs, 3, ss)
pbp_coverage_coords   = coverage_task_results[0]
centroids_coverage    = coverage_task_results[1]
clusters_coverage     = coverage_task_results[2]
centroids_ss_kmeans   = coverage_task_results[3]

ss_coverage = []
for i in range(2):
    ss_coverage = centroids_ss_kmeans[i].intersection(ss)
#%%
colors = ["red","blue","orange","green","brown","magenta","gray","yellow","cyan", "purple"]
figure = plt.figure()
axes = figure.add_subplot(111, projection = '3d')

axes.plot3D(ss_coords[:,0],
             ss_coords[:,1],
             z_out,
            color = "magenta")
#axes.plot3D(inner_bound[:,0],
#             inner_bound[:,1],
#             z_in,
#            color = "magenta")

axes.scatter(np.array(pbps)[:,0],
                np.array(pbps)[:,1],
                np.array(pbps)[:,2],
                s = 30 , alpha = 0.4, c = "gray")
axes.scatter(np.array(pbp_coverage_coords)[:,0],
                np.array(pbp_coverage_coords)[:,1],
                0,
                s = 20 , alpha = 0.5, c = "green")

axes.scatter(np.array(point_data_shifted)[:,0],
             np.array(point_data_shifted)[:,1],
             np.array(point_data_shifted)[:,2],
             color = "black" , s = 2)

for i in(range(len(centroids_ss_kmeans))):
    axes.plot(np.array(centroids_ss_kmeans[i].exterior.coords)[:,0],
              np.array(centroids_ss_kmeans[i].exterior.coords)[:,1],
              0,
              alpha = 0.3)


for i in range(len(clusters_coverage)):
    axes.scatter(np.array(clusters_coverage[i])[:,0],
                    np.array(clusters_coverage[i])[:,1],
                    np.array(clusters_coverage[i])[:,2],
                    c= colors[i])
    axes.scatter(centroids_coverage[i][0],
                    centroids_coverage[i][1],
                    0,
                    c=colors[i],marker = 'v', alpha = 0.5)
axes.view_init(elev = 90, azim = -90)
axes.set_xlim(-2000,3500)
axes.set_ylim(-3500,2000)
axes.set_zlim(0    ,3500)
plt.show()
#%%
rel_coverage = {}
diff_coverage = []
for i in range(len(pbp_coverage_coords)):
    for j in range(len(point_data_task)):
        diff_coverage.append(np.subtract(point_data_task[j],pbp_coverage_coords[i])/1000)
        rel_coverage[i] = diff_coverage
    diff_coverage = []

print("\nStarting IK calculations for coverage task.")
t_IK1 = time.time()
IK_coverage                  = atb.inverse_kinematics(rel_coverage,robot_coverage,robot_max_reach_coverage,90)
reachability_vector_coverage = IK_coverage[0]
manipulability_coverage      = IK_coverage[1]
t_IK2 = time.time()
print("IK Calculations done! It took "+ str(float('%.4g' % (t_IK2-t_IK1)))+ " seconds.\n") 

rvsums = []
for i in range(len(reachability_vector_coverage)):
    rvsums.append(sum(reachability_vector_coverage[i]))
print("Number of reachability vectors:" ,len(reachability_vector_coverage))

rv = copy.deepcopy(reachability_vector_coverage)
rvave = np.mean(rvsums)
print("Average reachability: ",rvave)
print("Removing reachability vectors that can't reach that many points")
for i in range(len(reachability_vector_coverage)):
    if np.sum(reachability_vector_coverage[i]) < int(rvave/2):
        del reachability_vector_coverage[i] 
        
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
max_combi_num = 3
combinations_results       = atb.reachability_combis(pbp_reach_coverage,max_combi_num)
valid_combos_coverage      = combinations_results[0]
incomplete_combos_coverage = combinations_results[1]
pbp_combi_coverage         = combinations_results[2]
total_checks_coverage      = combinations_results[3]
#%
if len(valid_combos_coverage) != 0:
    goal = 100
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
                

        valid_combo_distances.append(np.average(distances))
        valid_combo_manipulability.append(np.average(manipulability_sums))
        
#    valid_combo_torques = []
#    print("\nCalculating torques")
#    for i in tqdm(range(len(valid_combos_coverage))):
##        try:
#            current_combi = []
#            for j in valid_combos_coverage[i]:
#                current_combi.append(list(reachability_vector_coverage.keys())[j]) 
#            task_allocation = atb.allocation(current_combi,reachability_vector_coverage,point_data_task)
#            valid_combo_torques.append(atb.torques(current_combi,task_allocation,robot_coverage,pbp_coverage_coords,90))
#        except BaseException as e:
#            excepName = type(e).__name__
#            print(excepName)
#            valid_combo_torques.append(9999)
#            continue
#        except KeyError:
#            valid_combo_torques.append(9999)
#            continue
#        except UnboundLocalError:
#            valid_combo_torques.append(9999)
#            continue
#        except KeyboardInterrupt as ki:
#            print(ki)
#            sys.exit()
            
    #score matrix
    S_coverage = []
    for i in range(len(valid_combos_coverage)):
        S_coverage.append([
                           100/len(valid_combos_coverage[i]),
                           valid_combo_distances[i],
                           valid_combo_manipulability[i] , 
#                           valid_combo_torques[i]
                          ])
    
    S_coverage = np.array(S_coverage)
    W_coverage = np.array([100, 0.01, 5]) #without torques
#    W_coverage = np.array([100, 0.01, 5,-10]) #with torques
    weighted_sum_coverage = np.matmul(S_coverage,W_coverage)
    best = valid_combos_coverage[np.argmax(weighted_sum_coverage)]
    best_score_coverage = float('%.4g' % max(weighted_sum_coverage))/100

else:
    index_best = min(incomplete_combos_coverage, key=incomplete_combos_coverage.get)
    best = pbp_combi_coverage[index_best]
    best_score_coverage = 'incomplete'

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
task_allocation_coverage = atb.allocation(final_combi_coverage,reachability_vector_coverage,point_data_task)  
inputs.append(task_allocation_coverage)  
print("Boom we are done!")
#%%
all_reachable_points_coverage = []
non_reachable_points_coverage = []
for i in range(len(task_allocation_coverage)):
    for j in range(len(task_allocation_coverage[i])):
        all_reachable_points_coverage.append(task_allocation_coverage[i][j])
        if task_allocation_coverage[i][j] not in point_data_task:
            non_reachable_points_coverage.append(task_allocation_coverage[i][j])

#%%
ordered_task_allocation = {}

#for i in range(len(task_allocation_coverage)):
#    ordered_distances[i] = []
    
for i in range(len(task_allocation_coverage)):
    distt = []
    ordered_distances = []
    for j in range(len(task_allocation_coverage[i])):
        distt.append([distance.euclidean(final_bases_coverage[i], task_allocation_coverage[i][j]) ,task_allocation_coverage[i][j]])
        distt.sort()
    for k in distt: 
        ordered_distances.append(k[1])
    ordered_task_allocation[i] = ordered_distances         
        
        #%%
colors = ["red","blue","orange","green","brown","indigo","gray","yellow",
          "cyan", "purple","lime", "cadetblue","darkkhaki", "magenta"]

fig_allocation = plt.figure(9, figsize=(12, 9))
allocation_plot     = fig_allocation.add_subplot(111, projection = "3d")
agv_coverage = {}
agv_y = 978/2
agv_x = 776/2
igps_x = agv_x +5
igps_y = -agv_y +5 
def plot_3D_cylinder(radius, height, elevation, resolution, color, x_center, y_center):
    x = np.linspace(x_center-radius, x_center+radius, resolution)
    z = np.linspace(elevation, elevation+height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center # Pythagorean theorem

    allocation_plot.plot_surface(X, Y, Z, linewidth=0, color=color)
    allocation_plot.plot_surface(X, (2*y_center-Y), Z, linewidth=0, color=color)

    floor = Circle((x_center, y_center), radius, color=color)
    allocation_plot.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    allocation_plot.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation+height, zdir="z")
    plt.show()
    
for i in range(len(final_combi_coverage)):
        plot_3D_cylinder(50, 300, 510, 10, color='black', x_center=pbp_coverage_coords[final_combi_coverage[i]][0]+igps_x,
                     y_center=pbp_coverage_coords[final_combi_coverage[i]][1]+igps_y)

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
                     510],
                    [pbp_coverage_coords[final_combi_coverage[i]][0]+agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]-agv_y,
                     510],
                    [pbp_coverage_coords[final_combi_coverage[i]][0]-agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]-agv_y,
                     510],                      
                    [pbp_coverage_coords[final_combi_coverage[i]][0]-agv_x,
                     pbp_coverage_coords[final_combi_coverage[i]][1]+agv_y,
                     510],       
                  ]
    verts_agv_coverage = [[agv_coverage[0],agv_coverage[1],agv_coverage[2],agv_coverage[3]],
                          [agv_coverage[4],agv_coverage[5],agv_coverage[6],agv_coverage[7]],
                          [agv_coverage[0],agv_coverage[4],agv_coverage[7],agv_coverage[3]],
                          [agv_coverage[0],agv_coverage[1],agv_coverage[5],agv_coverage[4]],
                          [agv_coverage[1],agv_coverage[2],agv_coverage[6],agv_coverage[5]],
                          [agv_coverage[2],agv_coverage[3],agv_coverage[7],agv_coverage[6]]
                          ]
    

        
        
    allocation_plot.add_collection3d(Poly3DCollection(verts_agv_coverage, facecolors='cyan', edgecolors = "gray", alpha=.2,linewidths=0.05))   

allocation_plot.plot3D(np.array(ss_coords)[:,0],
                       np.array(ss_coords)[:,1],
                       np.array(z_out),
                       c = "magenta",  alpha = 0.5 )

#plot allocated task points
#labels_coverage =[]
#for i in range(len(pbp_coverage_coords)):
#    labels_coverage.append(i) 

    #for i in range(len(task_allocation_coverage)):
#    allocation_plot.scatter(np.array(task_allocation_coverage[i])[:,0],
#                            np.array(task_allocation_coverage[i])[:,1],
#                            np.array(task_allocation_coverage[i])[:,2],
#                            c = colors[i], marker = 'H', s = 60)
#    
for i in range(1):    
    allocation_plot.plot(np.array(ordered_task_allocation[i])[:,0],
                         np.array(ordered_task_allocation[i])[:,1],
                         np.array(ordered_task_allocation[i])[:,2],
                         c = colors[i])
#    labels = []
#    for j in range(len(ordered_task_allocation[i])):
#        labels.append(j)
#        allocation_plot.text(ordered_task_allocation[i][j][0],
#                             ordered_task_allocation[i][j][1],
#                             ordered_task_allocation[i][j][2],
#                             labels[j])
#    
#    
    #for i in range(len(point_data_product1)):
#    ax.text(np.array(point_data_product1)[i][0],
#                 np.array(point_data_product1)[i][1],
#                 np.array(point_data_product1)[i][2],
#                 labels_discrete1[i])        
    
#plot final base placements
for i in range(len(final_combi_coverage)):   
    allocation_plot.scatter(final_bases_coverage[i][0],
                            final_bases_coverage[i][1],
                            0,
                            c = colors[i], s= 80, marker = "X")

allocation_plot.scatter(np.array(pbp_coverage_coords)[:,0],
                        np.array(pbp_coverage_coords)[:,1],
                        0,
                        s = 30 , alpha = 0.5, c = "green")

allocation_plot.scatter(0,0,0, color = "black" , s = 100, marker = "X")

allocation_plot.scatter(np.array(pbps)[:,0],
                        np.array(pbps)[:,1],
                        np.array(pbps)[:,2],
                        s = 30 , alpha = 0.5, c = "gray")

allocation_plot.scatter(point_data_task[502][0],
                        point_data_task[502][1],
                        point_data_task[502][2],
                        marker = "X", color = "black")

allocation_plot.set_xlim(-2000,3500)
allocation_plot.set_ylim(-3500,2000)
allocation_plot.set_zlim(  0  ,5500)

allocation_plot.view_init(elev = 90, azim = -90)
allocation_plot.text(4000,8300,0,"resolution: "+str(resolution**2))

plt.show()
#%%

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

def plot_3D_cylinder(radius, height, elevation=0, resolution=100, color='r', x_center = 0, y_center = 0):
#    fig=plt.figure()
#    ax = Axes3D(fig, azim=30, elev=30)

    x = np.linspace(x_center-radius, x_center+radius, resolution)
    z = np.linspace(elevation, elevation+height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center # Pythagorean theorem

    allocation_plot.plot_surface(X, Y, Z, linewidth=0, color=color)
    allocation_plot.plot_surface(X, (2*y_center-Y), Z, linewidth=0, color=color)

    floor = Circle((x_center, y_center), radius, color=color)
    allocation_plot.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    allocation_plot.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation+height, zdir="z")
    plt.show()

# params
radius = 3
height = 10
elevation = -5
resolution = 10
color = 'r'
x_center = 3
y_center = -2

plot_3D_cylinder(radius, height, elevation=elevation, resolution=resolution, color=color, x_center=x_center, y_center=y_center)
#%%
#from    spatialmath import SE3
#qt = robot_coverage.ikine_LMS(SE3(rel_coverage[0][1]) * SE3.Rx(90,'deg'))
#
#
#robot_coverage.plot(qt.q)


