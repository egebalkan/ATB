import pandas as pd
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.ops import cascaded_union
import math
import itertools
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS  # , cluster_optics_dbscan
from sklearn.metrics import silhouette_score  # ,silhouette_samples
import roboticstoolbox as rtb
from spatialmath import SE3
import sys
from tqdm import tqdm
import random


class atb:

    #     collection of functions that are used for the autonomous
    #     task clustering and base placement algorithm.
    #
    #     some of the functions use an array called inputs.
    #     this is because they use the same inputs and it is
    #     just more convenient to do it this way instead of
    #     writing many arguments when calling the function
    #
    #     inputs is defined as follows:
    #     inputs.append(resolution)               #0
    #     inputs.append(buffer_in)                #1
    #     inputs.append(buffer_out)               #2
    #     inputs.append(point_data_discrete)      #3
    #     inputs.append(point_data_coverage)      #4
    #     inputs.append(agv)                      #5
    #     inputs.append(bounding_box1)            #6
    #     inputs.append(bounding_box2)            #7
    #     inputs.append(task_allocation_discrete) #8
    #     inputs.append(task_allocation_coverage) #9
    #
    #    last two entries are added after the algoritm runs and are
    #    used for performance evaluation. the rest are defined and
    #    added in the beginning.

    def KMeansClustering(point_data, k):
        #       takes the data set and number of clusters as inputs
        #       returns clusters, centroids thereof and silhouette score
        #       of the clustering

        #       in the current state of the algoritm, this function is not
        #       used explicitly but called within the coverage_task function

        # documentation of the sklean.KMEANS and
        # example https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        # initialize empty clusters and centroids
        clusters = {}
        centroids = {}
        for i in range(k):
            clusters[i] = []

        X = point_data
        X = np.array(X)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)

        for i in range(len(point_data)):
            clusters[y_kmeans[i]].append(point_data[i])
        for i in range(k):
            centroids[k] = kmeans.cluster_centers_

        silhouette_avg = silhouette_score(X, y_kmeans)
        return centroids, clusters, silhouette_avg

    def search_space(inputs):
        bounding_box1 = inputs[6]
        bounding_box2 = inputs[7]
        buffer_out = inputs[2]
        resolution = inputs[0]

        # create search space and place possible base positions
        bb = []
        # turn list of list into list:
        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        bb = list(itertools.chain(*bounding_box1))
        c = []
        c = [Point(coord[0], coord[1]) for coord in bb]
        points = MultiPoint(c)
        dilated_in1 = points.buffer(425)
        dilated_out1 = points.buffer(buffer_out)
        inner_bound1 = np.array(dilated_in1.exterior.coords)
        outer_bound1 = np.array(dilated_out1.exterior.coords)
        pgn1 = Polygon(outer_bound1, holes=[inner_bound1])

        z_in1 = []
        for i in range(len(inner_bound1)):
            z_in1.append(0)
        bb = []

        if bounding_box2 != 0:
            bb = list(itertools.chain(*bounding_box2))
            c = []
            c = [Point(coord[0], coord[1]) for coord in bb]
            points = MultiPoint(c)
            dilated_in2 = points.buffer(425)
            dilated_out2 = points.buffer(buffer_out)
            inner_bound2 = np.array(dilated_in2.exterior.coords)
            outer_bound2 = np.array(dilated_out2.exterior.coords)
            pgn2 = Polygon(outer_bound2, holes=[inner_bound2])

            z_in2 = []
            for i in range(len(inner_bound2)):
                z_in2.append(0)

            from shapely.ops import cascaded_union

            polygons = [pgn1, pgn2]
            pgn = cascaded_union(polygons)
            common_ss = pgn.difference(dilated_in2).difference(dilated_in1)
        else:
            common_ss = pgn1.difference(dilated_in1)
            inner_bound2 = 0

        common_ss_exterior = []
        for i in range(len(np.array(common_ss.exterior.coords))):
            common_ss_exterior.append(
                [
                    np.array(common_ss.exterior.coords)[i][0],
                    np.array(common_ss.exterior.coords)[i][1],
                    0,
                ]
            )

        min_out_x = min(np.array(common_ss.exterior.coords)[:, 0])
        max_out_x = max(np.array(common_ss.exterior.coords)[:, 0])
        min_out_y = min(np.array(common_ss.exterior.coords)[:, 1])
        max_out_y = max(np.array(common_ss.exterior.coords)[:, 1])

        lat_x = np.linspace(min_out_x, max_out_x, resolution)
        lat_y = np.linspace(min_out_y, max_out_y, resolution)

        points = MultiPoint(
            np.transpose([np.tile(lat_x, len(lat_y)), np.repeat(lat_y, len(lat_x))])
        )
        ss = points.intersection(common_ss)

        return ss, inner_bound1, inner_bound2, common_ss_exterior, points

    def coverage_task(inputs, n, ss):

        #        takes outer bound of the buffer, data set, agv height,
        #        number of clusters, and search space around the
        #        product as inputs.
        #
        #        calculates and returns possible base placements for the
        #        coverage task, clusters, centroids thereof, and search spaces
        #        around individual centroids as outputs using k-means clustering.

        buffer_out = inputs[2]
        point_data_coverage = inputs[4]
        agv = inputs[5]

        results_kmeans = []
        centroids_kmeans = {}
        clusters_kmeans = {}
        results_kmeans.append(atb.KMeansClustering(point_data_coverage, n))
        for i in range(n):
            centroids_kmeans[i] = results_kmeans[0][0][n][i]
            clusters_kmeans[i] = results_kmeans[0][1][i]

        centroids_ss_kmeans = {}
        for i in range(len(centroids_kmeans)):
            centroids_ss_kmeans[i] = Point(
                centroids_kmeans[i][0], centroids_kmeans[i][1], 0
            ).buffer(buffer_out + 600)

        final_ss_kmeans = []
        for i in range(len(centroids_ss_kmeans)):
            final_ss_kmeans.append(centroids_ss_kmeans[i].intersection(ss))

        pbp_kmeans = cascaded_union(final_ss_kmeans)

        pbp_kmeans_coords = []
        for i in range(len(pbp_kmeans)):
            pbp_kmeans_coords.append([pbp_kmeans[i].x, pbp_kmeans[i].y, agv])

        return pbp_kmeans_coords, centroids_kmeans, clusters_kmeans, centroids_ss_kmeans

    def discrete_task(inputs, min_samples, ss):

        #       takes the outer bound of buffer, data set, agv height,
        #       minimum samples for OPTICS and search space as inputs.
        #
        #       calculates and returns possible base placements for the
        #       discrete task, clusters and centroids thereof, search spaces
        #       around each centroid as outputs using the OPTICS algoritm

        buffer_out = inputs[2]
        point_data_discrete = inputs[3]
        agv = inputs[5]

        point_data_discrete_2d = []
        for i in range(len(point_data_discrete)):
            point_data_discrete_2d.append(
                [point_data_discrete[i][0], point_data_discrete[i][1]]
            )

        X = point_data_discrete_2d
        X = np.array(X)
        X = StandardScaler().fit_transform(X)

        clust = OPTICS(min_samples).fit(X)
        labels = clust.labels_[clust.ordering_]

        clusters_optics = {}  # initialize empty optics cluster dict
        for i in range(max(labels) + 1):
            clusters_optics[i] = []
        for i in range(len(point_data_discrete)):
            clusters_optics[labels[i]].append(point_data_discrete[i])

        centroids_optics = {}
        for i in range(max(labels) + 1):
            centroids_optics[i] = []
        for i in range(len(clusters_optics)):
            centroids_optics[i] = [
                np.mean(np.array(clusters_optics[i])[:, 0]),
                np.mean(np.array(clusters_optics[i])[:, 1]),
            ]
        centroids_ss_optics = []
        for i in range(len(centroids_optics)):
            centroids_ss_optics.append(
                Point(centroids_optics[i][0], centroids_optics[i][1], 0).buffer(
                    buffer_out + 200
                )
            )
        #        if len(centroids_ss_optics) > 1:
        #            final_ss_optics = centroids_ss_optics[0].union(centroids_ss_optics[1])
        #        else:
        #            final_ss_optics = centroids_ss_optics[0].union(centroids_ss_optics[0])
        #        for i in range(1,len(centroids_ss_optics)):
        #            final_ss_optics = final_ss_optics.union(centroids_ss_optics[i])
        final_ss_optics = cascaded_union(centroids_ss_optics)

        #        final_ss_optics = {}
        #        for i in range(len(centroids_ss_optics)):
        #            final_ss_optics[i] = centroids_ss_optics[i].intersection(ss)

        pbp_optics = final_ss_optics.intersection(ss)

        #        for i in range(len(final_ss_optics)-1):
        #            pbp_optics = final_ss_optics[i].union(final_ss_optics[i+1])

        pbp_optics_coords = []
        for i in range(len(pbp_optics)):
            pbp_optics_coords.append([pbp_optics[i].x, pbp_optics[i].y, agv])
        return (
            pbp_optics_coords,
            centroids_optics,
            clusters_optics,
            pbp_optics,
            final_ss_optics,
            centroids_ss_optics,
        )

    def inverse_kinematics(rel, robot, robot_max_reach, deg):

        #       takes coordinates of each point relative to each possible
        #       base placement, the DH parameters of the used robot,
        #       maximum reach of the robot and approaching angle of the
        #       end effector as inputs.
        #
        #       calculates and returns the reachability vector and the
        #       manipulability value at each possible base placement as outputs

        reach = []
        reachability_vector = {}
        manipulability_val = []
        manipulability_task = {}

        for i in tqdm(
            range(len(rel))
        ):  # tqdm is used for creating a progress bar, can be safely removed
            for j in range(len(rel[i])):
                if distance.euclidean([0, 0, 0], rel[i][j]) > robot_max_reach:
                    sol = False
                else:
                    IK = robot.ikine_LMS(SE3(rel[i][j]) * SE3.Rx(deg, "deg"))
                    #                    IK = robot.ikine_LMS(SE3(rel[i][j]))
                    joint_configs = IK[0]
                    sol = IK[1]

                if sol == True:
                    manipulability_val.append(robot.manipulability(joint_configs))
                    reach.append(1)
                else:
                    manipulability_val.append(0)
                    reach.append(0)
            manipulability_task[i] = np.sum(manipulability_val)
            manipulability_val = []
            reachability_vector[i] = np.array(reach)
            reach = []

        return reachability_vector, manipulability_task

    def reachability_combis(pbp_reach, max_combi_num):

        #       takes the reachability vectors of better-than-average possible
        #       base placements and the maximum number of base placemets as inputs.
        #
        #       calculates combinations of base placemets which fullfil the entire
        #       task and returns the valid combinations, invalid combinations, all
        #       of the combinations and the total number of checks as outputs.

        combi_num = 2
        valid_combos = []
        pbp_combi = []
        total_checks = 0

        while len(valid_combos) == 0:
            print("Checking combinations with " + str(combi_num) + " PBPs")

            for combo in combinations(pbp_reach, combi_num):
                pbp_combi.append(combo)
            total_checks += len(pbp_combi)
            print("There are " + str(len(pbp_combi)) + " of them")
            a = {}
            valid_combos = []
            incomplete_combos = {}
            for i in tqdm(range(len(pbp_combi))):
                for j in range(len(pbp_combi[i])):
                    a[j] = pbp_reach[pbp_combi[i][j]]
                cv = 0
                for k in range(len(a)):
                    cv += a[k]
                    if any([v == 0 for v in cv]) == False:
                        valid_combos.append(pbp_combi[i])
                    else:
                        unreachables = 0
                        for l in range(len(cv)):
                            if cv[l] == 0:
                                unreachables += 1
                        incomplete_combos[i] = unreachables
            valid_combos = list(dict.fromkeys(valid_combos))
            #            if len(valid_combos) != 0:
            #                min_req = combi_num #minimum required base positions, gonna check for one more
            combi_num += 1
            print("\n")
            if combi_num == max_combi_num:
                break

        #        if combi_num == min_req :
        print("Checking combinations with " + str(combi_num) + " PBPs")
        for combo in combinations(pbp_reach, combi_num):
            pbp_combi.append(combo)
        total_checks += len(pbp_combi)
        print("There are " + str(len(pbp_combi)) + " of them")

        a = {}
        for i in tqdm(range(len(pbp_combi))):
            for j in range(len(pbp_combi[i])):
                a[j] = pbp_reach[pbp_combi[i][j]]
            cv = 0
            for k in range(len(a)):
                cv += a[k]
                if any([v == 0 for v in cv]) == False:
                    valid_combos.append(pbp_combi[i])
                else:
                    unreachables = 0
                    for l in range(len(cv)):
                        if cv[l] == 0:
                            unreachables += 1
                    incomplete_combos[i] = unreachables
        valid_combos = list(dict.fromkeys(valid_combos))
        #            combi_num +=1

        print("\nValid combinations are calculated!")
        print(total_checks, "combinations were checked")
        print(len(valid_combos), "valid combinations found")
        return valid_combos, incomplete_combos, pbp_combi, total_checks

    def allocation(combi, reachability_vector, point_data):

        #       takes the combination with the best score, the dictionary of all
        #       reachability vectors and the data set as inputs.
        #
        #       allocates the task points to the base placements that suits them
        #       the best. returns allocated task points

        task_allocation = {}
        task_allocation_indices = {}
        for i in range(len(combi)):
            task_allocation[i] = []
            task_allocation_indices[i] = []
        tasks = []
        allocate = []
        indices = []
        for k in range(len(combi)):
            for i in range(len(reachability_vector[combi[k]])):
                if (reachability_vector[combi[k]])[i] == 1:
                    tasks.append(i)
            for i in tasks:
                indices.append(i)
            task_allocation_indices[k] = indices
            tasks = []
            indices = []

        allocate_combo = []
        for combo in combinations(task_allocation_indices, 2):
            allocate_combo.append(combo)

        for i in allocate_combo:
            common = list(
                set(task_allocation_indices[i[0]]).intersection(
                    task_allocation_indices[i[1]]
                )
            )
            if len(common) != 0:
                if len(task_allocation_indices[i[0]]) > len(
                    task_allocation_indices[i[1]]
                ):
                    for j in range(len(common)):
                        task_allocation_indices[i[0]].remove(common[j])
                else:
                    for j in range(len(common)):
                        task_allocation_indices[i[1]].remove(common[j])
        for i in range(len(task_allocation_indices)):
            for j in range(len(task_allocation_indices[i])):
                allocate.append(point_data[task_allocation_indices[i][j]])
            task_allocation[i] = allocate
            allocate = []
        return task_allocation

    def performance(inputs, total_time, best_score_discrete, best_score_coverage):

        #       takes pretty much every performance metric about the final base placement
        #       and returns an organized dictionary of them

        resolution = inputs[0]
        point_data_discrete = inputs[3]
        point_data_coverage = inputs[4]
        task_allocation_discrete = inputs[6]
        task_allocation_coverage = inputs[7]

        num_task_points = len(point_data_discrete) + len(point_data_coverage)
        search_space_resolution = resolution**2

        all_reachable_points_coverage = []
        for i in range(len(task_allocation_coverage)):
            for j in range(len(task_allocation_coverage[i])):
                all_reachable_points_coverage.append(task_allocation_coverage[i][j])

        all_reachable_points_discrete = []
        for i in range(len(task_allocation_discrete)):
            for j in range(len(task_allocation_discrete[i])):
                all_reachable_points_discrete.append(task_allocation_discrete[i][j])

        perc = (
            len(all_reachable_points_discrete) + len(all_reachable_points_coverage)
        ) / (len(point_data_coverage) + len(point_data_discrete))
        percentage_reached = float("%.2g" % perc) * 100
        num_robots = len(task_allocation_discrete) + len(task_allocation_coverage)

        if type(best_score_discrete) == str:
            score_discrete = 0.0
        else:
            score_discrete = best_score_discrete  # /len(task_allocation_discrete)
        if type(best_score_coverage) == str:
            score_coverage = 0.0
        else:
            score_coverage = best_score_coverage  # /len(task_allocation_coverage)

        total_score = float("%.5g" % (score_coverage + score_discrete))

        performance_data = []
        performance_data.append(num_task_points)
        performance_data.append(search_space_resolution)
        performance_data.append(total_time)
        performance_data.append(percentage_reached)
        performance_data.append(num_robots)
        performance_data.append(score_coverage)
        performance_data.append(score_discrete)
        performance_data.append(total_score)
        return performance_data

    def isReachable(pbp_reach):

        #       checks reachability vectors and returns a prompt if none of the
        #       points are reachable in case you want to stop the calculation
        #       and reconfigure stuff

        # create a dataframe of all points from reachability vectors
        # if a point is NOT reachable, entry is TRUE
        df_reach = pd.DataFrame.from_dict(pbp_reach, orient="index")
        # check whether any of the points are not reachable from any pbp
        is_reachable = []
        not_reachable = []
        for i in range(len(df_reach.columns)):
            if any(df_reach[i]):
                is_reachable.append(True)
            else:
                is_reachable.append(False)
                not_reachable.append(i)
        if all(is_reachable) == False:
            print("\nWARNING: NOT ALL POINTS ARE REACHABLE!\n")
            print(str(len(not_reachable)), " points are not reachable:")
            print(not_reachable)
            ans = input(
                "Do you want to stop the program and reconfigure paramaters? [Y/n] \n"
            )
            if ans == "Y":
                sys.exit()
            print("\n")
        else:
            print("All points are reachable!\n")

    def torques(current_combi, task_allocation, robot, pbp_coords, deg):

        #       calculates and returns the average torque on the robot joints at
        #       the considered base placement and allocation

        rel_traj = {}
        diff = []
        for i in range(len(current_combi)):
            for j in range(len(task_allocation[i])):
                diff.append(
                    np.subtract(task_allocation[i][j], pbp_coords[current_combi[i]])
                    / 1000
                )
                rel_traj[i] = diff
            diff = []
        traj_q = {}
        qq = []
        for i in range(len(rel_traj)):
            for j in range(len(rel_traj[i])):
                qq.append(robot.ikine_LMS(SE3(rel_traj[i][j]) * SE3.Rx(deg, "deg")).q)
            traj_q[i] = list(qq)
            qq = []

        trajectories = {}
        traj_temp = []
        for i in range(len(traj_q)):
            for j in range(len(traj_q[i])):
                traj_temp.append(list(traj_q[i][j]))
            trajectories[i] = traj_temp
            traj_temp = []

        multi_traj = {}
        for i in range(len(trajectories)):
            multi_traj[i] = rtb.mstraj(np.array(trajectories[i]), 1, 0.1, qdmax=1)
        #%torques
        torques = {}
        torq_temp = []
        for i in range(len(multi_traj)):
            for j in range(len(multi_traj[i])):
                torq_temp.append(
                    sum(robot.rne(multi_traj[i].q[j], np.zeros((6,)), np.zeros((6,))))
                )
            torques[i] = torq_temp
            torq_temp = []
        max_torques = {}
        for i in range(len(multi_traj)):
            max_torques[i] = max(np.abs(torques[i]))
        total_torque = float("%.5g" % np.average(list(max_torques.values())))
        return total_torque

    def beam_generator(center, rot, num_points, flag):

        #       creates user specified number of I-beams with user specified type and
        #       sized tasks, places them on center[x,y,z] and rotated at an angle of rot
        #       returns the vertices of the I-beam, the tasks placed on them and the
        #       bounding boxes around the I-beams

        x = center[0]
        y = center[1]
        z = center[2]

        c = math.cos(math.radians(rot))
        s = math.sin(math.radians(rot))

        A = np.array(
            [(x - 1500) * c - (y - 500) * s, (x - 1500) * s + (y - 500) * c, z]
        )
        B = np.array(
            [(x + 1500) * c - (y - 500) * s, (x + 1500) * s + (y - 500) * c, z]
        )
        C = np.array(
            [(x + 1500) * c - (y + 500) * s, (x + 1500) * s + (y + 500) * c, z]
        )
        D = np.array(
            [(x - 1500) * c - (y + 500) * s, (x - 1500) * s + (y + 500) * c, z]
        )

        bounding_box = []
        line1 = np.linspace(A, B, 10)
        line2 = np.linspace(B, C, 30)
        line3 = np.linspace(C, D, 10)
        line4 = np.linspace(D, A, 30)

        bounding_box.append(line1)
        bounding_box.append(line2)
        bounding_box.append(line3)
        bounding_box.append(line4)

        E = np.array(
            [(x - 1500) * c - (y - 500) * s, (x - 1500) * s + (y - 500) * c, z + 100]
        )
        F = np.array(
            [(x + 1500) * c - (y - 500) * s, (x + 1500) * s + (y - 500) * c, z + 100]
        )
        G = np.array(
            [(x + 1500) * c - (y + 500) * s, (x + 1500) * s + (y + 500) * c, z + 100]
        )
        H = np.array(
            [(x - 1500) * c - (y + 500) * s, (x - 1500) * s + (y + 500) * c, z + 100]
        )

        I = np.array(
            [(x - 1500) * c - (y - 500) * s, (x - 1500) * s + (y - 500) * c, z + 800]
        )
        J = np.array(
            [(x + 1500) * c - (y - 500) * s, (x + 1500) * s + (y - 500) * c, z + 800]
        )
        K = np.array(
            [(x + 1500) * c - (y + 500) * s, (x + 1500) * s + (y + 500) * c, z + 800]
        )
        L = np.array(
            [(x - 1500) * c - (y + 500) * s, (x - 1500) * s + (y + 500) * c, z + 800]
        )
        #
        M = np.array(
            [(x - 1500) * c - (y - 500) * s, (x - 1500) * s + (y - 500) * c, z + 900]
        )
        N = np.array(
            [(x + 1500) * c - (y - 500) * s, (x + 1500) * s + (y - 500) * c, z + 900]
        )
        O = np.array(
            [(x + 1500) * c - (y + 500) * s, (x + 1500) * s + (y + 500) * c, z + 900]
        )
        P = np.array(
            [(x - 1500) * c - (y + 500) * s, (x - 1500) * s + (y + 500) * c, z + 900]
        )
        top_surface = Polygon([M, N, O, P])
        #
        Q = np.array(
            [(x - 1500) * c - (y + 50) * s, (x - 1500) * s + (y + 50) * c, z + 100]
        )
        R = np.array(
            [(x - 1500) * c - (y - 50) * s, (x - 1500) * s + (y - 50) * c, z + 100]
        )
        S = np.array(
            [(x - 1500) * c - (y + 50) * s, (x - 1500) * s + (y + 50) * c, z + 800]
        )
        T = np.array(
            [(x - 1500) * c - (y - 50) * s, (x - 1500) * s + (y - 50) * c, z + 800]
        )
        #
        U = np.array(
            [(x + 1500) * c - (y + 50) * s, (x + 1500) * s + (y + 50) * c, z + 100]
        )
        V = np.array(
            [(x + 1500) * c - (y - 50) * s, (x + 1500) * s + (y - 50) * c, z + 100]
        )
        W = np.array(
            [(x + 1500) * c - (y + 50) * s, (x + 1500) * s + (y + 50) * c, z + 800]
        )
        X = np.array(
            [(x + 1500) * c - (y - 50) * s, (x + 1500) * s + (y - 50) * c, z + 800]
        )

        verts = [
            [A, B, C, D],
            [E, F, G, H],
            [I, J, K, L],
            [M, N, O, P],
            [A, B, F, E],
            [F, B, C, G],
            [G, C, D, H],
            [D, H, E, A],
            [I, M, N, J],
            [J, N, O, K],
            [K, O, P, L],
            [L, P, M, I],
            [Q, R, T, S],
            [R, V, X, T],
            [U, V, X, W],
            [W, S, Q, U],
        ]

        def random_points_within(poly, num_points):
            min_x, min_y, max_x, max_y = poly.bounds
            points = []
            while len(points) < num_points:
                random_point = Point(
                    [random.uniform(min_x, max_x), random.uniform(min_y, max_y), 900]
                )
                if random_point.within(poly):
                    points.append(random_point)
            return points

        task = []
        if flag == "c":
            rect = []
            points = random_points_within(top_surface, 6)
            for p in points:
                rect.append([p.x, p.y, p.z])
            poly = MultiPoint([rect[0], rect[1], rect[2], rect[3]]).convex_hull
            #%
            min_out_x = min(np.array(poly.exterior.coords)[:, 0])
            max_out_x = max(np.array(poly.exterior.coords)[:, 0])
            min_out_y = min(np.array(poly.exterior.coords)[:, 1])
            max_out_y = max(np.array(poly.exterior.coords)[:, 1])

            lat_x = np.linspace(min_out_x, max_out_x, num_points)
            lat_y = np.linspace(min_out_y, max_out_y, num_points)

            points = MultiPoint(
                np.transpose([np.tile(lat_x, len(lat_y)), np.repeat(lat_y, len(lat_x))])
            )
            task_points = list(points.intersection(poly))
            for p in task_points:
                task.append([p.x, p.y, p.z])

        elif flag == "d":
            points = random_points_within(top_surface, num_points)
            for p in points:
                task.append([p.x, p.y, p.z])

        return verts, bounding_box, task

    def search(resolution, bounding_boxes):
        #       creates a search space around the bounding boxes created by beam_generator()

        pgn = []
        buffer_out = 1300
        remove_afterwards = []
        ss = []
        inners = {}
        outers = {}
        dilated_ins = []
        #        dilated_outs = []
        for i in range(len(bounding_boxes)):
            # create search space and place possible base positions
            bb = []
            # turn list of list into list:
            # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
            bb = list(itertools.chain(*bounding_boxes[i]))
            c = []
            c = [Point(coord[0], coord[1]) for coord in bb]
            points = MultiPoint(c)
            dilated_in = points.buffer(425)
            dilated_out = points.buffer(buffer_out)
            inner_bound = np.array(dilated_in.exterior.coords)
            outer_bound = np.array(dilated_out.exterior.coords)

            remove_afterwards.append(dilated_in)

            pgn.append(Polygon(outer_bound, holes=[inner_bound]))
            min_x = min(np.array(pgn[i].exterior.coords)[:, 0])
            max_x = max(np.array(pgn[i].exterior.coords)[:, 0])

            min_y = min(np.array(pgn[i].exterior.coords)[:, 1])
            max_y = max(np.array(pgn[i].exterior.coords)[:, 1])

            lat_x = np.linspace(min_x, max_x, resolution)
            lat_y = np.linspace(min_y, max_y, resolution)

            points = MultiPoint(
                np.transpose([np.tile(lat_x, len(lat_y)), np.repeat(lat_y, len(lat_x))])
            )

            inners[i] = inner_bound
            outers[i] = outer_bound
            dilated_ins.append(dilated_in)
            ss.append(points.intersection(pgn[i]))

        ss = cascaded_union(ss)
        ss = ss.difference(cascaded_union(dilated_ins))
        pgn = cascaded_union(pgn)

        return ss, inners, outers, pgn
