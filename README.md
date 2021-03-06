# Autonomous Task Allocation and Base Placement
This repository contains the necessary files and scripts for my master thesis.🎓

## Synopsis

Autonomous Task Allocation and Base Placement, or simply `ATB`, takes any arbitrary industrial application as points in 3D space, 
divides the task into manageble subtasks and calculates where your industrial robot should be placed, in order to complete these tasks.  

Imagine having an industrial application like this:  

<p align="center">
  <img src="/images/2_perpendicular_task.JPG" />
</p>

where you have two I-beams on which you want to perform any industrial task, such as drilling, painting, sanding, inspection etc.  This task is abstracted as points in 3D space where the end-effector of your robot needs to reach.  Red and blue points correspond to different types of tasks(explained [here](https://github.com/egebalkan/ATB/blob/63e51b1f929570888e197cb1f5b89b54da3cdf13/ibeam_example.ipynb)).  

Due to the size of the I-beams, these tasks can not be completed by one robot placed in one position.  `ATB` creates a search space around the physical object(the I-beams in this case), discretizes this search space according to the positions of the task points, divides the task into smaller subtasks, calculates where your robot(or robots if you have multiple) needs to be positioned in order to complete these tasks and allocates the task points to each robot, like such:  
<p align="center">
  <img src="/images/result_2_37.png" />
</p>

`ATB` makes sure that every single task point is reached and tries to find task allocations that result in as few base placements as possible while also considering the strain on the robot arm, possible collisions between the robot and the product, and also if there are more than one robot present, possible collisions between robots.   

An example with detailed explanations on how things work can be found [here](ibeam_example.ipynb) 


### Dependencies: 

* [robotics-toolbox-python](https://github.com/petercorke/robotics-toolbox-python) and [spatialmath-python](https://github.com/petercorke/spatialmath-python)  for robot kinematics
* [tqdm](https://github.com/tqdm/tqdm) for cool progress bars  
* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) for dealing with csv files
* [Numpy](https://numpy.org/install/) for math
* [Matplotlib](https://matplotlib.org/stable/users/installing.html) for plots
* [Sklearn](https://scikit-learn.org/stable/install.html) and [Scipy](https://www.scipy.org/install.html)  for clustering algorithms
* [Shapely](https://pypi.org/project/Shapely/) for geometric calculations  
