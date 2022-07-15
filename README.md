# Autonomous Task Allocation and Base Placement
This repository contains the necessary files and scripts for my master thesis.🎓

## Synopsis

Autonomous Task Allocation and Base Placement, or simply `ATB`, takes any arbitrary industrial application as points in 3D space, 
divides the task into manageble subtasks and calculates where your industrial robot should be placed, in order to complete these tasks.  

Imagine having an industrial application like this:  
![task](/images/2_perpendicular_task.JPG)  
where you have two I-beams on which you want to do something.  Red and blue points correspond to different types of tasks(this will be explained later.)
Due to the size of the I-beams, these tasks can not be completed from one position.  `ATB` creates a search space around the physical object(the I-beams in this case), discretizes this search space according to the positions of the task points, divides the task into smaller subtasks and calculates where your robot(or robots if you have multiple) needs to be positioned in order to complete these tasks, like such:  
![result](/images/result_2_37.png)  
`ATB` makes sure that every single task point is reached and tries to find task allocations that result in as few base placements as possible.   

A more detailed example with detailed explanations on how things work can be found [here](/example) 


### Dependencies: 

* Robot kinematics are solved with [robotics-toolbox-python](https://github.com/petercorke/robotics-toolbox-python)  
* [tqdm](https://github.com/tqdm/tqdm) for cool progress bars  
* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
* [Numpy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
* [Sklearn](https://scikit-learn.org/stable/install.html)
* [Scipy](https://www.scipy.org/install.html)
* [Shapely](https://pypi.org/project/Shapely/) 