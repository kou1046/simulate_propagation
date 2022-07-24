from simulatepropagation import SimulatePropagation , make_gif , show_model
import numpy as np

width = 5; height = 5; h = 0.01; dt = 0.005
obstacle_x = [1,2,2,3,3,4,4,3,3,2,2,1,1]; obstacle_y = [2,2,1,1,2,2,3,3,4,4,3,3,2]
obstacle_vec = np.array([[[(obstacle_x[i],obstacle_y[i]),(obstacle_x[i+1],obstacle_y[i+1])] for i in range(len(obstacle_x)-1)]])
obstacle_grad = [['bottom','right','bottom','left','bottom','left','top','left','top','right','top','right']]
distorted_vec = np.array([[(0,height/2+0.2),(0,height/2-0.2)]])
freq = 5
g = lambda x,y,t:np.cos(2*np.pi*freq*t) if t <= 3*(1/freq) else 0

simulator = SimulatePropagation(width,height,h,dt,obstacle_vec,obstacle_grad,distorted_vec,g)
show_model(simulator)
make_gif(simulator,tend=10,save=False)

