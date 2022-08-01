from simulatepropagation import SimulatePropagation , make_gif, show_model
import numpy as np 

width = 3.; height = 3.; dt = 0.005; h = 0.01

obs1_x = [1,1.1,1.1,2,2,2.1,2.1,1,1]; obs1_y = [1.6,1.6,2.,2.,1.6,1.6,2.1,2.1,1.6]
obs1_vecs = [((obs1_x[i],obs1_y[i]),(obs1_x[i+1],obs1_y[i+1])) for i in range(len(obs1_x)-1)]
obs1_grads = ['bottom','left','bottom','right','bottom','left','top','right']

obs2_x = [1.,1.,2.1,2.1,2.,2.,1.1,1.1,1.]; obs2_y = [1.4,0.9,0.9,1.4,1.4,1.,1.,1.4,1.4]
obs2_vecs = [((obs2_x[i],obs2_y[i]),(obs2_x[i+1],obs2_y[i+1])) for i in range(len(obs1_x)-1)]
obs2_grads = ['right','bottom','left','top','right','top','left','top']

obs_vecs_arr = np.array([obs1_vecs,obs2_vecs])
obs_grads_list = [obs1_grads,obs2_grads]
simulator = SimulatePropagation(width,height,h,dt,obs_vecs_arr,obs_grads_list) #歪の境界なし
simulator.input_gauss(width/2,height/2,rad=9,A=0.1) #初期，中央にガウス入力

show_model(simulator)
make_gif(simulator,tend=10,save=False)


