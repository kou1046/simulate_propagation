from concurrent.futures import process
import os
import random
from typing import Callable, List
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
import time

from tqdm import tqdm

class SimulatePropagation:
    def __init__(self,width:int,height:int,h:float,dt:float,border_vecs:np.ndarray=None,prop_grads:List[str]=None,\
                distorted_vec:np.ndarray=None,distorted_func:Callable[[float,float,float],np.ndarray]=None,condition:str='neumann'):
        if condition not in {'neumann','diricre'}:
            raise ValueError('argument of condition must be \"neumann\" or \"diricre\"')
        self.width = width 
        self.height = height 
        self.dt = dt
        self.h = h
        self.time = 0.
        self.alpha = (dt/h)**2
        self._u = np.full((int(self.width/h),int(self.height/h)),0)
        self._u_pre = self._u.copy()
        self.condition = condition
        self.g = distorted_func
        
        #各境界付近のインデックス番号[X,Y]の配列
        self.T_idxes:List[List[int],List[int]] = [list(range(1,self._u.shape[0]-1))] + [[0]*(self._u.shape[0]-2)] #上
        self.B_idxes:List[List[int],List[int]] = [list(range(1,self._u.shape[0]-1))] + [[self._u.shape[1]-1]*(self._u.shape[0]-2)] #下
        self.L_idxes:List[List[int],List[int]] = [[0]*(self._u.shape[1]-2)] + [list(range(1,self._u.shape[1]-1))] #左
        self.R_idxes:List[List[int],List[int]] = [[self._u.shape[0]-1]*(self._u.shape[1]-2)] + [list(range(1,self._u.shape[1]-1))] #右
        self.LT_idxes:List[List[int],List[int]] = [[0],[0]]
        self.RT_idxes:List[List[int],List[int]] = [[self._u.shape[0]-1],[0]] #右上角
        self.RB_idxes:List[List[int],List[int]] = [[self._u.shape[0]-1],[self._u.shape[1]-1]] #右下角
        self.LB_idxes:List[List[int],List[int]] = [[0],[self._u.shape[1]-1]] #左下角

        #障害物がある場合,障害物の境界付近のインデックス番号[X,Y]を追加する
        if border_vecs is not None:
            self.border_vecs = np.round(border_vecs/h).astype(int)
            for border_vec,prop_grad in zip(self.border_vecs,prop_grads):
                for i,vec in enumerate(border_vec):
                    x1 , y1 = vec[0]
                    x2 , y2 = vec[1]
                    x_init = x1 if x1 <= x2 else x2
                    x_end = x1 if x1 >= x2 else x2
                    y_init = y1 if y1 <= y2 else y2
                    y_end = y1 if y1 >= y2 else y2
                    grad = prop_grad[i]
                    next_grad = prop_grad[i+1] if i+1 < len(prop_grad) else prop_grad[0]
                    prev_grad = prop_grad[i-1] if i-1 >= 0 else prop_grad[-1]
                    if grad == 'right':
                        x = x_init 
                        ymin = y_init if next_grad == 'bottom' else [y_init + 1 , self.RT_idxes[0].append(x) , self.RT_idxes[1].append(y_init)][0] #else以降の配列に深い意味はなく，appendしたいだけ
                        ymax = y_end if prev_grad == 'top' else [y_end - 1 , self.RB_idxes[0].append(x) , self.RB_idxes[1].append(y_end)][0] #else以降の配列に深い意味はなく，appendしたいだけ
                        [[self.R_idxes[0].append(x),self.R_idxes[1].append(y)] for y in range(ymin,ymax+1)] #この配列も特に意味はない，内包表記を利用してappendを繰り返しているだけ
                    if grad == 'left':
                        x = x_init
                        ymin = y_init if prev_grad == 'bottom' else [y_init + 1 , self.LT_idxes[0].append(x) , self.LT_idxes[1].append(y_init)][0] 
                        ymax = y_end if next_grad == 'top' else [y_end - 1 , self.LB_idxes[0].append(x) , self.LB_idxes[1].append(y_end)][0]
                        [[self.L_idxes[0].append(x),self.L_idxes[1].append(y)] for y in range(ymin,ymax+1)]
                    if grad == 'bottom':
                        xmin = x_init if prev_grad == 'right' else x_init + 1
                        xmax = x_end if next_grad == 'left' else x_end - 1
                        y = y_init
                        [[self.B_idxes[0].append(x),self.B_idxes[1].append(y)] for x in range(xmin,xmax+1)]
                    if grad == 'top':
                        xmin = x_init if next_grad == 'right' else x_init + 1
                        xmax = x_end if prev_grad == 'left' else x_end - 1
                        y = y_init
                        [[self.T_idxes[0].append(x),self.T_idxes[1].append(y)] for x in range(xmin,xmax+1)]
        else:
            self.border_vecs = []
        #境界にひずみがある場合、その境界のインデックス番号[X,Y]を取得しておく
        if distorted_vec is not None:
            distorted_vec = np.round(distorted_vec/h).astype(int)
            for vec in distorted_vec:
                if vec[0][0] == vec[1][0]:
                    init_y ,end_y = min(vec[:,1]) , max(vec[:,1])
                    self.D_idxes = [[0]*(end_y-init_y+1)] + [list(range(init_y,end_y+1))]
                else:
                    init_x , end_x = min(vec[:,0]) , max(vec[:,0])
                    self.D_idxes = [list(range(init_x,end_x+1))] + [[0]*(end_x-init_x+1)]
        else:
            self.D_idxes = []
    def plot_model(self,ax):
        for XY,color,label in zip([self.R_idxes,self.L_idxes,self.T_idxes,self.B_idxes,self.LT_idxes,self.RT_idxes,self.RB_idxes,self.LB_idxes,self.D_idxes],\
                          ['b','g','m','c','r','r','r','r','k'],
                          ['right','left','top','bottom','corner',None,None,None,'input']):
            if not XY:
                break
            ax.scatter(*XY,color=color,label=label)
            ax.legend(bbox_to_anchor=(0.5, -0.5), loc='upper center',ncol=3)
            ax.set(xlim=[0,self._u.shape[0]],ylim=[0,self._u.shape[1]],aspect='equal',xlabel='x grid num',ylabel='y grid num')
            ax.invert_yaxis()
    def input_gauss(self,x0,y0,rad):
        x = np.linspace(0,self.width,int(self.width/self.h)).reshape(-1,1)
        y = np.linspace(0,self.height,int(self.height/self.h))
        u = np.exp(-((x-x0)**2)*rad**2) * np.exp(-((y-y0)**2)*rad**2)
        self._u = self._u + u
        self._u_pre = self._u_pre + u
    def update(self):
        uR = np.roll(self._u,-1,1)
        uL = np.roll(self._u,1,1)
        uB = np.roll(self._u,-1,0) 
        uT = np.roll(self._u,1,0) 

        #一旦全ての点を拘束なしの条件でまとめて計算
        new_u = 2*self._u - self._u_pre + self.alpha*(uL+uR+uB+uT-4*self._u) 

        if self.condition == 'neumann': #ノイマン境界条件
            #左端
            X , Y = np.array(self.L_idxes)
            new_u[X,Y] = 2*self._u[X,Y] - self._u_pre[X,Y] + self.alpha*(2*self._u[X+1,Y]+self._u[X,Y-1]+self._u[X,Y+1]-4*self._u[X,Y])
            o_idxes = X>0
            new_u[X[o_idxes]-1,Y[o_idxes]] = 0 #障害物内部に波が侵入しないようにする処理

            #上端
            X , Y = np.array(self.T_idxes)
            new_u[X,Y] = 2*self._u[X,Y] - self._u_pre[X,Y] + self.alpha*(self._u[X-1,Y]+self._u[X+1,Y]+2*self._u[X,Y+1]-4*self._u[X,Y])
            o_idxes = Y>0
            new_u[X[o_idxes],Y[o_idxes]-1] = 0 #障害物内部に波が侵入しないようにする処理

            #右端
            X , Y = np.array(self.R_idxes)
            new_u[X,Y] = 2*self._u[X,Y] - self._u_pre[X,Y] + self.alpha*(2*self._u[X-1,Y]+self._u[X,Y-1]+self._u[X,Y+1]-4*self._u[X,Y])
            o_idxes = X+1<self._u.shape[0]
            new_u[X[o_idxes]+1,Y[o_idxes]] = 0 #障害物内部に波が侵入しないようにする処

            #下端
            X , Y = np.array(self.B_idxes)
            new_u[X,Y] = 2*self._u[X,Y] - self._u_pre[X,Y] + self.alpha*(self._u[X-1,Y]+self._u[X+1,Y]+2*self._u[X,Y-1]-4*self._u[X,Y])
            o_idxes = (Y+1<self._u.shape[1])
            new_u[X[o_idxes],Y[o_idxes]+1] = 0 #障害物内部に波が侵入しないようにする処理

            #左上端
            X , Y = np.array(self.LT_idxes)
            new_u[X,Y] = 2*self._u[X,Y] - self._u_pre[X,Y] + self.alpha*(2*self._u[X+1,Y]+2*self._u[X,Y+1]-4*self._u[X,Y])
            o_idxes_1 , o_idxes_2 = X>0, Y>0
            new_u[X[o_idxes_1]-1,Y[o_idxes_1]] = 0
            new_u[X[o_idxes_2],Y[o_idxes_2]-1] = 0 #障害物内部に波が侵入しないようにする処理

            #右上
            X , Y = np.array(self.RT_idxes)
            new_u[X,Y] = 2*self._u[X,Y] - self._u_pre[X,Y] + self.alpha*(2*self._u[X-1,Y]+2*self._u[X,Y+1]-4*self._u[X,Y])
            o_idxes_1 , o_idxes_2 = X+1<self._u.shape[0] , Y>0
            new_u[X[o_idxes_1]+1,Y[o_idxes_1]] = 0
            new_u[X[o_idxes_2],Y[o_idxes_2]-1] = 0 #障害物内部に波が侵入しないようにする処理

            #右下
            X , Y = np.array(self.RB_idxes)      
            new_u[X,Y] = 2*self._u[X,Y] - self._u_pre[X,Y] + self.alpha*(2*self._u[X-1,Y]+2*self._u[X,Y-1]-4*self._u[X,Y])
            o_idxes_1 , o_idxes_2 = X+1<self._u.shape[0] , Y+1<self._u.shape[1]
            new_u[X[o_idxes_1]+1,Y[o_idxes_1]] = 0
            new_u[X[o_idxes_2],Y[o_idxes_2]+1] = 0 #障害物内部に波が侵入しないようにする処理

            #左下
            X , Y = np.array(self.LB_idxes)      
            new_u[X,Y] = 2*self._u[X,Y] - self._u_pre[X,Y] + self.alpha*(2*self._u[X+1,Y]+2*self._u[X,Y-1]-4*self._u[X,Y])
            o_idxes_1 , o_idxes_2 = X>0, Y+1<self._u.shape[1]
            new_u[X[o_idxes_1]-1,Y[o_idxes_1]] = 0
            new_u[X[o_idxes_2],Y[o_idxes_2]+1] = 0 #障害物内部に波が侵入しないようにする処理

        elif self.condition == 'diricre': #ディリクレ境界条件
            for XY in (self.L_idxes,self.R_idxes,self.B_idxes,self.T_idxes,self.LT_idxes,self.RT_idxes,self.RB_idxes,self.LB_idxes):
                X , Y = XY
                new_u[X,Y] = 0

        if self.D_idxes: 
            X , Y = np.array(self.D_idxes) #ひずみの境界
            new_u[X,Y] = 2*self._u[X,Y] - self._u_pre[X,Y] + self.alpha*(self._u[X+1,Y]+self._u[X,Y+1]+self._u[X,Y-1]-4*self._u[X,Y]-2*self.h*self.g(X,Y,self.time))

        self._u_pre = self._u.copy()
        self._u = new_u.copy()
        self.time += self.dt
    def reset(self):
        self._u = np.full(self._u.shape,0)
        self._u_pre = self._u.copy()
        self.time = 0
    @property
    def u(self):
        return self._u.T
    
def time_measurement(func):
    def inner(*arg,**kwarg):
        start = time.time()
        print(f'{func.__name__ } 実行中...')
        func(*arg,**kwarg)
        print(f'{func.__name__} 終了.({time.time()-start} sec)')
    return inner
    
def show_model(sim:SimulatePropagation):
    fig , ax = plt.subplots()
    for border_vec in sim.border_vecs:
        ax.plot(border_vec[:,[0],[0]],border_vec[:,[0],[1]],'k')    
    sim.plot_model(ax)
    plt.show()
    
@time_measurement
def make_gif(sim:SimulatePropagation,tend:float,save:bool=False,output:str=os.path.join(__file__,'..')):
    fig , ax = plt.subplots()
    ims = []
    time_range = np.arange(0,tend+sim.dt,sim.dt)
    for _ in tqdm(time_range):
        sim.update() #dtだけ更新
        im = ax.imshow(sim.u,cmap='summer',extent=[0,sim.width,sim.height,0],vmin=-0.01,vmax=0.01)
        title = ax.text(0.5, 1.01, f'Time = {round(sim.time,2)}',
                     ha='center', va='bottom',
                     transform=ax.transAxes, fontsize='large')
        ims.append([im,title])
    anim = animation.ArtistAnimation(fig,ims,interval=30)
    fig.colorbar(im,orientation='horizontal')
    if save:
        anim.save(os.path.join(output,'propagation.gif'))
    else:
        plt.show()

@time_measurement
def save_png(sim:SimulatePropagation,tend:float,step:float,output:str=os.path.join(__file__,'..')): #step [sec]
    out_dir = os.path.join(output,'png_result')
    os.makedirs(out_dir,exist_ok=True)
    time_range = np.arange(0,tend+sim.dt,sim.dt)
    for i,_ in enumerate(time_range):
        simulator.update()
        if i*sim.dt % step == 0:
            fig , ax = plt.subplots()
            ax.imshow(sim.u,cmap='binary',vmin=-0.01,vmax=0.01,extent=[0,width,0,height])
            ax.set(title=f't = {round(sim.time,2)}',xlabel='width',ylabel='height')
            fig.savefig(os.path.join(out_dir,f'time_{round(sim.time,2)}_propagation.png'))
    
if __name__ == '__main__':

    for option,value in zip(['font.family','font.size'],['Times New Roman',20]):
        plt.rcParams[option] = value
    
    width = 5 #幅
    height = 1 #高さ
    obstacle_height_1 = 0.4
    obstacle_height_2 = 0.6
    obstacle_width = 0.8

    #障害物のx座標
    obstacle_x = [ 
        width/2 - obstacle_width/2,
        width/2 , 
        width/2 , 
        width/2 + obstacle_width/2,
        width/2 + obstacle_width/2,
        width/2,
        width/2,
        width/2 - obstacle_width/2,
        width/2 - obstacle_width/2,
    ]
    #障害物のy座標
    obstacle_y = [
        height/2 - obstacle_height_1/2,
        height/2 - obstacle_height_1/2,
        height/2 - obstacle_height_2/2,
        height/2 - obstacle_height_2/2,
        height/2 + obstacle_height_2/2,
        height/2 + obstacle_height_2/2,
        height/2 + obstacle_height_1/2,
        height/2 + obstacle_height_1/2,
        height/2 - obstacle_height_1/2,
    ]

    #障害物をベクトル表示
    obstacle_vec = np.array([[[(obstacle_x[i],obstacle_y[i]),(obstacle_x[i+1],obstacle_y[i+1])] for i in range(len(obstacle_x)-1)]]) #[[(x1,y1),(x2,y2)]]
    #obstacle_vecとセット　障害物の（波が当たる）向き
    grad = [['bottom','right','bottom','left','top','right','top','right']]
    #ひずみがある境界座標ベクトル
    distorted_vec = np.array([[(0,height/2+0.2),(0,height/2-0.2)]])
    #歪の関数
    f = 3
    distorted_func = lambda x,y,t:np.cos(2*np.pi*f*t) if  t <= 3*(1/f) else 0
    
    h = 0.01 #空間刻み幅
    dt = 0.005 #時間刻み
    tend = 10 #計測時間
    simulator = SimulatePropagation(width, #幅
                                    height, #高さ
                                    h, #空間刻み
                                    dt, #時間刻み
                                    obstacle_vec, #障害物
                                    grad, #向き
                                    distorted_vec, #歪がある境界
                                    distorted_func, #歪の関数
                                    condition='neumann' #neumann:自由端反射 , diricre:固定端反射になる
                                    )
    
    show_model(simulator)
    
    output = os.path.join(__file__,'..') #このスクリプトの場所
    make_gif(simulator,tend,save=False,output=output)
    
    simulator.reset()
    step = 0.5 #[sec]
    save_png(simulator,tend,step,output)
  