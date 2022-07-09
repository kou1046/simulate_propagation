import os
import random
from typing import List
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation

class SimulatePropagation:
    def __init__(self,width:int,height:int,h:float,dt:float,border_vec:np.ndarray=None,prop_grad:List[str]=None,\
                distorted_vec:np.ndarray=None,distorted_func=None,condition:str='neumann'):
        if condition not in ['neumann','diricre']:
            raise ValueError('argument of condition must be \"neumann\" or \"diricre\"')
        self.width = width 
        self.height = height 
        self.dt = dt
        self.h = h
        self.time = 0.
        self.alpha = (dt/h)**2
        self.u = np.full((int(self.width/h),int(self.height/h)),0)
        self.u_pre = self.u.copy()
        self.condition = condition
        self.g = distorted_func
        
        #各境界付近のインデックス番号[X,Y]の配列
        self.T_idxes:List[List[int],List[int]] = [list(range(1,self.u.shape[0]-1))] + [[0]*(self.u.shape[0]-2)] #上
        self.B_idxes:List[List[int],List[int]] = [list(range(1,self.u.shape[0]-1))] + [[self.u.shape[1]-1]*(self.u.shape[0]-2)] #下
        self.L_idxes:List[List[int],List[int]] = [[0]*(self.u.shape[1]-2)] + [list(range(1,self.u.shape[1]-1))] #左
        self.R_idxes:List[List[int],List[int]] = [[self.u.shape[0]-1]*(self.u.shape[1]-2)] + [list(range(1,self.u.shape[1]-1))] #右
        self.LT_idxes:List[List[int],List[int]] = [[0],[0]]
        self.RT_idxes:List[List[int],List[int]] = [[self.u.shape[0]-1],[0]] #右上角
        self.RB_idxes:List[List[int],List[int]] = [[self.u.shape[0]-1],[self.u.shape[1]-1]] #右下角
        self.LB_idxes:List[List[int],List[int]] = [[0],[self.u.shape[1]-1]] #左下角

        #障害物がある場合,障害物の境界付近のインデックス番号[X,Y]を追加する
        if border_vec is not None:
            border_vec = np.round(border_vec/h).astype(int)
            for i,vec in enumerate(border_vec):
                x1 , y1 = vec[0]
                x2 , y2 = vec[1]
                xmin = x1 if x1 <= x2 else x2
                xmax = x1 if x1 >= x2 else x2
                ymin = y1 if y1 <= y2 else y2
                ymax = y1 if y1 >= y2 else y2
                grad = prop_grad[i]
                next_grad = prop_grad[i+1] if i+1 < len(prop_grad) else prop_grad[0]
                prev_grad = prop_grad[i-1] if i-1 >= 0 else prop_grad[-1]
                if x1 == x2:
                    #self.u[xmin,ymin:ymax+1] = 0
                    if grad == 'right':
                        if prev_grad == 'bottom':
                            [[self.R_idxes[0].append(xmin-1),self.R_idxes[1].append(y)] for y in range(ymin,ymax-1)]
                            self.RB_idxes[0].append(xmin-1); self.RB_idxes[1].append(ymax-1)
                        if prev_grad == 'top':
                            [[self.R_idxes[0].append(xmin-1),self.R_idxes[1].append(y)] for y in range(ymin-1 if next_grad == 'bottom' else ymin+1,ymax+1)]
                    if grad == 'left':
                        if prev_grad == 'bottom':
                            [[self.L_idxes[0].append(xmin),self.L_idxes[1].append(y)] for y in range(ymin,ymax+1)]
                        if prev_grad == 'top':
                            [[self.L_idxes[0].append(xmin),self.L_idxes[1].append(y)] for y in range(ymin-1,ymax)]
                if y1 == y2:
                    #self.u[xmin:xmax+1,ymin] = 0
                    if grad == 'bottom':
                        if prev_grad == 'left':
                            [[self.B_idxes[0].append(x),self.B_idxes[1].append(ymin-1)] for x in range(xmin,xmax)]
                        if prev_grad == 'right':
                            [[self.B_idxes[0].append(x),self.B_idxes[1].append(ymin-1)] for x in range(xmin,xmax+1 if next_grad == 'left' else xmax-1)]
                    if grad == 'top':
                        if prev_grad == 'left':
                            [[self.T_idxes[0].append(x),self.T_idxes[1].append(ymin)] for x in range(xmin,xmax+1)]
                        if prev_grad == 'right':
                            [[self.T_idxes[0].append(x),self.T_idxes[1].append(ymin)] for x in range(xmin,xmax-1)]
                            self.RT_idxes[0].append(xmax-1); self.RT_idxes[1].append(ymin)
            
        #境界にひずみがある場合、その境界のインデックス番号[X,Y]を取得しておく
        if distorted_vec is not None:
            distorted_vec = np.round(distorted_vec/h).astype(int)
            for vec in distorted_vec:
                if vec[0][0] == 0:
                    init_y ,end_y = min(vec[:,1]) , max(vec[:,1])
                    self.D_idxes = [[0]*(end_y-init_y+1)] + [list(range(init_y,end_y+1))]
                else:
                    init_x , end_x = min(vec[:,0]) , max(vec[:,0])
                    self.D_idxes = [list(range(init_x,end_x+1))] + [[0]*(end_x-init_x+1)]
        else:
            self.D_idxes = []
    def plot_model(self,ax):
        model = self.u.copy()
        for i,XY in enumerate([self.R_idxes,self.L_idxes,self.T_idxes,self.B_idxes,self.LT_idxes,self.RT_idxes,self.RB_idxes,self.LB_idxes,self.D_idxes]):
            model[XY] = i
        ax.imshow(model.T,extent=[0,self.width,0,self.height],cmap='jet')
    def input_gauss(self,x0,y0,rad):
        x = np.linspace(0,self.width,int(self.width/self.h)).reshape(-1,1)
        y = np.linspace(0,self.height,int(self.height/self.h))
        z = np.exp(-((x-x0)**2)*rad**2) * np.exp(-((y-y0)**2)*rad**2)
        self.u = self.u + z
        if self.time == 0.:
            self.u_pre = self.u.copy()
            self.time = dt
    def update(self):
        uR = np.roll(self.u,-1,1) 
        uL = np.roll(self.u,1,1)
        uB = np.roll(self.u,-1,0)
        uT = np.roll(self.u,1,0)
        new_u = 2*self.u - self.u_pre + self.alpha*(uL+uR+uB+uT-4*self.u) #拘束なしの点をまとめて計算

        if self.condition == 'neumann': #ノイマン境界条件
            #左端
            X , Y = np.array(self.L_idxes)
            new_u[X,Y] = 2*self.u[X,Y] - self.u_pre[X,Y] + self.alpha*(2*self.u[X+1,Y]+self.u[X,Y-1]+self.u[X,Y+1]-4*self.u[X,Y])
            o_idxes = X>0
            new_u[X[o_idxes]-1,Y[o_idxes]] = 0 #障害物内部に波が侵入しないようにする処理

            #上端
            X , Y = np.array(self.T_idxes)
            new_u[X,Y] = 2*self.u[X,Y] - self.u_pre[X,Y] + self.alpha*(self.u[X-1,Y]+self.u[X+1,Y]+2*self.u[X,Y+1]-4*self.u[X,Y])
            o_idxes = Y>0
            new_u[X[o_idxes],Y[o_idxes]-1] = 0 #障害物内部に波が侵入しないようにする処理

            #右端
            X , Y = np.array(self.R_idxes)
            new_u[X,Y] = 2*self.u[X,Y] - self.u_pre[X,Y] + self.alpha*(2*self.u[X-1,Y]+self.u[X,Y-1]+self.u[X,Y+1]-4*self.u[X,Y])
            o_idxes = X+1<self.u.shape[0]
            new_u[X[o_idxes]+1,Y[o_idxes]] = 0 #障害物内部に波が侵入しないようにする処

            #下端
            X , Y = np.array(self.B_idxes)
            new_u[X,Y] = 2*self.u[X,Y] - self.u_pre[X,Y] + self.alpha*(self.u[X-1,Y]+self.u[X+1,Y]+2*self.u[X,Y-1]-4*self.u[X,Y])
            o_idxes = Y+1<self.u.shape[1]
            new_u[X[o_idxes],Y[o_idxes]+1] = 0 #障害物内部に波が侵入しないようにする処理

            #左上端
            X , Y = np.array(self.LT_idxes)
            new_u[X,Y] = 2*self.u[X,Y] - self.u_pre[X,Y] + self.alpha*(2*self.u[X+1,Y]+2*self.u[X,Y+1]-4*self.u[X,Y])
            o_idxes_1 , o_idxes_2 = X>0, Y>0
            new_u[X[o_idxes_1]-1,Y[o_idxes_1]] = 0
            new_u[X[o_idxes_2],Y[o_idxes_2]-1] = 0 #障害物内部に波が侵入しないようにする処理

            #右上
            X , Y = np.array(self.RT_idxes)
            new_u[X,Y] = 2*self.u[X,Y] - self.u_pre[X,Y] + self.alpha*(2*self.u[X-1,Y]+2*self.u[X,Y+1]-4*self.u[X,Y])
            o_idxes_1 , o_idxes_2 = X+1<self.u.shape[0] , Y>0
            new_u[X[o_idxes_1]+1,Y[o_idxes_1]] = 0
            new_u[X[o_idxes_2],Y[o_idxes_2]-1] = 0 #障害物内部に波が侵入しないようにする処理

            #右下
            X , Y = np.array(self.RB_idxes)      
            new_u[X,Y] = 2*self.u[X,Y] - self.u_pre[X,Y] + self.alpha*(2*self.u[X-1,Y]+2*self.u[X,Y-1]-4*self.u[X,Y])
            o_idxes_1 , o_idxes_2 = X+1<self.u.shape[0] , Y+1<self.u.shape[1]
            new_u[X[o_idxes_1]+1,Y[o_idxes_1]] = 0
            new_u[X[o_idxes_2],Y[o_idxes_2]+1] = 0 #障害物内部に波が侵入しないようにする処理

            #左下
            X , Y = np.array(self.LB_idxes)      
            new_u[X,Y] = 2*self.u[X,Y] - self.u_pre[X,Y] + self.alpha*(2*self.u[X+1,Y]+2*self.u[X,Y-1]-4*self.u[X,Y])
            o_idxes_1 , o_idxes_2 = X>0, Y+1<self.u.shape[1]
            new_u[X[o_idxes_1]-1,Y[o_idxes_1]] = 0
            new_u[X[o_idxes_2],Y[o_idxes_2]+1] = 0 #障害物内部に波が侵入しないようにする処理

        elif self.condition == 'diricre': #ディリクレ境界条件
            for XY in (self.L_idxes,self.R_idxes,self.B_idxes,self.T_idxes,self.LT_idxes,self.RT_idxes,self.RB_idxes,self.LB_idxes):
                new_u[XY] = 0

        if self.D_idxes: 
            X , Y = np.array(self.D_idxes) #ひずみの境界
            new_u[X,Y] = 2*self.u[X,Y] - self.u_pre[X,Y] + self.alpha*(self.u[X+1,Y]+self.u[X,Y+1]+self.u[X,Y-1]-4*self.u[X,Y]-2*self.h*self.g(X,Y,self.time))

        self.u_pre = self.u.copy()
        self.u = new_u.copy()
        self.time += self.dt
    @property
    def result(self):
        return self.u.T
   

def distorted_func(x,y,t):
    return np.cos(2*np.pi*3*t) if t < 1 else 0

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
    obstacle_vec = np.array([[(obstacle_x[i],obstacle_y[i]),(obstacle_x[i+1],obstacle_y[i+1])] for i in range(len(obstacle_x)-1)]) #[[(x1,y1),(x2,y2)]]
    #obstacle_vecとセット，obstacle_vecに垂直で波がぶつかる方向を示す配列 bottom or top or left or right 
    grad = ['bottom','right','bottom','left','top','right','top','right']

    #ひずみがある境界座標ベクトル
    distorted_vec = np.array([[(0,height/2+0.2),(0,height/2-0.2)]])
    
    h = 0.01 #空間刻み幅
    dt = 0.005 #時間刻み
    tend = 10 #計測時間
    simulator = SimulatePropagation(width, #幅
                                    height, #高さ
                                    h, #空間刻み
                                    dt, #時間刻み
                                    obstacle_vec, #障害物
                                    grad, #障害物の向き
                                    distorted_vec, #歪境界
                                    distorted_func, #歪の関数
                                    condition='diricre' #neumann:自由端反射 , diricre:固定端反射になる
                                    )
    
    fig , ax = plt.subplots()
    ims = []
    #gif表示
    simulator.input_gauss(width,height/2,9)
    while True:
        simulator.update()
        im = ax.imshow(simulator.result,cmap='binary',extent=[0,width,0,height])
        title = ax.text(0.5, 1.01, f'Time = {round(simulator.time,2)}',
                     ha='center', va='bottom',
                     transform=ax.transAxes, fontsize='large')
        ims.append([im,title])
        if simulator.time > tend:
            break
    anim = animation.ArtistAnimation(fig,ims,interval=30)
    plt.show()
    
    #png保存
    #while True: 
    #    simulator.update()
    #    if np.round(simulator.time*100) % 50 == 0:
    #        fig , ax = plt.subplots()
    #        
    #        ax.imshow(simulator.u.T,cmap='binary',vmin=-0.1,vmax=0.1,extent=[0,width,0,height])
    #        ax.set(title=f't = {round(simulator.time,2)}',xlabel='width',ylabel='height')
    #        fig.savefig(os.path.join(os.path.dirname(__file__),f'time_{round(simulator.time,2)}_propagation.png'))
    #    if simulator.time > tend:
    #        break
 