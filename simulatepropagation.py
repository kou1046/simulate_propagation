import os
from typing import List
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation

class SimulatePropagation:
    def __init__(self,width:int,height:int,h:float,dt:float,border_vec:np.ndarray=None,prop_grad:List[str]=None,condition='neumann'):
        if condition not in ['neumann','diricre']:
            raise ValueError('argument of condition must be \"neumann\" or \"diricre\"')
        self.width = width 
        self.height = height 
        self.dt = dt
        self.h = h
        self.time = dt
        self.alpha = (dt/h)**2
        self.u = np.full((int(self.width/h),int(self.height/h)),0)
        self.u_pre = self.u.copy()
        self.condition = condition

        #各境界付近のインデックス番号(i,j)の配列
        self.T_idxes = [(x,0) for x in range(1,self.u.shape[0]-1)] #上
        self.B_idxes = [(x,self.u.shape[1]-1) for x in range(1,self.u.shape[0]-1)] #下
        self.L_idxes = [(0,y) for y in range(1,self.u.shape[1]-1)] #左
        self.R_idxes = [(self.u.shape[0]-1,y) for y in range(1,self.u.shape[1]-1)] #右
        self.LT_idxes = [(0,0)] #左上角
        self.RT_idxes = [(self.u.shape[0]-1,0)] #右上角
        self.RB_idxes = [(self.u.shape[0]-1,self.u.shape[1]-1)] #右下角
        self.LB_idxes = [(0,self.u.shape[1]-1)] #左下角

        #障害物がある場合,障害物の境界付近のインデックス番号(i,j)を追加する
        if border_vec is not None:
            border_vec = [[np.round((xy1)/h).astype(int),np.round((xy2)/h).astype(int)] for xy1,xy2 in border_vec]
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
                            self.R_idxes += [(xmin-1,y) for y in range(ymin,ymax-1)]
                            self.RB_idxes += [(xmin-1,ymax-1)]
                        if prev_grad == 'top':
                            self.R_idxes += [(xmin-1,y-1) for y in range(ymin if next_grad == 'bottom' else ymin+2,ymax+1)]
                    if grad == 'left':
                        if prev_grad == 'bottom':
                            self.L_idxes += [(xmin,y) for y in range(ymin,ymax+1)]
                        if prev_grad == 'top':
                            self.L_idxes += [(xmin,y) for y in range(ymin-1,ymax)]
                if y1 == y2:
                    #self.u[xmin:xmax+1,ymin] = 0
                    if grad == 'bottom':
                        if prev_grad == 'left':
                            self.B_idxes += [(x,ymin-1) for x in range(xmin,xmax)]
                        if prev_grad == 'right':
                            self.B_idxes += [(x,ymin-1) for x in range(xmin,xmax+1 if next_grad =='left' else xmax-1)]
                    if grad == 'top':
                        if prev_grad == 'left':
                            self.T_idxes += [(x,ymin) for x in range(xmin,xmax+1)]
                        if prev_grad == 'right':
                            self.T_idxes += [(x,ymin) for x in range(xmin,xmax-1)]
                            self.RT_idxes += [(xmax-1,ymin)]
    def plot_model(self,ax):
        model = self.u.copy()
        for i,XY in enumerate([self.R_idxes,self.L_idxes,self.T_idxes,self.B_idxes,self.LT_idxes,self.RT_idxes,self.RB_idxes,self.LB_idxes]):
            for x,y in XY:
                model[x,y] = 1 if i ==0 else 2 if i==1 else 3 if i==2 else 4 if i==3 else 5
        ax.imshow(model.T,extent=[0,self.width,0,self.height],cmap='jet')
    def set_input(self,x0,y0,kind='gauss'):
        def gauss(x0:float,y0:float,rad,X:np.ndarray,Y:np.ndarray) -> np.ndarray:
            return np.exp(-((X-x0)**2)*rad**2) * np.exp(-((Y-y0)**2)*rad**2)
        x = np.linspace(0,self.width,int(self.width/self.h)).reshape(-1,1)
        y = np.linspace(0,self.height,int(self.height/self.h))
        if kind == 'gauss':
            self.u = self.u + gauss(x0,y0,9,x,y)
        self.u_pre = self.u.copy()
    def update(self):
        uR = np.roll(self.u,-1,1) 
        uL = np.roll(self.u,1,1)
        uB = np.roll(self.u,-1,0)
        uT = np.roll(self.u,1,0)
        new_u = 2*self.u - self.u_pre + self.alpha*(uL+uR+uB+uT-4*self.u)
        if self.condition == 'neumann': #ノイマン境界条件(自由端反射)
            
            for x,y in self.L_idxes: #左端は右の波
                new_u[x,y] = new_u[x+1,y]
                if x > 0 : #障害物内部に波が侵入しないようにする処理
                    new_u[x-1,y] = 0
            for x,y in self.R_idxes: #右端は左の波
                new_u[x,y] = new_u[x-1,y]
                if x+1 < self.u.shape[0]: #障害物内部に波が侵入しないようにする処理
                    new_u[x+1,y] = 0
            for x,y in self.T_idxes: #上端は下の波
                new_u[x,y] = new_u[x,y+1]
                if y > 0: #障害物内部に波が侵入しないようにする処理
                    new_u[x,y-1] = 0 
            for x,y in self.B_idxes: #下端は上の波
                new_u[x,y] = new_u[x,y-1]
                if y+1 < self.u.shape[1]: #障害物内部に波が侵入しないようにする処理
                    new_u[x,y+1] = 0
            for x,y in self.LT_idxes: #左上は右と下の波平均
                new_u[x,y] = (new_u[x+1,y] + new_u[x,y+1]) / 2
                #障害物内部に波が侵入しないようにする処理
                if x > 0 : 
                    new_u[x-1,y] = 0
                if y > 0: 
                    new_u[x,y-1] = 0 
            for x,y in self.RT_idxes: #右上は左と下の波平均
                new_u[x,y] = (new_u[x-1,y] + new_u[x,y+1]) / 2
                 #障害物内部に波が侵入しないようにする処理
                if x+1 < self.u.shape[0]:
                    new_u[x+1,y] = 0
                if y > 0: 
                    new_u[x,y-1] = 0 
            for x,y in self.RB_idxes: #右下は左と上の波平均
                new_u[x,y] = (new_u[x-1,y] + new_u[x,y-1]) / 2
                #障害物内部に波が侵入しないようにする処理
                if x+1 < self.u.shape[0]: 
                    new_u[x+1,y] = 0
                if y+1 < self.u.shape[1]: 
                    new_u[x,y+1] = 0
            for x,y in self.LB_idxes: #左下は右と上の波平均
                new_u[x,y] = (new_u[x+1,y] + new_u[x,y-1]) / 2
                #障害物内部に波が侵入しないようにする処理
                if x > 0 : 
                    new_u[x-1,y] = 0
                if y+1 < self.u.shape[1]: 
                    new_u[x,y+1] = 0
                    
        if self.condition == 'diricre': #ディリクレ境界条件(固定端反射)
            for XY in [self.L_idxes,self.R_idxes,self.T_idxes,self.B_idxes,self.LT_idxes,self.RT_idxes,self.RB_idxes,self.LB_idxes]:
                for x,y in XY:
                    new_u[x,y] = 0 #端は全て零
        self.u_pre = self.u.copy()
        self.u = new_u.copy()
        self.time += self.dt
    def reset(self):
        self.u = 0 
        self.u_pre = 0
        self.time = 0

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
    
    
    h = 0.01 #空間刻み幅
    dt = 0.005 #時間刻み
    tend = 10 #計測時間
    simulator = SimulatePropagation(width, #幅
                                    height, #高さ
                                    h, #空間刻み
                                    dt, #時間刻み
                                    obstacle_vec, #障害物ベクトル
                                    grad, #障害物に当たる直線の波の向き
                                    condition='neumann' #newumann:自由端反射 , diricre:固定端反射になる
                                    )
    
    fig , ax = plt.subplots()
    ax.plot(obstacle_x,obstacle_y,'k') #境界線の描画
    simulator.set_input(0,height/2,kind='gauss') #初期値の入力 kind='gauss'でガウス布関数
    ims = []
    
    #gif表示
    while True: 
        simulator.update()
        im = ax.imshow(simulator.u.T,cmap='binary',vmin=-0.1,vmax=0.1,extent=[0,width,0,height])
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
 