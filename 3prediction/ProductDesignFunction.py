# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:23:07 2022

@author: Xinyu Cao

property&corresponding index:
    
0   4	tb	          0
1	4	vc	
2	4	tc	          2
3	4	pc	          3
4	2	ait	
5	4	Bcf	
6	1	Gf	
7	1	Hf	
8	4	Hfus	
9	4	Hsolp	
10	1	Hv	          4
11	4	Lc50_fm	      5
12	4	Ld50	      6
13	4	Lmv	
14	1	Logp	
15	4	Logws	
16	4	Osha_twa	
17	4	Pco	
18	4	Pka	
19	1	Tm	          1

"""


import math
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF
import numpy as np



def R_square(y_test,y_predict):
    ybar = np.sum(y_test) / len (y_test)
    SSE=np.sum((y_test - y_predict)**2)
    SSR = np.sum((ybar-y_predict)**2)
    SST = np.sum((y_test - ybar)**2)
    return 1-SSE/SST
    
    
  

def RMSE(y_test,y_predict):
    y_test=y_test.reshape(-1)
    y_predict=y_predict.reshape(-1)
    rmse=math.sqrt(mean_squared_error(y_test, y_predict))
    return rmse


