# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:45:11 2022

@author: Xinyu Cao
"""

import pickle
import pandas as pd
import numpy as np
from ProductDesignFunction import RMSE,R_square
import os
os.chdir(r"C:\Users\cheese_cake\Desktop\jenny\程序\22_12product design")
import warnings
warnings.filterwarnings("ignore")

for index in range(1):
    # index=0
    print("index:",index)
    name=["tb","vc","tc","pc","ait",
      "Bcf","Gf","Hf","Hfus","Hsolp",
      "Hv","Lc50_fm","Ld50","Lmv","Logp",
      "Logws","Osha_twa","Pco","Pka","Tm"]
    
    data_path=r"1dataset/dataset_"+str(index)+"_"+name[index]+".npy"
    temp=np.load(data_path,allow_pickle=True)
    res_dict_dataset=temp.item()
    data_output=res_dict_dataset["data_output"]
    train_output=res_dict_dataset["train_output"]
    test_output=res_dict_dataset["test_output"]
    data_input=res_dict_dataset["data_input"]
    train_input=res_dict_dataset["train_input"]
    test_input=res_dict_dataset["test_input"]
    
    
    model_path="2model/allmodel_"+str(index)+"_"+name[index]+".sav"
    model = pickle.load(open(model_path, 'rb'))
    
    
    #for simple
    print(RMSE(test_input@model["simple_model_coef"]+model["simple_model_intercept"],test_output))
    
    
    #for normal GP
    test_pre,test_std=model["normal_model"].predict(test_input,1)
    print(RMSE(test_pre,test_output))
    # print(R_square(test_output,test_pre))
    
    
    #for warping function only
    test_input_distort=np.log(test_input+1)/np.log(model["distort_alpha"]) 
    test_pre,test_std=model["distort_model"].predict(test_input_distort,1)
    print(RMSE(test_pre,test_output))
    # print(R_square(test_output,test_pre))
    
    
    #for non-zero prior only
    test_pre=model["prior_model"].predict(test_input)+test_input@model["simple_model_coef"]+model["simple_model_intercept"]
    print(RMSE(test_pre,test_output))
    # print(R_square(test_output,test_pre))
    
    
    #for conbination(with non-zero prior and warping function)
    test_input_distort=np.log(test_input+1)/np.log(model["combination_alpha"])
    test_pre=model["combination_model"].predict(test_input_distort)+test_input@model["simple_model_coef"]+model["simple_model_intercept"]
    print(RMSE(test_pre,test_output))
    # print(R_square(test_output,test_pre))





