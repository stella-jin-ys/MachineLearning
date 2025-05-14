#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 09:20:17 2025

@author: stella
"""

import pandas as pd
from pickle import load 
import numpy as np

data = pd.read_csv('scaled_data-csv')

scaler_file = open('jena_scaler.pkl', 'rb')

scaler = load(scaler_file)

data_array = np.array(data['Scaled temp'])

data_array = data_array.reshape(-1,1)

data_original =  scaler.inverse_transform(data_array)

print(data_original[-1])

inversed_prediction = scaler.inverse_transform([[0.5]])

print(inversed_prediction)