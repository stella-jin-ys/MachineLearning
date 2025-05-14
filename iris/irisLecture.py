#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:50:47 2025

@author: stella
"""

Virginica: [0,0,1]

5, 3.3, 1.4, .2 : [1,0,0]

Hyperparameters

Tuple: Like a list but with paranthesis instead of square brackets.
(4,)
[4]

[5.1, 3.5, 1.4, .2]
[w1, w2, w3, w4]

5.1*w1 + 3.5*w2 + 1.4*w3 + .2*w4 + bias

^y = [[0.76869136 0.46562028 0.12968335]] : prediction during the training

y = [[1,0,0]]

loss = -(1*log(0.76869136) + 0*log(0.46562028) + 0*log(0.12968335)) = -log(0.76869136) = 0.12

Back-propagation: the process of the model adjusting weights and biases based on the size of the loss.