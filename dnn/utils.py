###
# File: utils.py
# Description: 
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Monday, 31st October 2022, 2:17:16 pm
# Last Modified: Monday, 31st October 2022, 2:17:27 pm
#  
# Copyright (c) 2022, Bruno R. de Abreu, National Center for Supercomputing Applications.
# All rights reserved.
# License: This program and the accompanying materials are made available to any individual
#          under the citation condition that follows: On the event that the software is
#          used to generate data that is used implicitly or explicitly for research
#          purposes, proper acknowledgment must be provided in the citations section of
#          publications. This includes both the author's name and the National Center
#          for Supercomputing Applications. If you are uncertain about how to do
#          so, please check this page: https://github.com/babreu-ncsa/cite-me.
#          This software cannot be used for commercial purposes in any way whatsoever.
#          Omitting this license when redistributing the code is strongly disencouraged.
#          The software is provided without warranty of any kind. In no event shall the
#          author or copyright holders be liable for any kind of claim in connection to
#          the software and its usage.
###

import numpy as np

def initialize_parameters(layers_dims):
    """
    Initializes parameters randomly. Returns dictionary with parameters.

    Arguments:
        - layers_dims: list of integers, each integer represents the size of the layer
    
    Returns:
        - parameters: dictionary containing the parameters W and b for each layer
    """
    parameters = {}

    for layer in range(1, len(layers_dims)):
        parameters['W' + str(layer)] = np.random.randn(layers_dims[layer], layers_dims[layer-1]) * np.sqrt(1. / layers_dims[layer])
        parameters['b' + str(layer)] = np.random.randn(layers_dims[layer], 1) * np.sqrt(1. / layers_dims[layer])

    return parameters


def one_hot(y):
    """
    Converts a vector of labels into a one-hot matrix.

    Arguments:
        - y: vector of labels
    
    Returns:
        - one_hot: one-hot matrix
    """
    one_hot = np.zeros((y.shape[0], y.max() + 1))
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot.T