import numpy as np
import math
import json
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy import sparse
from collections import defaultdict
from sets import Set
import datetime
import random
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.8f}'.format(x)})
from iglhelpers import *
U = []
for i in range(2,6):
	u = igl.eigen.MatrixXd()
	igl.readDMAT("snapshots/"+str(i)+".dmat", u)
	U.append(e2p(u))

A = np.array(U)
print(A[:,:,0].shape)
red, S, V = np.linalg.svd(A[:,:,0].T)
print(red[:,0:4])