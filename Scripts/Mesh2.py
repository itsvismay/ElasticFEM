import numpy as np
import math
from collections import defaultdict
from sets import Set
import json
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy import sparse
from scipy.cluster.vq import vq, kmeans, whiten
import random
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.5f}'.format(x)})
from iglhelpers import *
temp_png = os.path.join(os.getcwd(),"out.png")

from Helpers import *

class Mesh2:
	def __init__(self, V, T, u, s_ind, r_ind, sW, emat, fix, mov):
		self.V = V
		self.T = T
		self.FIX = fix
		self.MOV = mov
		self.elem_youngs = np.array([600000 if e<0.5 else 6e8 for e in emat])
		self.elem_poisson = np.array([0.45 if e<0.5 else 0.45 for e in emat])
		
		self.x0 = np.ravel(self.V)
		self.g = np.zeros(len(self.V)*2)
		self.u = u
		

