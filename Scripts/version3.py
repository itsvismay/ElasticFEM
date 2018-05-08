import numpy as np
import math
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import numdifftools as nd
import random
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.8f}'.format(x)})


Create Mesh
 - set U values and mesh density
 - Set fixed vertices and fixed rotations

Start Static Solve
 - Move some of the fixed vertices 
 - Run BFGS to minimize ARAP + Elastic energies
 	Energy
	 	- ARAP ~100-200 iterations
	 		- (unconstrained) SVD for rotations  
	 		- (constrained) KKT for translations
	 	- Elastic principle stretch energy 
	 		- Volume preservation term is janky 

	Gradient
		- Make sure ARAP Energy is minimized 
		- Solve for dg/ds and dr/ds 
			- Lots of tensor calc to get Hessians
			- Constrained KKT solve
