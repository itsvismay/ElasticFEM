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
	def __init__(self, iVTU=None, ito_fix=[], ito_mov=[], modes_used=None, read_in=False, muscle=True):
		self.V = np.array(iVTU[0])
		self.T = iVTU[1]
		print("MeshSize:")
		print(self.V.shape, self.T.shape)
		self.mov = list(set(ito_mov))

		self.x0 = np.ravel(self.V)
		self.g = np.zeros(len(self.V)*2)#+np.ravel(self.V)
		self.u = iVTU[2] if iVTU[2] is not None else np.zeros(len(self.T))
		self.u_clusters_element_map = None 
		self.u_toggle = np.ones(len(self.T))

		self.number_of_verts_fixed_on_element = None
		self.P = None
		self.A = None
		self.C = None
		self.N = None
		self.BLOCK = None
		self.ANTI_BLOCK = None
		self.Mass = None

		self.G = None
		self.Q = None
		self.Eigvals = None
		self.z = None
		
		t_size = len(self.T)
		self.GF = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GR = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GS = sparse.diags([np.zeros(6*t_size-1), np.ones(6*t_size), np.zeros(6*t_size-1)],[-1,0,1]).tolil()
		self.GU = sparse.diags([np.ones(6*t_size)],[0]).tolil()
		self.GUSUt = sparse.diags([np.zeros(6*t_size-1), np.ones(6*t_size), np.zeros(6*t_size-1)],[-1,0,1]).tocsc()


		# Modal analysis
		if modes_used is None:
			self.Q = None 
			self.G = np.eye(2*len(self.V))
		else:
			self.Q =self.setupModes(modes_used=modes_used)
			self.G = self.Q[:,:]
		self.z = np.zeros(self.G.shape[1])

		# Rotation clusterings
		self.red_r = None
		self.r_element_cluster_map = None
		self.r_cluster_element_map = defaultdict(list)
		self.RotationBLOCK = None
		self.setupRotClusters(rclusters=False, nrc=nrc)

		#S skinnings
		self.red_s = None
		self.s_handles_ind = None
		self.sW = None
		self.setupStrainSkinnings(shandles=False, nsh=nsh)
		self.red_s_dot = np.zeros(len(self.red_s))

		print("\n+ Setup GF")
		self.getGlobalF(updateR = True, updateS = True, updateU=True)
		print("- Done with GF")

		Ax = self.getA().dot(self.x0)
		self.areas = []
		for t in range(len(self.T)):
			self.areas.append(get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6]))
	