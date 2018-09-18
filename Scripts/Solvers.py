import numpy as np
import math
from collections import defaultdict
from sets import Set
import json
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy import sparse

import random
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.5f}'.format(x)})
from iglhelpers import *

temp_png = os.path.join(os.getcwd(),"out.png")

class TimeIntegrator:

	def __init__(self, imesh, iarap, ielastic = None):
		self.time = 1
		self.timestep = 0.001
		self.mesh = imesh
		self.arap = iarap
		self.elastic = ielastic
		self.adder = .3
		# self.set_random_strain()
		self.mov = self.mesh.mov
		self.bnds = [(0, None) if i%3==2 else (1e-5, None) for i in range(len(self.mesh.red_s)) ]
		self.add_on = 10
	
	def set_strain(self):
		for i in range(len(self.mesh.T)):
			pass
			# self.mesh.q[3*i+1] =1.01+ np.sin(self.time)
			# self.mesh.q[3*i +2] = 1.01 + np.sin(self.time)

	def set_random_strain(self):
		for i in range(len(self.mesh.T)):
			# self.mesh.q[3*i + 1] = 1.01 + np.random.uniform(0,2)
			self.mesh.q[3*i + 2] = 1.0 + np.random.uniform(0.001,0.1)

	def iterate(self):

		# if(self.time%10 == 0):
		# 	self.adder *= -1

		# self.mesh.g[2*self.mov+1] -= self.adder
		self.elastic.muscle_fibre_mag += 100
		print(self.elastic.muscle_fibre_mag)
		self.time += 1

	def move_g(self):
		if(self.time%self.add_on == 0):
			if(self.add_on==25):
				exit()
			self.adder *= -1
			self.add_on = 25

		for i in range(len(self.mov)):
			self.mesh.g[2*self.mov[i]] += self.adder
			# self.mesh.g[2*self.mov[i]+1] += self.adder
		# print("moved")
		# self.mesh.red_s[3*np.arange(len(self.mesh.red_s)/3)+1] += 0.2
		# self.mesh.red_s[4] += 0.2
		# self.mesh.getGlobalF(updateR=False, updateS=True, updateU=False)

	def toggle_muscle_group(self, num):
		#toggle 1 to 0 and 0 to 1
		self.mesh.u_toggle[self.mesh.u_clusters_element_map[num]] = self.mesh.u_toggle[self.mesh.u_clusters_element_map[num]] == 0
		self.static_solve()
		
	def static_solve(self):
		print("Static Solve")
		s0 = self.mesh.red_s + np.zeros(len(self.mesh.red_s))

		alpha1 =1e5
		alpha2 =1e1

		def energy(s):
			for i in range(len(s)):
				self.mesh.red_s[i] = s[i]
			self.arap.updateConstUSUtPAx()


			self.arap.iterate()

			E_arap = self.arap.Energy()
			E_elastic =  self.elastic.Energy(irs=self.mesh.red_s)
			# print("s", self.mesh.red_s)
			# print("E", E_arap, E_elastic)

			return alpha1*E_arap + alpha2*E_elastic

		def jacobian(s):
			for i in range(len(s)):
				self.mesh.red_s[i] = s[i]
			self.arap.updateConstUSUtPAx()
			
			dgds = None
			# self.arap.iterate()
			J_arap, dgds, drds = self.arap.Jacobian()

			J_elastic = -1*self.elastic.Forces(irs = self.mesh.red_s, idgds=dgds, idrds = drds)
			dRdr = self.arap.sparseDRdr()
			dSds = self.arap.sparseDSds()	
			JMJ_Mass = self.elastic.JMJ_MassMatrix(idrds=drds, idRdr =dRdr, idSds=dSds)
			return  alpha1*J_arap + alpha2*J_elastic

		res = scipy.optimize.minimize(energy, s0, method='L-BFGS-B', bounds=self.bnds,  jac=jacobian, options={'gtol': 1e-6, 'ftol':1e-4, 'disp': False, 'eps':1e-8})

		for i in range(len(res.x)):
			self.mesh.red_s[i] = res.x[i]

		self.mesh.getGlobalF(updateR=False, updateS=True, updateU=False)

		# print("r1", self.mesh.red_r)
		# print("s1", res.x)
		# print("g1", self.mesh.z)
		# print(res)
		# print("static solve")