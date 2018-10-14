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
		self.timestep = 0.01
		self.meshes = imesh
		self.araps = iarap
		self.elastics = ielastic
		self.adder = .3
		# self.set_random_strain()
		# self.bnds = [(None, None) if i%3==2 else (1e-5, 1e6) for i in range(len(self.meshes[i.red_s)) ]
		self.add_on = 10

	def constTimeMassJ(self, idrds, im):
		self.meshes[im].getGlobalF(updateR = True, updateS = True, updateU = False)
		RU = self.meshes[im].GR.dot(self.meshes[im].GU).toarray()
		UtPAx0 = self.meshes[im].GU.T.dot(self.meshes[im].getP().dot(self.meshes[im].getA().dot(self.meshes[im].x0)))
		dxdS = np.einsum("ij, k", RU, UtPAx0)
		dSds = self.araps[im].sparseDSds()

		J = np.zeros((6*len(self.meshes[im].T), len(dSds)))
		J1 = np.zeros((6*len(self.meshes[im].T), len(self.meshes[im].red_r)))

		for i in range(len(self.meshes[im].red_r)):
			c1, c2 = -np.sin(self.meshes[im].red_r[i]), -np.cos(self.meshes[im].red_r[i])
			self.araps[im].PreProcJ[0][i].dot(self.meshes[im].red_s)*c1


		J2 = np.zeros((6*len(self.meshes[im].T), len(self.meshes[im].red_s)))
		dxdS = np.einsum("ij, k", RU, UtPAx0)
		for i in range(dxdS.shape[0]):
			for j in range(len(dSds)):
				J2[i, j] = dSds[j].multiply(dxdS[i,:,:]).sum()

		J = J1.dot(idrds) + J2
		M = self.meshes[im].getMassMatrix()

		return J.T.dot(M.dot(J))
	

	def iterate(self):
		pass
		# self.elastic.muscle_fibre_mag += 100
		# print(self.elastic.muscle_fibre_mag)
		# self.time += 1

	def move_g(self, im):
		if(self.time%self.add_on == 0):
			if(self.add_on==25):
				exit()
			self.adder *= -1
			self.add_on = 25

		for i in range(len(self.meshes[im].mov)):
			self.meshes[im].g[2*self.meshes[im].mov[i]+1] += self.adder
			# self.mesh.g[2*self.mov[i]+1] += self.adder
		# print("moved")
		# self.mesh.red_s[3*np.arange(len(self.mesh.red_s)/3)+1] += 0.2
		# self.mesh.red_s[4] += 0.2
		# self.mesh.getGlobalF(updateR=False, updateS=True, updateU=False)
		

	def toggle_muscle_group(self, num):
		#toggle 1 to 0 and 0 to 1
		self.meshes[0].u_toggle[self.mesh.u_clusters_element_map[num]] = self.mesh.u_toggle[self.mesh.u_clusters_element_map[num]] == 0
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

			J_elastic = -1*self.elastic.Forces(irs = self.mesh.red_s, idgds=dgds)
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

	def dynamics(self, im):
		print("Dynamics Solve")
		s0 = self.meshes[im].red_s + np.zeros(len(self.meshes[im].red_s))
		s_dot0 = self.meshes[im].red_s_dot + np.zeros(len(self.meshes[im].red_s))
		alpha1 =1e5
		alpha2 =1e1

		def energy(s_dot):
			for i in range(len(s_dot)):
				self.meshes[im].red_s[i] = s0[i] + self.timestep*s_dot[i]
				self.meshes[im].red_s_dot[i] = s_dot[i]
			# for i in range(len(s)):
				# self.meshes[im].red_s[i] = s[i]

			self.araps[im].updateConstUSUtPAx()

			self.araps[im].iterate()
			E_arap = self.araps[im].Energy()
			E_elastic =  self.elastics[im].Energy(irs=self.meshes[im].red_s)
			V_energy = alpha1*E_arap + alpha2*E_elastic
			print("arap, elastic ", E_arap, E_elastic)
			
			J_arap, dgds, drds = self.araps[im].Jacobian()
			JMJ = self.constTimeMassJ(idrds=drds)

			K_energy = 0.5*s_dot.T.dot(JMJ.dot(s_dot)) - s_dot.T.dot(JMJ.dot(s_dot0))
			TotEn = V_energy + K_energy
			# K_energy = s.T.dot(JMJ.dot(s)) - s.T.dot(JMJ.dot(s0)) - self.timestep*s.T.dot(JMJ.dot(s_dot0))
			# TotEn = self.timestep*self.timestep*V_energy + K_energy
			print("Energy: ", K_energy, TotEn)
			return TotEn


		def jacobian(s_dot):
			for i in range(len(s_dot)):
				self.meshes[im].red_s[i] = s0[i] + self.timestep*s_dot[i]
				self.meshes[im].red_s_dot[i] = s_dot[i]
			# for i in range(len(s)):
			# 	self.meshes[im].red_s[i] = s[i]
			self.araps[im].updateConstUSUtPAx()
			print("s")
			print(self.meshes[im].red_s)
			
			dgds = None
			# self.araps[im].iterate()
			J_arap, dgds, drds = self.araps[im].Jacobian()
			J_elastic = self.elastics[im].PEGradient(irs = self.meshes[im].red_s, idgds=dgds)
			PEGradient =  alpha1*J_arap + alpha2*J_elastic

			JMJ = self.constTimeMassJ(idrds=drds)
			# KEGradient =  JMJ.dot(s) - JMJ.dot(s0) - self.timestep*JMJ.dot(s_dot0)
			# return self.timestep*self.timestep*PEGradient + KEGradient
			KEGradient = JMJ.dot(s_dot) - JMJ.dot(s_dot0)
			totGrad = KEGradient + self.timestep*PEGradient
			print(totGrad)
			return totGrad

		res = scipy.optimize.minimize(energy, s_dot0, method='L-BFGS-B',  jac=jacobian, options={'gtol': 1e-6, 'ftol':1e-4, 'disp': False, 'eps':1e-8})
		
		for i in range(len(s0)):
			self.meshes[im].red_s_dot[i] = res.x[i]
			self.meshes[im].red_s[i] = s0[i]+self.timestep*self.meshes[im].red_s_dot[i]
			# self.meshes[im].red_s_dot[i] = (self.meshes[im].red_s[i] - s0[i])*self.timestep
		
		self.araps[im].updateConstUSUtPAx()
		self.meshes[im].getGlobalF(updateR=False, updateS=True, updateU=False)