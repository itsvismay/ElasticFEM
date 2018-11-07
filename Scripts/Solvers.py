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
		self.mesh = imesh
		self.arap = iarap
		self.elastic = ielastic
		self.adder = .3
		# self.set_random_strain()
		self.s_bnds = []
		if (len(self.mesh.shandle_muscle) == 0):
			self.s_bnds = [(None, None) if i%3==2 else (1e-5, 1e6) for i in range(len(self.mesh.red_s)) ]
		else:
			for i in range(len(self.mesh.shandle_muscle)):
				if self.mesh.shandle_muscle[i]<0.5:
					#bone
					self.s_bnds.append((1-1e-3, 1+1e-3))
					self.s_bnds.append((1-1e-3, 1+1e-3))
					self.s_bnds.append((-1e-3, 1e-3))
				if self.mesh.shandle_muscle[i]>0.5:
					#muscle
					self.s_bnds.append((1e-5, 1e6))
					self.s_bnds.append((1e-5, 1e6))
					self.s_bnds.append((None, None))

		self.s_dot_bnds = []

		self.add_on = 10

	def constTimeMassJ(self, idrds):
		self.mesh.getGlobalF(updateR = True, updateS = True, updateU = False)
		RU = self.mesh.GR.dot(self.mesh.GU).toarray()
		UtPAx0 = self.mesh.GU.T.dot(self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.x0)))
		dSds = self.arap.sparseDSds()

		J = np.zeros((6*len(self.mesh.T), len(dSds)))
		J1 = np.zeros((6*len(self.mesh.T), len(self.mesh.red_r)))

		for i in range(len(self.mesh.red_r)):
			c1, c2 = -np.sin(self.mesh.red_r[i]), -np.cos(self.mesh.red_r[i])
			self.arap.PreProcJ[0][i].dot(self.mesh.red_s)*c1


		J2 = np.zeros((6*len(self.mesh.T), len(self.mesh.red_s)))
		cvec = []
		for i in range(len(self.mesh.red_r)):
			c1, c2 = np.cos(self.mesh.red_r[i]), -np.sin(self.mesh.red_r[i])
			cvec.append(c1)
			cvec.append(c2)
		cvec = np.array(cvec)
		for i in range(len(self.mesh.red_s)):
			J2[:,i] = self.arap.PreProcJ[1][i].dot(cvec)

		J = J1.dot(idrds) + J2
		M = self.mesh.getMassMatrix()

		return J.T.dot(M.dot(J))

	def iterate(self):
		pass
		# self.elastic.muscle_fibre_mag += 100
		# print(self.elastic.muscle_fibre_mag)
		# self.time += 1

	def move_g(self):
		# if(self.time%self.add_on == 0):
		# 	if(self.add_on==25):
		# 		exit()
		# 	self.adder *= -1
		# 	self.add_on = 25

		# self.mesh.red_s[6] += 0.2
		# print(self.mesh.red_s)
		# for i in range(len(self.mesh.mov)):
			# self.mesh.g[2*self.mesh.fixed[i]+1] += self.adder
			# self.mesh.z[2*self.mesh.mov[i]+1] += self.adder
		# print("moved")
		# self.mesh.red_s[3*np.arange(len(self.mesh.red_s)/3)+1] += 0.2
		# self.mesh.red_s[4] += 0.2
		self.mesh.getGlobalF(updateR=False, updateS=True, updateU=False)

	def static_solve(self):
		print("Static Solve")
		s0 = self.mesh.red_s + np.zeros(len(self.mesh.red_s))

		alpha1 =1e3
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

			J_elastic = self.elastic.PEGradient(irs = self.mesh.red_s, idgds=dgds)
			return  alpha1*J_arap + alpha2*J_elastic

		res = scipy.optimize.minimize(energy, s0, method='L-BFGS-B', bounds=self.s_bnds,  jac=jacobian, options={'gtol': 1e-6, 'ftol':1e-4, 'disp': False, 'eps':1e-8})

		for i in range(len(res.x)):
			self.mesh.red_s[i] = res.x[i]

		self.mesh.getGlobalF(updateR=False, updateS=True, updateU=False)

		# print("r1", self.mesh.red_r)
		# print("s1", res.x)
		# print("g1", self.mesh.z)
		# print(res)
		# print("static solve")

	def dynamics(self):
		print("Dynamics Solve")
		s0 = self.mesh.red_s + np.zeros(len(self.mesh.red_s))
		s_dot0 = self.mesh.red_s_dot + np.zeros(len(self.mesh.red_s))
		alpha1 =1e5
		alpha2 =1e1

		def energy(s_dot):
			for i in range(len(s_dot)):
				self.mesh.red_s[i] = s0[i] + self.timestep*s_dot[i]
				self.mesh.red_s_dot[i] = s_dot[i]

			# for i in range(len(s)):
				# self.mesh.red_s[i] = s[i]

			self.arap.updateConstUSUtPAx()
			self.arap.iterate()

			E_arap = self.arap.Energy()
			E_elastic =  self.elastic.Energy(irs=self.mesh.red_s)
			V_energy = alpha1*E_arap + alpha2*E_elastic
			print("arap, elastic ", E_arap, E_elastic)

			J_arap, dgds, drds = self.arap.Jacobian()
			JMJ = self.constTimeMassJ(idrds=drds)

			K_energy = 0.5*s_dot.T.dot(JMJ.dot(s_dot)) - s_dot.T.dot(JMJ.dot(s_dot0))
			TotEn = V_energy + K_energy
			# K_energy = s.T.dot(JMJ.dot(s)) - s.T.dot(JMJ.dot(s0)) - self.timestep*s.T.dot(JMJ.dot(s_dot0))
			# TotEn = self.timestep*self.timestep*V_energy + K_energy
			print("Energy: ", K_energy, TotEn)
			return TotEn


		def jacobian(s_dot):
			for i in range(len(s_dot)):
				self.mesh.red_s[i] = s0[i] + self.timestep*s_dot[i]
				self.mesh.red_s_dot[i] = s_dot[i]
			# for i in range(len(s)):
			# 	self.mesh.red_s[i] = s[i]
			self.arap.updateConstUSUtPAx()
			print("s")
			print(self.mesh.red_s)

			dgds = None
			# self.arap.iterate()
			J_arap, dgds, drds = self.arap.Jacobian()
			J_elastic = self.elastic.PEGradient(irs = self.mesh.red_s, idgds=dgds)
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
			self.mesh.red_s_dot[i] = res.x[i]
			self.mesh.red_s[i] = s0[i]+self.timestep*self.mesh.red_s_dot[i]
			# self.mesh.red_s_dot[i] = (self.mesh.red_s[i] - s0[i])*self.timestep

		self.arap.updateConstUSUtPAx()
		self.mesh.getGlobalF(updateR=False, updateS=True, updateU=False)
