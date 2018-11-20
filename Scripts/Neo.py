import numpy as np
import math
from collections import defaultdict
from sets import Set
import json
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy import sparse
from Helpers import *
import datetime
import random
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
from iglhelpers import *
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.5f}'.format(x)})

temp_png = os.path.join(os.getcwd(),"out.png")

class NeohookeanElastic:

	def __init__(self, imesh):
		self.mesh = imesh
		self.f = np.zeros(2*len(self.mesh.T))
		self.v = np.zeros(2*len(self.mesh.V))
		# self.M = self.mesh.getMassMatrix()
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix()

		self.dimensions = 2

		self.grav = np.array([0, 981]) #cm/s^2
		self.rho = 6.4 #in Grams/cm^2

		self.muscle_fibre_mag = 2000000 #g*cm/s^2

		self.fastMuscleEnergy = []

		self.preComputeFastMatrices()

	def preComputeFastMatrices(self):
		ME = sparse.kron(sparse.eye(len(self.mesh.T)), np.array([[1,0,0],[0,0,1]]))
		MEsW = ME.dot(self.mesh.sW)
		U_tog = 0.5*self.muscle_fibre_mag*sparse.diags(np.kron(self.mesh.u_toggle, np.array([1,1]))).tocsc()
		fastMusc = MEsW.T.dot(U_tog.dot(MEsW))
		# s = self.mesh.sW.dot(self.mesh.red_s)
		# # print(s)
		# print(ME.dot(s))
		# print(MEsW.dot(self.mesh.red_s))
		# print(s*s)
		# print(self.mesh.red_s.T.dot(MEsW.T.dot(U_tog.dot(MEsW.dot(self.mesh.red_s)))))
		self.fastMuscleEnergy.append(fastMusc) 

	def GravityElementEnergy(self, rho, grav, cag, area, t):
		e = rho*area*grav.dot(cag)
		return e

	def GravityEnergy(self):
		Eg = 0

		CAg = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.getg()))
		for t in range(len(self.mesh.T)):
			area = self.mesh.areas[t]
			Eg += self.GravityElementEnergy(self.rho, self.mesh.getU(t).dot(self.grav), CAg[6*t:6*t+2], area, t)

		return Eg

	def GravityElementForce(self, rho, area, grav, cadgds, t):
		gt = -rho*area*np.dot(grav, cadgds)
		return gt

	def GravityForce(self, dzds):
		fg = np.zeros(len(self.mesh.red_s))
		Ax = self.mesh.getA().dot(self.mesh.x0)

		CAdzds = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.G.dot(dzds)))

		for t in range(len(self.mesh.T)):
			area = self.mesh.areas[t]
			gv = self.mesh.getU(t).dot(self.grav)
			fg += self.GravityElementForce(self.rho, area, gv, CAdzds[6*t:6*t+2, :], t)

		return fg

	def WikipediaPrinStretchElementForce(self, a, rs, wx, wy, wo, md, mc):
		t_0 = np.dot(wx, rs)
		t_1 = np.dot(wy, rs)
		t_2 = np.log(((t_0 * t_1) - (np.dot(wo, rs) ** 2)))
		t_3 = np.dot(rs, wy)
		t_4 = np.dot(rs, wx)
		t_5 = np.dot(rs, wo)
		t_6 = ((t_4 * t_3) - (t_5 ** 2))
		t_7 = (mc / t_6)
		t_8 = np.log(t_6)
		t_9 = (((2 * md) * t_8) / (4 * t_6))
		gradient = ((((((mc * wx) + (mc * wy)) - ((((t_7 * t_3) * wx) + ((t_7 * t_4) * wy)) - ((((2 * mc) / t_6) * t_5) * wo))) + ((t_9 * t_3) * wx)) + ((t_9 * t_4) * wy)) - ((((md * t_8) / t_6) * t_5) * wo))
		return -gradient
	
	def WikipediaForce(self, _rs):
		force = np.zeros(len(self.mesh.red_s))
		Ax = self.mesh.getA().dot(self.mesh.x0)
		for t in range(len(self.mesh.T)):
			if self.mesh.u_toggle[t]>0.5:
				md = 0.5*(self.mesh.elem_youngs[t]*self.mesh.elem_poissons[t])/((1.0+self.mesh.elem_poissons[t])*(1.0-2.0*self.mesh.elem_poissons[t]))
				mc = 0.5*self.mesh.elem_youngs[t]/(2.0*(1.0+self.mesh.elem_poissons[t]))
				area = self.mesh.areas[t]
				force += self.WikipediaPrinStretchElementForce(area, _rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:], md, mc)

		return force

	def WikipediaPrinStretchElementEnergy(self, area, rs, wx, wy, wo, md, mc):
		#MATH version
		#E = mc*((wx'*rs) + (wy'*rs)-2 - log(wx'*rs*wy'*rs - (wo'*rs)^2)) + (md/4)*(log(wx'*rs*wy'*rs - (wo'*rs)^2)^2)
		
		sx = np.dot(wx, rs)
		sy = np.dot(wy, rs)
		so = np.dot(rs, wo)
		t_2 = np.log(((sx * sy) - (so ** 2)))

		if(sx<=0 or sy<=0 or sx*sy-so*so<=0):
			return 1e40

		functionValue = ((mc * ((sx - 2 + sy) - t_2) ) + ((md * (t_2 ** 2)) / 4))
		return functionValue

	def WikipediaEnergy(self,_rs):
		E = 0
		Ax = self.mesh.getA().dot(self.mesh.x0)
		for t in range(len(self.mesh.T)):
			if self.mesh.u_toggle[t]>0.5:
				md = 0.5*(self.mesh.elem_youngs[t]*self.mesh.elem_poissons[t])/((1.0+self.mesh.elem_poissons[t])*(1.0-2.0*self.mesh.elem_poissons[t]))
				mc = 0.5*self.mesh.elem_youngs[t]/(2.0*(1.0+self.mesh.elem_poissons[t]))
				area = self.mesh.areas[t]
				E += self.WikipediaPrinStretchElementEnergy(area, _rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:],self.mesh.sW[3*t+2,:], md, mc)

		return E

	def MuscleElementEnergy(self, rs,wx, wy, wo, u, tog):
		sx = wx.dot(rs)
		sy = wy.dot(rs)
		so = wo.dot(rs)
		if(sx<=0 or sy<=0 or sx*sy-so*so<=0):
			return 1e40

		c = np.array([[sx, so],[so, sy]])
		print(c.dot(c.T))
		return 0.5*self.muscle_fibre_mag*tog*(u.dot(c.dot(c.T.dot(u.T))))

	def MuscleEnergy(self, _rs):
		En = 0
		En = self.mesh.red_s.T.dot(self.fastMuscleEnergy[0].dot(self.mesh.red_s))
		

		# for t in range(len(self.mesh.T)):
		# 	alpha = self.mesh.u[t]
		# 	c, s = np.cos(alpha), np.sin(alpha)
		# 	u = np.array([c,s]).dot(np.array([[c,-s], [s, c]]))
		# 	toggle = self.mesh.u_toggle[t]
		# 	E = self.MuscleElementEnergy(_rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:],u, toggle)
		# 	En += E
		# exit()
		return En

	def MuscleElementForce(self, rs, wx, wy, wo, u1, u2, tog):

		t0 = self.muscle_fibre_mag*u1*u1*tog
		t1 = t0*(rs.dot(wx)*wx + rs.dot(wo)*wo)


		t2 = 0.5*self.muscle_fibre_mag*u1*u2*tog
		t3 = t2*rs.dot(wo)
		t4 = t3*wx + t2*rs.dot(wx)*wo + t3*wy + t2*rs.dot(wy)*wo

		t5 = self.muscle_fibre_mag *u2*u2*tog
		t6 = t5*(rs.dot(wy)*wy + rs.dot(wo)*wo)

		return -1*(t1 + 2*t4 + t6)

	def MuscleForce(self, _rs):
		force = np.zeros(len(self.mesh.red_s))
		force -= 2*self.fastMuscleEnergy[0].dot(self.mesh.red_s)
		# for t in range(len(self.mesh.T)):
		# 	alpha = self.mesh.u[t]
		# 	c, s = np.cos(alpha), np.sin(alpha)
		# 	u = np.array([c,s]).dot(np.array([[c,-s],[s, c]]))
		# 	toggle = self.mesh.u_toggle[t]
		# 	force += self.MuscleElementForce(_rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:],u[0], u[1], toggle)

		return force

	def Energy(self, irs):
		e2 = self.WikipediaEnergy(_rs=irs)
		# e1 = self.GravityEnergy()
		e3 = self.MuscleEnergy(_rs=irs)
		print("e123", e2, e3)
		return e2 + e3

	def PEGradient(self, irs, idgds):
		f2 = self.WikipediaForce(_rs=irs)
		# f1 =  -1*self.GravityForce(idgds)
		f3 = self.MuscleForce(_rs=irs)
		return -(f2 + f3)

	def JMJ_MassMatrix(self, idrds, idRdr, idSds):
		self.mesh.getGlobalF(updateR = True, updateS = True, updateU = False)
		RU = self.mesh.GR.dot(self.mesh.GU).toarray()
		UtPAx0 = self.mesh.GU.T.dot(self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.x0)))
		USUtPAx0 = self.mesh.GU.dot(self.mesh.GS.dot(UtPAx0))

		dxdR = np.einsum("ij, k", np.eye(len(USUtPAx0), len(USUtPAx0)), USUtPAx0)
		dxdS = np.einsum("ij, k", RU, UtPAx0)
		return dxdR, dxdS
				