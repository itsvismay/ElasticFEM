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


		self.mD = 0.5*(self.mesh.youngs*self.mesh.poissons)/((1.0+self.mesh.poissons)*(1.0-2.0*self.mesh.poissons))
		self.mC = 0.5*self.mesh.youngs/(2.0*(1.0+self.mesh.poissons))
		self.dimensions = 2

		self.grav = np.array([0, 9.81])
		self.rho = 10

		self.muscle_fiber_mag_target = 100
		self.muscle_fibre_mag = 50000

	def GravityElementEnergy(self, rho, grav, cag, area, t):
		e = rho*area*grav.dot(cag)
		return e

	def GravityEnergy(self):
		Eg = 0

		Ax = self.mesh.getA().dot(self.mesh.x0)
		CAg = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.getg()))

		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			Ug = self.mesh.getU(t).dot(self.grav)
			Eg += self.GravityElementEnergy(self.rho, Ug, CAg[6*t:6*t+2], area, t)

		return Eg

	def GravityElementForce(self, rho, area, grav, cadgds, t):
		gt = -rho*area*np.dot(grav, cadgds)
		return gt

	def GravityForce(self, dzds):
		fg = np.zeros(len(self.mesh.red_s))
		Ax = self.mesh.getA().dot(self.mesh.x0)

		CAdzds = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.G.dot(dzds)))

		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			gv = self.mesh.getU(t).dot(self.grav)
			fg += self.GravityElementForce(self.rho, area, gv, CAdzds[6*t:6*t+2, :], t)

		return fg

	def WikipediaPrinStretchElementForce(self, a, rs, wx, wy, wo):
		md = self.mD
		mc = self.mC
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
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			force += self.WikipediaPrinStretchElementForce(area, _rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:] )

		return force

	def WikipediaPrinStretchElementEnergy(self, area, rs, wx, wy, wo):

		md = self.mD
		mc = self.mC

		#MATH version
		#E = mc*((wx'*rs) + (wy'*rs)-2 - log(wx'*rs*wy'*rs - (wo'*rs)^2)) + (md/4)*(log(wx'*rs*wy'*rs - (wo'*rs)^2)^2)
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
		if(t_0<=0 or t_1<=0 or t_0*t_1-t_5*t_5<=0):
			return 1e40

		functionValue = ((mc * (((t_0 - 2) + t_1) - t_2)) + ((md * (t_2 ** 2)) / 4))
		return functionValue

	def WikipediaEnergy(self,_rs):
		E = 0

		Ax = self.mesh.getA().dot(self.mesh.x0)

		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			E += self.WikipediaPrinStretchElementEnergy(area, _rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:],self.mesh.sW[3*t+2,:])

		return E

	def MuscleElementEnergy(self, rs,wx, wy, wo, u, tog):
		sx = wx.dot(rs)
		sy = wy.dot(rs)
		so = wo.dot(rs)
		if(sx<=0 or sy<=0 or sx*sy-so*so<=0):
			return 1e40

		c = np.array([[sx, so],[so, sy]])

		return 0.5*self.muscle_fibre_mag*tog*(u.dot(c.dot(c.T.dot(u.T))))

	def MuscleEnergy(self, _rs):
		En = 0
		for t in range(len(self.mesh.T)):
			alpha = self.mesh.u[t]
			c, s = np.cos(alpha), np.sin(alpha)
			u = np.array([c,s]).dot(np.array([[c,-s],[s, c]]))
			toggle = self.mesh.u_toggle[t]
			E = self.MuscleElementEnergy(_rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:],u, toggle)
			En += E
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
		for t in range(len(self.mesh.T)):
			alpha = self.mesh.u[t]
			c, s = np.cos(alpha), np.sin(alpha)
			u = np.array([c,s]).dot(np.array([[c,-s],[s, c]]))
			toggle = self.mesh.u_toggle[t]
			force += self.MuscleElementForce(_rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:],u[0], u[1], toggle)

		return force

	def Energy(self, irs):
		e2 = self.WikipediaEnergy(_rs=irs)
		# e1 = self.GravityEnergy()
		e3 = self.MuscleEnergy(_rs=irs)
		print("e123", e2, e3)
		return e2 + e3

	def Forces(self, irs, idgds):
		f2 = self.WikipediaForce(_rs=irs)
		# f1 =  -1*self.GravityForce(idgds)
		f3 = self.MuscleForce(_rs=irs)
		return f2 + f3