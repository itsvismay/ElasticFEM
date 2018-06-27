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
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.3f}'.format(x)})
from iglhelpers import *

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

		self.grav = np.array([0,9.81])
		self.rho = 10

		self.muscle_fiber_mag_target = 1000
		self.muscle_fibre_mag = 200

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

		t_0 = np.dot(wo, rs)
		t_1 = np.dot(wx, rs)
		t_2 = np.dot(wy, rs)



		if(t_1<=0 or t_2<=0 or t_1*t_2-t_0*t_0<=0):
			return 1e40

		t_3 = ((t_1 * t_2) - (t_0 * t_0))
		t_4 = np.dot(rs, wo)
		t_5 = (2.0 / 6)
		t_6 = np.dot(rs, wx)
		t_7 = np.dot(rs, wy)
		t_8 = ((t_6 * t_7) - (t_4 * t_4))
		t_9 = ((mc * (t_8 ** -(1 + t_5))) * (t_6 + t_7))
		t_10 = (((2 * a) * t_9) / 6.0)
		t_11 = -t_5
		t_12 = ((a * mc) * (t_8 ** t_11))
		t_13 = (1.0 / 2)
		t_14 = ((a * md) * ((t_8 ** (t_13 - 1)) * ((t_8 ** t_13) - 1)))
		functionValue = (a * ((mc * (((t_3 ** t_11) * (t_1 + t_2)) - 2)) + (md * (((t_3 ** t_13) - 1) ** 2))))
		gradient = (((((((((((4 * a) * t_9) / 6.0) * t_4) * wo) - (((t_10 * t_7) * wx) + ((t_10 * t_6) * wy))) + (t_12 * wx)) + (t_12 * wy)) + ((t_14 * t_7) * wx)) + ((t_14 * t_6) * wy)) - ((((4 * t_14) / 2.0) * t_4) * wo))

		return -gradient

	def WikipediaForce(self, _rs):
		force = np.zeros(len(self.mesh.red_s))
		Ax = self.mesh.getA().dot(self.mesh.x0)
		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			force += self.WikipediaPrinStretchElementForce(area, _rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:] )

		return force

	def WikipediaPrinStretchElementEnergy(self, area, rs, wx, wy, wo):

		m_D = self.mD
		m_C = self.mC

		#MATH version
		# c = [[sx, so],[so, sy]]
		# I1 = np.trace(c)
		# I3 = np.linalg.det(c)
		# J = math.sqrt(I3)
		# I1_b = math.pow(J, -2.0/3)*(I1)
		# v1 = area*(m_C*(I1_b -2) + m_D*(J-1)*(J-1))

		sx = wx.dot(rs)
		sy = wy.dot(rs)
		so = wo.dot(rs)
		# print(sx, sy, so)
		if(sx<=0 or sy<=0 or sx*sy-so*so<=0):
			return 1e40
		term1 = m_C*(math.pow(math.sqrt(sx*sy - so*so), -2.0/3)*(sx + sy) - 2)
		term2 = m_D*math.pow((math.pow(sx*sy -so*so, 1/2.0) - 1), 2)
		v2 = area*(term1 + term2)
		# print(math.pow(math.sqrt(sx*sy - so*so), -2.0/3), (sx + sy), v2)
		return v2

	def WikipediaEnergy(self,_rs):
		E = 0

		Ax = self.mesh.getA().dot(self.mesh.x0)

		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			E += self.WikipediaPrinStretchElementEnergy(area, _rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:],self.mesh.sW[3*t+2,:])

		return E

	def MuscleElementEnergy(self, rs,wx, wy, wo, u):
		sx = wx.dot(rs)
		sy = wy.dot(rs)
		so = wo.dot(rs)
		if(sx<=0 or sy<=0 or sx*sy-so*so<=0):
			return 1e40

		c = np.array([[sx, so],[so, sy]])

		return 0.5*self.muscle_fibre_mag*(u.dot(c.dot(c.T.dot(u.T))))

	def MuscleEnergy(self, _rs):
		En = 0
		for t in range(len(self.mesh.T)):
			alpha = self.mesh.u[t]
			c, s = np.cos(alpha), np.sin(alpha)
			u = np.array([c,s]).dot(np.array([[c,-s],[s, c]]))

			En += self.MuscleElementEnergy(_rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:],u)

		return En

	def MuscleElementForce(self, rs, wx, wy, wo, u1, u2):

		t0 = self.muscle_fibre_mag *u1*u1
		t1 = t0*(rs.dot(wx)*wx + rs.dot(wo)*wo)


		t2 = 0.5*self.muscle_fibre_mag*u1*u2
		t3 = t2*rs.dot(wo)
		t4 = t3*wx + t2*rs.dot(wx)*wo + t3*wy + t2*rs.dot(wy)*wo

		t5 = self.muscle_fibre_mag *u2*u2
		t6 = t5*(rs.dot(wy)*wy + rs.dot(wo)*wo)

		return -1*(t1 + 2*t4 + t6)

	def MuscleForce(self, _rs):
		force = np.zeros(len(self.mesh.red_s))
		for t in range(len(self.mesh.T)):
			alpha = self.mesh.u[t]
			c, s = np.cos(alpha), np.sin(alpha)
			u = np.array([c,s]).dot(np.array([[c,-s],[s, c]]))
			force += self.MuscleElementForce(_rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:],u[0], u[1])

		return force

	def Energy(self, irs):
		e2 = self.WikipediaEnergy(_rs=irs)
		e1 = -1*self.GravityEnergy()
		e3 = self.MuscleEnergy(_rs=irs)
		# print("e3 ", e1, e2, e3)
		return e1 + e2 + e3

	def Forces(self, irs, idgds):
		f2 = self.WikipediaForce(_rs=irs)
		f1 =  -1*self.GravityForce(idgds)
		f3 = self.MuscleForce(_rs=irs)
		# print("f3 ", f3)
		return f1 + f2 + f3