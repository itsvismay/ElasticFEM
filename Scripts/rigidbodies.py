import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
from iglhelpers import *
import random
import numpy as np
import math
from collections import defaultdict
from sets import Set
import datetime
import json
import scipy
from scipy import sparse
from scipy import linalg

class SE3:
	def __init__(self, whd, E, phi):
		self.whd = whd
		self.E = E
		self.phi = phi
		self.M = np.zeros((6,6))
		self.createM()



	def invE(self, E):
		R = E[0:3, 0:3]
		p = E[0:3, 3]
		Einv = np.vstack((np.hstack((R.T, -R.T.dot(p)[:,np.newaxis])), np.array([[0,0,0,1]])))
		return Einv

	def cross_prod(self, v1, v2):
		return np.cross(v1, v2)

	def Gamma(self, r):
		brac_ix = self.brac(r[0:3])
		return np.hstack((brac_ix.T, np.eye(3)))

	def brac(self, x):
		if len(x)< 6:
			return np.array([[0, -x[2], x[1]], 
							[x[2], 0 , -x[0]], 
							[-x[1], x[0], 0]])
		else:
			return np.array([[ 0, -x[2], x[1], x[3]],
							[ x[2], 0, -x[0], x[4]],
							[ -x[1], x[0], 0, x[5]],
							[0,0,0,0]])

	def Adjoint(self, E):
		A = np.zeros((6,6))
		R = E[0:3, 0:3]
		p = E[0:3, 3]
		A[0:3, 0:3] = R 
		A[3:6, 3:6] = R 
		A[3:6, 0:3] = self.brac(p).dot(R)
		return A

	def adjoint(self, phi):
		a = np.zeros((6,6))
		w = phi[0:3]
		v = phi[3:6]
		W = self.brac(w)
		a[0:3, 0:3] = W 
		a[3:6, 0:3] = self.brac(v)
		a[3:6, 3:6] = W
		return a

	def Addot(self, E, phi):
		dA = np.zeros((6,6))
		R = E[0:3, 0:3]
		p = E[0:3, 3]
		W = phi[0:3, 0:3]
		v = phi[0:3, 3]

	def expm(self, phi):
		w = phi[0:3]
		v = np.array([phi[3:6]])

		brac_w = self.brac(w)
		S = np.vstack((np.hstack((brac_w, v.T)), np.zeros((1,4))))
		return scipy.linalg.expm(self.brac(phi))

	def logm(self, E):
		return scipy.linalg.logm(E)

	def createM(self):
		m = np.zeros(6)
		density = 1.0
		mass = 1.0*self.whd[0]*self.whd[1]*self.whd[2]
		m[0] = (1.0/12)*mass*self.whd[[1,2]].dot(self.whd[[1,2]])
		m[1] = (1.0/12)*mass*self.whd[[0,2]].dot(self.whd[[0,2]])
		m[2] = (1.0/12)*mass*self.whd[[0,1]].dot(self.whd[[0,1]])
		m[3] = mass
		m[4] = mass
		m[5] = mass
		self.M = np.diag(m)
		print(self.M)


class Solver:

	def __init__(self, rbds):
		self.h = 1e-3
		self.time = 0
		self.rbds = rbds
		self.grav = np.array([0,0,-1])*98

		self.sysM = np.zeros((6*len(self.rbds), 6*len(self.rbds)))
		self.sysK = np.zeros((6*len(self.rbds), 6*len(self.rbds)))
		self.sysG = None
		self.sysf = np.zeros(6*len(self.rbds))
		self.sysv = np.zeros(6*len(self.rbds))
		self.KKT_fac = None
		self.sysSetup()

	def sysSetup(self):
		for i in range(len(self.rbds)):
			self.sysM[6*i:6*i+6, 6*i:6*i+6] = self.rbds[i].M
		
		self.KKT_fac = scipy.linalg.lu_factor(self.sysM)

	def sysForces(self):
		for i in range(len(self.rbds)):
			R = self.rbds[i].E[0:3, 0:3]
			phi = self.rbds[i].phi
			self.sysv[6*i:6*i+6] = phi

			fcor = self.rbds[i].adjoint(phi).T.dot(self.rbds[i].M.dot(phi))
			fext = np.hstack((np.zeros(3).T, R.T.dot(self.grav)*self.rbds[i].M[4,4]))
			self.sysf[6*i:6*i+6] = fcor + fext
			

	def display(self):
		viewer = igl.glfw.Viewer()

		red = igl.eigen.MatrixXd([[1,0,0]])
		purple = igl.eigen.MatrixXd([[1,0,1]])
		green = igl.eigen.MatrixXd([[0,1,0]])
		black = igl.eigen.MatrixXd([[0,0,0]])

		tempR = igl.eigen.MatrixXuc(1280, 800)
		tempG = igl.eigen.MatrixXuc(1280, 800)
		tempB = igl.eigen.MatrixXuc(1280, 800)
		tempA = igl.eigen.MatrixXuc(1280, 800)
		randc = [[random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)] for i in range(1000)]
		
		def key_down(viewer):
			viewer.data().clear()
			# if (aaa == 65):
			self.step()

			for i in range(len(self.rbds)):
				PTS = np.ones((4, 8))
				whd = self.rbds[i].whd/2.0
				PTS[0:3, 0] = np.array([-whd[0], -whd[1] , -whd[2]])
				PTS[0:3, 1] = np.array([whd[0], -whd[1] , -whd[2]])
				PTS[0:3, 2] = np.array([whd[0], whd[1] , -whd[2]])
				PTS[0:3, 3] = np.array([-whd[0], whd[1] , -whd[2]])

				PTS[0:3, 4] = np.array([-whd[0], -whd[1] , whd[2]])
				PTS[0:3, 5] = np.array([whd[0], -whd[1] , whd[2]])
				PTS[0:3, 6] = np.array([whd[0], whd[1] , whd[2]])
				PTS[0:3, 7] = np.array([-whd[0], whd[1] , whd[2]])
				PTS = self.rbds[i].E.dot(PTS)[0:3, :]
				viewer.data().add_points(igl.eigen.MatrixXd(PTS.T), green)
				
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,0]]), igl.eigen.MatrixXd([PTS[:,1]]), black)
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,0]]), igl.eigen.MatrixXd([PTS[:,3]]), black)
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,0]]), igl.eigen.MatrixXd([PTS[:,4]]), black)
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,1]]), igl.eigen.MatrixXd([PTS[:,2]]), black)

				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,2]]), igl.eigen.MatrixXd([PTS[:,3]]), black)
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,2]]), igl.eigen.MatrixXd([PTS[:,6]]), black)
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,1]]), igl.eigen.MatrixXd([PTS[:,5]]), black)
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,5]]), igl.eigen.MatrixXd([PTS[:,6]]), black)
				
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,7]]), igl.eigen.MatrixXd([PTS[:,6]]), black)
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,7]]), igl.eigen.MatrixXd([PTS[:,4]]), black)
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,7]]), igl.eigen.MatrixXd([PTS[:,3]]), black)
				viewer.data().add_edges(igl.eigen.MatrixXd([PTS[:,4]]), igl.eigen.MatrixXd([PTS[:,5]]), black)
			
		key_down(viewer)
		viewer.callback_pre_draw = key_down
		viewer.core.is_animating = False
		viewer.launch()

	def step(self):
		self.sysForces()

		Mtilde = self.sysM - self.h*self.h*self.sysK
		ftilde = self.sysM.dot(self.sysv) + self.h*self.sysf


		rhs = ftilde
		sol = scipy.linalg.lu_solve(self.KKT_fac, rhs)
		for i in range(len(self.rbds)):
			self.rbds[i].phi = sol[6*i:6*i+6]
			expE =self.rbds[i].expm(self.h*self.rbds[i].phi)
			self.rbds[i].E = self.rbds[i].E.dot(expE)
			


def test():
	body1 = SE3(whd = np.array([10,1,1]), E = np.eye(4), phi = np.array([0,5,0,0,0,0]))
	# body2 = SE3(whd = np.array([2,2,2]), E = np.eye(4), phi = np.array([0,0,0,0,0,0]))
	solv = Solver(rbds=[body1])
	solv.display()



test()


