#Display all sorts of stuff
import re
import cProfile, pstats, StringIO
import timeit

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

from Helpers import *
from Mesh import Mesh
from Neo import NeohookeanElastic
from Arap import ARAP

class Display:
	def __init__(self, imesh, iarap, ineo, isolve):
		self.last_mouse = None
		self.mode = 0
		self.mesh = imesh
		self.arap = iarap
		self.neo = ineo
		self.time_integrator = isolve

	def display(self):
		self.mesh.red_r[0] = 0.1

		viewer = igl.glfw.Viewer()

		tempR = igl.eigen.MatrixXuc(1280, 800)
		tempG = igl.eigen.MatrixXuc(1280, 800)
		tempB = igl.eigen.MatrixXuc(1280, 800)
		tempA = igl.eigen.MatrixXuc(1280, 800)

		def mouse_up(viewer, btn, bbb):
			if btn==1:
				coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
				print("up", coord)
				up = e2p(coord)
				print("vec", up - self.last_mouse)
			for i in range(len(self.time_integrator.mov)):
				# self.mesh.g[2*self.time_integrator.mov[i]]   -= self.time_integrator.adder
				self.mesh.g[2*self.time_integrator.mov[i]+1] -= self.time_integrator.adder
			return False

		def mouse_down(viewer, btn, bbb):
			if btn==1:
				coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
				self.last_mouse = e2p(coord)

			return False

		def key_down(viewer,aaa, bbb):
			viewer.data().clear()
		
			if(aaa==65):
				self.time_integrator.move_g()
				# self.arap.iterate()
				self.time_integrator.static_solve()
				self.time_integrator.time +=1

				
			DV, DT = self.mesh.getDiscontinuousVT()
			RV, RT = self.mesh.getContinuousVT()
			V2 = igl.eigen.MatrixXd(RV)
			T2 = igl.eigen.MatrixXi(RT)
			viewer.data().set_mesh(V2, T2)

			red = igl.eigen.MatrixXd([[1,0,0]])
			purple = igl.eigen.MatrixXd([[1,0,1]])
			green = igl.eigen.MatrixXd([[0,1,0]])
			black = igl.eigen.MatrixXd([[0,0,0]])


			for e in DT:
				P = DV[e]
				DP = np.array([P[1], P[2], P[0]])
				viewer.data().add_edges(igl.eigen.MatrixXd(P), igl.eigen.MatrixXd(DP), purple)


			FIXED = []
			disp_g = self.mesh.getg()
			for i in range(len(self.mesh.fixed)):
				FIXED.append(disp_g[2*self.mesh.fixed[i]:2*self.mesh.fixed[i]+2])

			viewer.data().add_points(igl.eigen.MatrixXd(np.array(FIXED)), red)


			CAg = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.getg()))
			#centroids and rotation clusters
			for i in range(len(self.mesh.T)):
				S = self.mesh.getS(i)
				C = np.matrix([CAg[6*i:6*i+2],CAg[6*i:6*i+2]])
				U = 0.01*self.mesh.getU(i).transpose()+C
				if(np.linalg.norm(self.mesh.sW[2*i,:])>=1):
					viewer.data().add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)
					viewer.data().add_edges(igl.eigen.MatrixXd(C[1,:]), igl.eigen.MatrixXd(U[1,:]), green)
				else:
					viewer.data().add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)
					viewer.data().add_edges(igl.eigen.MatrixXd(C[1,:]), igl.eigen.MatrixXd(U[1,:]), red)
				viewer.data().add_points(igl.eigen.MatrixXd(np.array([CAg[6*i:6*i+2]])), igl.eigen.MatrixXd([[0, .2*self.mesh.r_element_cluster_map[i],1-0.2*self.mesh.r_element_cluster_map[i]]]))

			#snapshot
			if(aaa==65 and not self.mesh.reduced_g):
				displacements = disp_g - self.mesh.x0
				igl.writeDMAT("snapshots/"+str(self.time_integrator.time)+".dmat", igl.eigen.MatrixXd(displacements), False)

			#Write image
			if(aaa==65):
				viewer.core.draw_buffer(viewer.data(), False, tempR, tempG, tempB, tempA)
				igl.png.writePNG(tempR, tempG, tempB, tempA, "frames/"+str(self.time_integrator.time)+".png")



			return True

		# for clicks in range(40):
		key_down(viewer, 'b', 123)
		viewer.callback_key_down = key_down
		viewer.callback_mouse_down = mouse_down
		viewer.callback_mouse_up = mouse_up
		viewer.core.is_animating = False
		viewer.launch()
