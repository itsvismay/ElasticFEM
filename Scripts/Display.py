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
	def __init__(self, isolve):
		self.last_mouse = None
		self.mode = 0
		self.time_integrator = isolve

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

		def mouse_up(viewer, btn, bbb):
			return False

		def mouse_down(viewer, btn, bbb):
			return False

		def key_down(viewer,aaa, bbb):
			viewer.data().clear()
		
			if(aaa==65):
				# self.time_integrator.move_g()
				# self.time_integrator.arap.iterate()
				self.time_integrator.static_solve()
				self.time_integrator.time +=1

			if(aaa>=49 and aaa<=57):
				self.time_integrator.toggle_muscle_group(aaa-49)

			print("DRAWING--------")
			# DV, DT = self.time_integrator.mesh.getDiscontinuousVT()
			RV, RT = self.time_integrator.mesh.getContinuousVT()
			V2 = igl.eigen.MatrixXd(RV)
			T2 = igl.eigen.MatrixXi(RT)
			viewer.data().set_mesh(V2, T2)

			# for e in DT:
			# 	P = DV[e]
			# 	DP = np.array([P[1], P[2], P[0]])
			# 	viewer.data().add_edges(igl.eigen.MatrixXd(P), igl.eigen.MatrixXd(DP), purple)


			MOV = []
			disp_g = self.time_integrator.mesh.getg()
			for i in range(len(self.time_integrator.mesh.mov)):
				MOV.append(disp_g[2*self.time_integrator.mesh.mov[i]:2*self.time_integrator.mesh.mov[i]+2])
			viewer.data().add_points(igl.eigen.MatrixXd(np.array(MOV)), green)


			FIXED = []
			disp_g = self.time_integrator.mesh.getg()
			for i in range(len(self.time_integrator.mesh.fixed)):
				FIXED.append(disp_g[2*self.time_integrator.mesh.fixed[i]:2*self.time_integrator.mesh.fixed[i]+2])
			viewer.data().add_points(igl.eigen.MatrixXd(np.array(FIXED)), red)


			#Muscle fiber directions
			CAg = self.time_integrator.mesh.getC().dot(self.time_integrator.mesh.getA().dot(self.time_integrator.mesh.getg()))
			for i in range(len(self.time_integrator.mesh.T)):
				S = self.time_integrator.mesh.getS(i)
				C = np.matrix([CAg[6*i:6*i+2],CAg[6*i:6*i+2]])
				U = 0.003*self.time_integrator.mesh.getU(i)+C
				viewer.data().add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)

			Colors = np.ones(self.time_integrator.mesh.T.shape)
			if aaa==67:
				for i in range(len(self.time_integrator.mesh.u_clusters_element_map)):
					for j in range(len(self.time_integrator.mesh.u_clusters_element_map[i])):
						k = self.time_integrator.mesh.u_clusters_element_map[i][j]
						Colors[k,:] = randc[i]
			elif aaa==82:
				for i in range(len(self.time_integrator.mesh.T)): 
					color = black
					Colors[i,:] = randc[self.time_integrator.mesh.r_element_cluster_map[i]]
			elif aaa>=49 and aaa<=57:
				for j in range(len(self.time_integrator.mesh.u_clusters_element_map[aaa-49])):
					k = self.time_integrator.mesh.u_clusters_element_map[aaa-49][j]
					Colors[k,:] = randc[aaa-49]
			Colors[np.array([self.time_integrator.mesh.s_handles_ind]),:] = np.array([0,0,0])
			viewer.data().set_colors(igl.eigen.MatrixXd(np.array(Colors)))
			print("Done drawing--------")
			#snapshot
			if(aaa==65):
				displacements = disp_g - self.time_integrator.mesh.x0
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

	def headless(self):

		pr = cProfile.Profile()
		pr.enable()

		# self.time_integrator.mesh.getGlobalF(updateR=True, updateS=False, updateU=False)
		self.time_integrator.static_solve()
		
		pr.disable()
		s = StringIO.StringIO()
		sortby = 'cumulative'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats(1)
		print(s.getvalue())

		return