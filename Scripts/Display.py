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

	def display_statics(self):
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
			bc = igl.eigen.MatrixXd()
			fid = igl.eigen.MatrixXi(np.array([-1]))
			coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
			hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
			viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(self.time_integrator.mesh.V), igl.eigen.MatrixXi(self.time_integrator.mesh.T), fid, bc)
			ind = e2p(fid)[0][0]
			print(ind)
			if hit and btn==0:
				# paint hit red
				print("fix",self.time_integrator.mesh.T[ind][np.argmax(bc)])
				return True
			
			if hit and btn==2:
				# paint hit red
				print("mov",self.time_integrator.mesh.T[ind][np.argmax(bc)])
				return True
			return False

		def key_down(viewer,aaa, bbb):
			viewer.data().clear()
		
			if(aaa==65):
				# self.time_integrator.move_g()
				# print(self.time_integrator.arap.Energy())
				# self.time_integrator.arap.iterate()
				# print(self.time_integrator.arap.Energy())
				# print(self.time_integrator.mesh.red_r)
				# print(self.time_integrator.mesh.z)
				self.time_integrator.dynamics()
				self.time_integrator.time +=1

			if(aaa>=49 and aaa<=57):
				self.time_integrator.toggle_muscle_group(aaa-49)

			print("DRAWING--------")
			DV, DT = self.time_integrator.mesh.getDiscontinuousVT()
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
				U = 0.1*self.time_integrator.mesh.getU(i)+C
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
			elif aaa==66:
				for i in range(len(self.time_integrator.mesh.T)):
					if self.time_integrator.mesh.u_toggle[i]<0.5:
						Colors[i,:] = black


			Colors[np.array([self.time_integrator.mesh.s_handles_ind]),:] = np.array([1,0.5,1])
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

	def display_dynamics(self):
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

		def key_down(viewer,aaa, bbb):
			viewer.data().clear()

			if(aaa==65):
				# self.time_integrator.move_g()
				# self.time_integrator.arap.iterate()
				self.time_integrator.dynamics()
				self.time_integrator.time +=1

			# if(aaa>=49 and aaa<=57):
			# 	self.time_integrator.toggle_muscle_group(aaa-49)

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
				U = 0.3*self.time_integrator.mesh.getU(i)+C
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
	
			#Write image
			if(aaa==65 or aaa==66):
				viewer.core.draw_buffer(viewer.data(), False, tempR, tempG, tempB, tempA)
				igl.png.writePNG(tempR, tempG, tempB, tempA, "frames/"+str(self.time_integrator.time)+".png")

			return True

		# for clicks in range(40):
		# key_down(viewer, 'b', 123)
		viewer.callback_key_down = key_down
		viewer.core.is_animating = False
		viewer.launch()

	def display_arap(self):
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
				for im in range(len(self.time_integrator.meshes)):
					self.time_integrator.move_g(im)
					self.time_integrator.araps[im].iterate()

			for im in range(len(self.time_integrator.meshes)):
				print("DRAWING--------")
				DV, DT = self.time_integrator.meshes[im].getDiscontinuousVT()
				RV, RT = self.time_integrator.meshes[im].getContinuousVT()
				V2 = igl.eigen.MatrixXd(RV)
				T2 = igl.eigen.MatrixXi(RT)
				# viewer.data().set_mesh(V2, T2)

				for e in DT:
					P = DV[e]
					DP = np.array([P[1], P[2], P[0]])
					viewer.data().add_edges(igl.eigen.MatrixXd(P), igl.eigen.MatrixXd(DP), purple)


				MOV = []
				disp_g = self.time_integrator.meshes[im].getg()
				for i in range(len(self.time_integrator.meshes[im].mov)):
					MOV.append(disp_g[2*self.time_integrator.meshes[im].mov[i]:2*self.time_integrator.meshes[im].mov[i]+2])
				viewer.data().add_points(igl.eigen.MatrixXd(np.array(MOV)), green)


				FIXED = []
				disp_g = self.time_integrator.meshes[im].getg()
				for i in range(len(self.time_integrator.meshes[im].fixed)):
					FIXED.append(disp_g[2*self.time_integrator.meshes[im].fixed[i]:2*self.time_integrator.meshes[im].fixed[i]+2])
				viewer.data().add_points(igl.eigen.MatrixXd(np.array(FIXED)), red)


				#Muscle fiber directions
				CAg = self.time_integrator.meshes[im].getC().dot(self.time_integrator.meshes[im].getA().dot(self.time_integrator.meshes[im].getg()))
				for i in range(len(self.time_integrator.meshes[im].T)):
					S = self.time_integrator.meshes[im].getS(i)
					C = np.matrix([CAg[6*i:6*i+2],CAg[6*i:6*i+2]])
					U = 0.3*self.time_integrator.meshes[im].getU(i)+C
					viewer.data().add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)

				Colors = np.ones(self.time_integrator.meshes[im].T.shape)
				if aaa==67:
					for i in range(len(self.time_integrator.meshes[im].u_clusters_element_map)):
						for j in range(len(self.time_integrator.meshes[im].u_clusters_element_map[i])):
							k = self.time_integrator.meshes[im].u_clusters_element_map[i][j]
							Colors[k,:] = randc[i]
				elif aaa==82:
					for i in range(len(self.time_integrator.meshes[im].T)): 
						color = black
						Colors[i,:] = randc[self.time_integrator.meshes[im].r_element_cluster_map[i]]
				elif aaa>=49 and aaa<=57:
					for j in range(len(self.time_integrator.meshes[im].u_clusters_element_map[aaa-49])):
						k = self.time_integrator.meshes[im].u_clusters_element_map[aaa-49][j]
						Colors[k,:] = randc[aaa-49]
				Colors[np.array([self.time_integrator.meshes[im].s_handles_ind]),:] = np.array([0,0,0])
				# viewer.data().set_colors(igl.eigen.MatrixXd(np.array(Colors)))
				print("Done drawing--------")


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