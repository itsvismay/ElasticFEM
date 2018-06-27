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
	def __init__(self):
		self.last_mouse = None
		self.mode = 0

	def display(self):
		# VTU, to_fix = feather_muscle2_test_setup(p1 = 200, p2 = 100)
		VTU = rectangle_mesh(70, 50,angle=np.pi/2, step=.1)
		print(len(VTU[0]), len(VTU[1]))
		to_fix = get_max(VTU[0],a=1, eps=1e-2)
		to_mov = []# get_min(VTU[0], a=1, eps=1e-2)
		mesh = Mesh(VTU,ito_fix=to_fix, ito_mov=to_mov, red_g = True)

		neoh =NeohookeanElastic(imesh=mesh )
		arap = ARAP(imesh=mesh, filen="snapshots/")
		time_integrator = TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neoh)
		mesh.red_r[0] = 0.1

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
			for i in range(len(time_integrator.mov)):
				# mesh.g[2*time_integrator.mov[i]]   -= time_integrator.adder
				mesh.g[2*time_integrator.mov[i]+1] -= time_integrator.adder
			return False

		def mouse_down(viewer, btn, bbb):
			if btn==1:
				coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
				self.last_mouse = e2p(coord)

			return False

		def key_down(viewer,aaa, bbb):
			viewer.data().clear()
		
			if(aaa==65):
				time_integrator.move_g()
				# arap.iterate()
				time_integrator.static_solve()
				time_integrator.time +=1

				
			DV, DT = mesh.getDiscontinuousVT()
			RV, RT = mesh.getContinuousVT()
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
			disp_g = mesh.getg()
			for i in range(len(mesh.fixed)):
				FIXED.append(disp_g[2*mesh.fixed[i]:2*mesh.fixed[i]+2])

			viewer.data().add_points(igl.eigen.MatrixXd(np.array(FIXED)), red)


			CAg = mesh.getC().dot(mesh.getA().dot(mesh.getg()))
			#centroids and rotation clusters
			for i in range(len(mesh.T)):
				S = mesh.getS(i)
				C = np.matrix([CAg[6*i:6*i+2],CAg[6*i:6*i+2]])
				U = 0.01*mesh.getU(i).transpose()+C
				if(np.linalg.norm(mesh.sW[2*i,:])>=1):
					viewer.data().add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)
					viewer.data().add_edges(igl.eigen.MatrixXd(C[1,:]), igl.eigen.MatrixXd(U[1,:]), green)
				else:
					viewer.data().add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)
					viewer.data().add_edges(igl.eigen.MatrixXd(C[1,:]), igl.eigen.MatrixXd(U[1,:]), red)
				viewer.data().add_points(igl.eigen.MatrixXd(np.array([CAg[6*i:6*i+2]])), igl.eigen.MatrixXd([[0, .2*mesh.r_element_cluster_map[i],1-0.2*mesh.r_element_cluster_map[i]]]))

			#snapshot
			if(aaa==65 and not mesh.reduced_g):
				displacements = disp_g - mesh.x0
				igl.writeDMAT("snapshots/"+str(time_integrator.time)+".dmat", igl.eigen.MatrixXd(displacements), False)

			#Write image
			if(aaa==65):
				viewer.core.draw_buffer(viewer.data(), False, tempR, tempG, tempB, tempA)
				igl.png.writePNG(tempR, tempG, tempB, tempA, "frames/"+str(time_integrator.time)+".png")



			return True

		# for clicks in range(40):
		key_down(viewer, 'b', 123)
		viewer.callback_key_down = key_down
		viewer.callback_mouse_down = mouse_down
		viewer.callback_mouse_up = mouse_up
		viewer.core.is_animating = False
		viewer.launch()
	
	def WiggleModes(self):
		# VTU, to_fix = feather_muscle2_test_setup(p1 = 100, p2 = 50)
		# VTU, to_fix = feather_muscle2_test_setup(p1 = 200, p2 = 100)
		VTU = rectangle_mesh(5, 5,angle=0, step=.1)

		to_fix = get_min_max(VTU[0],a=1, eps=1e-2)
		to_mov = get_min(VTU[0], a=1, eps=1e-2)
		mesh = Mesh(VTU,ito_fix=to_fix, ito_mov=to_mov, red_g = True)

		neoh =NeohookeanElastic(imesh=mesh )
		arap = ARAP(imesh=mesh, filen="snapshots/")
		time_integrator = TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neoh)

		viewer = igl.glfw.Viewer()

		tempR = igl.eigen.MatrixXuc(1280, 800)
		tempG = igl.eigen.MatrixXuc(1280, 800)
		tempB = igl.eigen.MatrixXuc(1280, 800)
		tempA = igl.eigen.MatrixXuc(1280, 800)

		def key_down(viewer, aaa, bbb):
			#1,2 number keys are toggles
			if aaa==49:
				self.mode -=1
			if aaa==50:
				self.mode +=1
			return True

		def post_draw(viewer):
			viewer.data().clear()
			rmode_i = np.zeros((len(mesh.V),1))
			wiggleg = mesh.getg()
			if(viewer.core.is_animating):
				mode_i = mesh.G[:,self.mode]*.1*np.sin(math.sqrt(mesh.Eigvals[self.mode])*time_integrator.time)
				for e in range(len(mesh.g)):
					wiggleg[e] = mesh.x0[e] + mode_i[e,0]
				time_integrator.time +=1
				if(time_integrator.time%10==0):
					time_integrator.add_on += 1
				print(time_integrator.time, mesh.Eigvals[self.mode])


			RV, RT = mesh.getContinuousVT(g = wiggleg)
			RV = np.concatenate((RV, rmode_i), axis =1)
			V2 = igl.eigen.MatrixXd(RV)
			T2 = igl.eigen.MatrixXi(RT)
			viewer.data().set_mesh(V2, T2)

			red = igl.eigen.MatrixXd([[1,0,0]])
			purple = igl.eigen.MatrixXd([[1,0,1]])
			green = igl.eigen.MatrixXd([[0,1,0]])
			black = igl.eigen.MatrixXd([[0,0,0]])


			FIXED = []
			disp_g = mesh.getg()
			for i in range(len(mesh.fixed)):
				FIXED.append(disp_g[2*mesh.fixed[i]:2*mesh.fixed[i]+2])
			viewer.data().add_points(igl.eigen.MatrixXd(np.array(FIXED)), red)

			

			#Write image
			if(viewer.core.is_animating):
				viewer.core.draw_buffer(viewer.data(), False, tempR, tempG, tempB, tempA)
				igl.png.writePNG(tempR, tempG, tempB, tempA, "frames/"+str(time_integrator.time)+".png")

			return True

		# for clicks in range(40):
		# key_down(viewer, 'b', 123)
		viewer.callback_post_draw = post_draw
		viewer.callback_key_down = key_down
		viewer.core.is_animating = False
		viewer.launch()

	def headless(self):
		times = []
		for i in range (1, 9):
			VTU = rectangle_mesh(20*i, 10*i, angle=np.pi/4, step=.1)
			tr, tl, br, bl = get_corners(VTU[0], top=True, eps =1e-2)
			to_fix = [tr, tl, br, bl]
			to_mov = [br, bl]
			mesh = Mesh(VTU,ito_fix=to_fix, ito_mov=to_mov, red_g = True)

			neoh =NeohookeanElastic(imesh=mesh )
			arap = ARAP(imesh=mesh, filen="snapshots/600x300/")
			time_integrator = TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neoh)
			
			mesh.getGlobalF(False, True, False)

			pr = cProfile.Profile()
			pr.enable()

			# time_integrator.move_g()
			arap.Hessians()

			pr.disable()
			s = StringIO.StringIO()
			sortby = 'cumulative'
			ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
			ps.print_stats(1)
			print(s.getvalue())

			timer = timeit.Timer(arap.Hessians)
			timefor1 = timer.timeit(1)
			print(len(VTU[0]), len(VTU[1]), timefor1)
			times.append((len(VTU[0]), len(VTU[1]), timefor1))
		print("###################")
		print("TIMES")
		print(times)

