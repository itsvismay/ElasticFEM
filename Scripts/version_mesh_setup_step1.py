#running the code version 3
import Meshwork
import Arap
import Display
import Solvers
import Neo
import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
from iglhelpers import *
from scipy.spatial import Delaunay
import random

VTU1 = Meshwork.rectangle_mesh(x=5, y=1, step=1.0, offset=(0,0))
VTU2 = Meshwork.rectangle_mesh(x=5, y=1, step=1.0, offset=(5,0))
VTU3 = Meshwork.torus_mesh(r1=3, r2=3, r3=5, step=1.0, offset=(5,1))

def display_mesh(meshes):
	red = igl.eigen.MatrixXd([[1,0,0]])
	purple = igl.eigen.MatrixXd([[1,0,1]])
	green = igl.eigen.MatrixXd([[0,1,0]])
	black = igl.eigen.MatrixXd([[0,0,0]])
	blue = igl.eigen.MatrixXd([[0,0,1]])
	white = igl.eigen.MatrixXd([[1,1,1]])

	randc = [[random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)] for i in range(10)]

	viewer = igl.glfw.Viewer()

	V = None
	T = None

	def mouse_down(viewer, btn, bbb):
		# Cast a ray in the view direction starting from the mouse position
		for im in range(len(meshes)):
			bc = igl.eigen.MatrixXd()
			fid = igl.eigen.MatrixXi(np.array([-1]))
			coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
			hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
			viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(meshes[im][0]), igl.eigen.MatrixXi(meshes[im][1]), fid, bc)
			ind = e2p(fid)[0][0]

			if hit and btn==0:
				# paint hit red
				# print(im, ind)
				meshes[im][1] = np.delete(meshes[im][1], (ind), axis=0)
				return True
			
			if hit and btn==2:
				# paint hit red
				print("mov", im, ind)
				# self.Mov.append(self.T[ind][np.argmax(bc)])
				# print("mov",self.T[ind][np.argmax(bc)])
				return True

		return False

	def key_down(viewer,aaa, bbb):
		if aaa==65:
			#save current config
			V = meshes[0][0]
			T = meshes[0][1]
			for im in range(1,len(meshes)):
				T = np.vstack((T, np.add(meshes[im][1], len(V))))
				V = np.vstack((V, meshes[im][0]))

			V2 = igl.eigen.MatrixXd(V)
			T2 = igl.eigen.MatrixXi(T)
			viewer.data().set_mesh(V2, T2)

		if aaa==83:
			#output mesh
			V = meshes[0][0]
			T = meshes[0][1]
			for im in range(1,len(meshes)):
				T = np.vstack((T, np.add(meshes[im][1], len(V))))
				V = np.vstack((V, meshes[im][0]))
			V = np.hstack((V, np.zeros((len(V),1))))
			V2 = igl.eigen.MatrixXd(V)
			T2 = igl.eigen.MatrixXi(T)
			igl.writeOBJ("./MeshSetups/step1.obj",V2,T2)



	def pre_draw(viewer):
		viewer.data().clear()
		for im in range(len(meshes)):
			for e in meshes[im][1]:
				P = meshes[im][0][e]
				DP = np.array([P[1], P[2], P[0]])
				viewer.data().add_edges(igl.eigen.MatrixXd(P), igl.eigen.MatrixXd(DP), purple)

	key_down(viewer, "b", 123)
	viewer.callback_mouse_down = mouse_down
	# viewer.callback_mouse_up = mouse_up
	# viewer.callback_mouse_move = mouse_move
	viewer.callback_key_down = key_down
	viewer.callback_pre_draw = pre_draw
	viewer.core.is_animating = False
	viewer.launch()

# # Mesh creation
# # Read-in mesh: read V, T, U, rot-clusters, skinning handles, Modes (if reduced)
# VTU1 = Meshwork.rectangle_mesh(x=5, y=1, step=1.0, offset=(0,0))
# pp1 = Meshwork.Preprocessing(_VT = VTU1)
# pp1.Fix = [11]	
# pp1.Mov = Meshwork.get_min(pp1.V, a=0, eps=1e-2)
# mesh1 = pp1.getMesh(muscle=False)
# #ARAP setup
# arap1 = Arap.ARAP(imesh=mesh1, filen="snapshots/")
# #Elasticity setup
# neo1 = Neo.NeohookeanElastic(imesh = mesh1)


# VTU2 = Meshwork.rectangle_mesh(x=5, y=1, step=1.0, offset=(5,0))
# pp2 = Meshwork.Preprocessing(_VT = VTU2)
# pp2.Fix = [1]	
# pp2.Mov = Meshwork.get_max(pp2.V, a=0, eps=1e-2)
# mesh2 = pp2.getMesh(muscle=False)
# #ARAP setup
# arap2 = Arap.ARAP(imesh=mesh2, filen="snapshots/")
# #Elasticity setup
# neo2 = Neo.NeohookeanElastic(imesh = mesh2)

# VTU3 = Meshwork.torus_mesh(r1=2, r2=4, r3=5, step=1.0, offset=(5,1))
# pp3 = Meshwork.Preprocessing(_VT = VTU3)
# pp3.Fix = Meshwork.get_max(pp3.V, a=1, eps=0.2)	
# pp3.Mov = Meshwork.get_max(pp3.V, a=0, eps=1e-2) + Meshwork.get_min(pp3.V, a=0, eps=1e-2) 
# mesh3 = pp3.getMesh(muscle=True)
# #ARAP setup
# arap3 = Arap.ARAP(imesh=mesh3, filen="snapshots/")
# #Elasticity setup
# neo3 = Neo.NeohookeanElastic(imesh = mesh3)

# araps = [arap3, arap1, arap2]
# neos = [neo3, neo1, neo2]
meshes = [VTU1, VTU2, VTU3]
display_mesh(meshes)




# #Solver setup
# ti = Solvers.TimeIntegrator(imesh = meshes, iarap = araps, ielastic = neos)

# #Running
# disp = Display.Display(isolve = ti)

# # disp.headless()
# disp.display_arap()
