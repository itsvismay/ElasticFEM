#running the code version 3
import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
from iglhelpers import *
from scipy.spatial import Delaunay
import Helpers
import random

mesh1 = Helpers.rectangle_mesh(x=5, y=1, step=1.0, offset=(0,0))
mesh2 = Helpers.rectangle_mesh(x=5, y=1, step=1.0, offset=(5,0))
mesh3 = Helpers.torus_mesh(r1=3, r2=3, r3=5, step=1.0, offset=(5,1))


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
			viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(meshes[im]["V"]), igl.eigen.MatrixXi(meshes[im]["T"]), fid, bc)
			ind = e2p(fid)[0][0]

			if hit and btn==0:
				print(im, ind)
				meshes[im]["T"] = np.delete(meshes[im]["T"], (ind), axis=0)
				return True

		return False

	def key_down(viewer,aaa, bbb):
		# viewer.data().clear()
		if aaa==65:
			#save 3 meshes separately
			for im in range(0):
				pass


	def pre_draw(viewer):
		viewer.data().clear()
		for im in range(len(meshes)):
			for e in meshes[im]["T"]:
				P = meshes[im]["V"][e]
				DP = np.array([P[1], P[2], P[0]])
				viewer.data().add_edges(igl.eigen.MatrixXd(P), igl.eigen.MatrixXd(DP), purple)

		
	key_down(viewer, "b", 123)
	viewer.callback_mouse_down = mouse_down
	viewer.callback_key_down = key_down
	viewer.callback_pre_draw = pre_draw
	viewer.core.is_animating = False
	viewer.launch()


meshes = [mesh1, mesh2, mesh3]
display_mesh(meshes)





