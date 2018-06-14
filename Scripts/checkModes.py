import numpy as np
from version2 import triangle_mesh, rectangle_mesh, torus_mesh, featherize,get_min, get_max, get_min_max, NeohookeanElastic, ARAP, Mesh
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.3f}'.format(x)})

# VTU, to_fix = feather_muscle2_test_setup(p1 = 200, p2 = 100)
VTU = rectangle_mesh(10, 30,angle=0, step=.1)
# # VTU = torus_mesh(5, 4, 3, .1)
to_fix = get_min_max(VTU[0],a=1, eps=1e-1)
to_mov = get_min(VTU[0], a=1, eps=1e-1)
mesh = Mesh(VTU,ito_fix=to_fix, ito_mov=to_mov, red_g = True)

neoh =NeohookeanElastic(imesh=mesh )
arap = ARAP(imesh=mesh, filen="snapshots/")


def display():
	viewer = igl.glfw.Viewer()

	tempR = igl.eigen.MatrixXuc(1280, 800)
	tempG = igl.eigen.MatrixXuc(1280, 800)
	tempB = igl.eigen.MatrixXuc(1280, 800)
	tempA = igl.eigen.MatrixXuc(1280, 800)
	def key_down(viewer,aaa, bbb):
		if(aaa==65):
			for i in range(len(mesh.mov)):
				# mesh.g[2*time_integrator.mov[i]]   -= time_integrator.adder
				mesh.g[2*time_integrator.mov[i]+1] -= time_integrator.adder
			arap.iterate()
		
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



	
	key_down(viewer, 'b', 123)
	viewer.callback_key_down = key_down
	viewer.core.is_animating = False
	viewer.launch()

display()