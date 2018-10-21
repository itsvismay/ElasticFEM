#running the code version 3
import sys, os
import scipy
from scipy import sparse
from scipy.sparse import linalg
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
from iglhelpers import *
from scipy.spatial import Delaunay
import random
from Mesh import Mesh
import Arap
import Solvers
import Neo

FOLDER = "./MeshSetups/"+"TestArm/"
print("reading from: "+FOLDER)

eV = igl.eigen.MatrixXd()
eT = igl.eigen.MatrixXi()
eu = igl.eigen.MatrixXd()
es_ind = igl.eigen.MatrixXi()
er_ind = igl.eigen.MatrixXi()
esW = igl.eigen.MatrixXd()
emat = igl.eigen.MatrixXd()

igl.readOBJ(FOLDER+"muscle_bone/"+"combined.obj", eV, eT)
igl.readDMAT(FOLDER+"muscle_bone/"+"u.dmat", eu)
igl.readDMAT(FOLDER+"muscle_bone/"+"shandles.dmat", es_ind)
igl.readDMAT(FOLDER+"muscle_bone/"+"e_to_c.dmat", er_ind)
igl.readDMAT(FOLDER+"muscle_bone/"+"sW.dmat", esW)
igl.readDMAT(FOLDER+"muscle_bone/"+"elem_material.dmat", emat)



V = e2p(eV)[:,:2]
T = e2p(eT)
u = e2p(eu)[:, 0]
s_ind = e2p(es_ind)[:, 0]
r_ind = e2p(er_ind)[:, 0]
sW = e2p(esW)
elem_material = e2p(emat)[:, 0]

mesh = Mesh(read_in = True)
mesh.init_muscle_bone(V, T, u, s_ind, r_ind, sW, elem_material,[0,1,10], [])

#ARAP setup
arap = Arap.ARAP(imesh=mesh, filen="snapshots/")

#Elasticity setup
neo = Neo.NeohookeanElastic(imesh = mesh)

#Solver setup
ti = Solvers.TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neo)


def display_mesh():
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
		bc = igl.eigen.MatrixXd()
		fid = igl.eigen.MatrixXi(np.array([-1]))
		coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
		hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
		viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(mesh.V), igl.eigen.MatrixXi(mesh.T), fid, bc)
		ind = e2p(fid)[0][0]
		
		if hit and btn==0:
			# paint hit red
			# meshes[im]["Fix"].append(meshes[im]["T"][ind][np.argmax(bc)])
			print("fix", ind, np.argmax(bc))
			return True

		if hit and btn==2:
			# paint hit red
			# meshes[im]["Mov"].append(meshes[im]["T"][ind][np.argmax(bc)])
			print("mov", ind, np.argmax(bc))
			return True

		return False

	def key_down(viewer,aaa, bbb):
		viewer.data().clear()
		if(aaa==65):
			# ti.move_g()
			ti.arap.iterate()
			# ti.static_solve()
			ti.time +=1

		RV, RT = ti.mesh.getContinuousVT()
		V2 = igl.eigen.MatrixXd(RV)
		T2 = igl.eigen.MatrixXi(RT)
		viewer.data().set_mesh(V2, T2)
		

		MOV = []
		disp_g = ti.mesh.getg()
		for i in range(len(ti.mesh.mov)):
			MOV.append(disp_g[2*ti.mesh.mov[i]:2*ti.mesh.mov[i]+2])
		viewer.data().add_points(igl.eigen.MatrixXd(np.array(MOV)), green)


		FIXED = []
		disp_g = ti.mesh.getg()
		for i in range(len(ti.mesh.fixed)):
			FIXED.append(disp_g[2*ti.mesh.fixed[i]:2*ti.mesh.fixed[i]+2])
		viewer.data().add_points(igl.eigen.MatrixXd(np.array(FIXED)), red)
		
		CAx0 = ti.mesh.getC().dot(ti.mesh.getA().dot(ti.mesh.x0))
		for i in range(len(ti.mesh.T)):
			c = np.matrix([CAx0[6*i:6*i+2],CAx0[6*i:6*i+2]])
			alpha = ti.mesh.u[i]
			cU, sU = np.cos(alpha), np.sin(alpha)
			U = np.array(((cU,-sU), (sU, cU)))
			scaledU = np.multiply(U, np.array([[.1],[.1]])) + c
			viewer.data().add_edges(igl.eigen.MatrixXd(c[0,:]), igl.eigen.MatrixXd(scaledU[0,:]), black)
		Colors = np.ones(ti.mesh.T.shape)

		for i in range(len(ti.mesh.T)): 
			color = black
			Colors[i,:] = randc[ti.mesh.r_element_cluster_map[i]]
		
		Colors[np.array([ti.mesh.s_handles_ind]),:] = np.array([0,0,0])
		viewer.data().set_colors(igl.eigen.MatrixXd(np.array(Colors)))

	
	key_down(viewer, "b", 123)
	viewer.callback_mouse_down = mouse_down
	viewer.callback_key_down = key_down
	viewer.core.is_animating = False
	viewer.launch()


display_mesh()














