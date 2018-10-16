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


V2 = igl.eigen.MatrixXd()
T2 = igl.eigen.MatrixXi()
igl.readOBJ("./MeshSetups/step1.obj", V2, T2)
V = e2p(V2)
T = e2p(T2)
muscles = np.ones(len(T))

def getA(iV, iT):
	A = sparse.lil_matrix((6*len(iT), 2*len(iV)))
	for i in range(len(iT)):
		e = iT[i]
		for j in range(len(e)):
			v = e[j]
			A[6*i+2*j, 2*v] = 1
			A[6*i+2*j+1, 2*v+1] = 1

	A = A.tocsc()
	A.eliminate_zeros()
	return A

def getC(iT):
	C = sparse.kron(sparse.eye(len(T)), sparse.kron(np.ones((3,3))/3 , np.eye(2)))
	return C

def get_area(p1, p2, p3):
	# print(p1- p2)
	# print(p1 - p3)
	# print(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))*0.5

def getMassDiags(iV, iT):
	mass_diag = np.zeros(2*len(iV))
	density = 1000
	for i in range(len(iT)):
		e = iT[i]
		undef_area = density*get_area(iV[e[0]], iV[e[1]], iV[e[2]])
		mass_diag[2*e[0]+0] += undef_area/3.0
		mass_diag[2*e[0]+1] += undef_area/3.0

		mass_diag[2*e[1]+0] += undef_area/3.0
		mass_diag[2*e[1]+1] += undef_area/3.0

		mass_diag[2*e[2]+0] += undef_area/3.0
		mass_diag[2*e[2]+1] += undef_area/3.0


		# print("Creating Mass Matrix")
		# mass_diag = np.zeros(6*len(iT))
		# density = 1000.0
		# for i in range(len(iT)):
		# 	e = iT[i]
		# 	undef_area = density*get_area(iV[e[0]], iV[e[1]], iV[e[2]])
		# 	mass_diag[6*i+0] += undef_area/3.0
		# 	mass_diag[6*i+1] += undef_area/3.0

		# 	mass_diag[6*i+2] += undef_area/3.0
		# 	mass_diag[6*i+3] += undef_area/3.0

		# 	mass_diag[6*i+4] += undef_area/3.0
		# 	mass_diag[6*i+5] += undef_area/3.0
		
		# print("Done with Mass matrix")
	return mass_diag

def heat_method(iV, iT, iFix, iMov):
	t = 1e-1
	eLc = igl.eigen.SparseMatrixd()
	igl.cotmatrix(igl.eigen.MatrixXd(iV), igl.eigen.MatrixXi(iT), eLc)
	Lc = e2p(eLc)
	Mdiag = getMassDiags(iV, iT)[2*np.arange(Lc.shape[0])]
	Mc = sparse.diags(Mdiag)


	#Au = b st. Cu = Cu0
	u0 = np.zeros(len(iV))
	fixed = list(set(iFix) - set(iMov))
	u0[fixed] = 2
	u0[iMov] = -2

	Id = sparse.eye(len(iV)).tocsc()
	fixedverts = [i for i in range(len(u0)) if u0[i]!=0]
	C = Id[:,fixedverts]

	# print(Lc.shape)
	# print(mesh.iV.shape)
	A = (Mc - t*Lc)
	col1 = sparse.vstack((A, C.T))
	col2 = sparse.vstack((C, sparse.csc_matrix((C.shape[1], C.shape[1]))))
	KKT = sparse.hstack((col1, col2))
	lhs = np.concatenate((u0, C.T.dot(u0)))
	u = sparse.linalg.spsolve(KKT.tocsc(), lhs)[0:len(u0)]
	
	eG = igl.eigen.SparseMatrixd()
	nV = np.concatenate((iV, u[:,np.newaxis]), axis=1)
	igl.grad(igl.eigen.MatrixXd(nV), igl.eigen.MatrixXi(T), eG)
	eu = igl.eigen.MatrixXd(u)
	eGu = (eG*eu).MapMatrix(len(T), 3)
	Gu = e2p(eGu)

	gradu = np.zeros(len(T))
	uvecs = np.zeros((len(T),2))
	for i in range(len(T)):
		e = T[i]
		
		# area2 = 2*get_area(mesh.V[e[0]], mesh.V[e[1]], mesh.V[e[2]])
		# p0 = np.concatenate((mesh.V[e[0]], [0]))
		# p1 = np.concatenate((mesh.V[e[1]], [0]))
		# p2 = np.concatenate((mesh.V[e[2]], [0]))
		# normal = get_unit_normal(p0, p1, p2)
		# s1 = u[e[0]]*np.cross(normal, mesh.V[e[1]] - mesh.V[e[2]])[0:2]
		# s2 = u[e[1]]*np.cross(normal, mesh.V[e[2]] - mesh.V[e[0]])[0:2]
		# s3 = u[e[2]]*np.cross(normal, mesh.V[e[0]] - mesh.V[e[1]])[0:2]
		# uvec = (1/area2)*(s1+s2+s3)
		uvec = Gu[i,0:2]
		uvecs[i,:] = uvec
		veca = uvec
		vecb = np.array([1,0])
		# theta = np.arccos(np.dot(veca,vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))
		x1 = np.cross(veca, vecb).dot(np.array([0,0,1]))
		x2 = np.linalg.norm(veca)*np.linalg.norm(vecb) + veca.dot(vecb)
		theta = 2*np.arctan2(x1, x2)[2]
		# print(theta)
		gradu[i] = theta

	return gradu, u, eGu, uvecs

def display_full_mesh():
	red = igl.eigen.MatrixXd([[1,0,0]])
	purple = igl.eigen.MatrixXd([[1,0,1]])
	green = igl.eigen.MatrixXd([[0,1,0]])
	black = igl.eigen.MatrixXd([[0,0,0]])
	blue = igl.eigen.MatrixXd([[0,0,1]])
	white = igl.eigen.MatrixXd([[1,1,1]])

	randc = [[random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)] for i in range(10)]

	viewer = igl.glfw.Viewer()
	Fix = []
	Mov = []
	x0 = np.ravel(V[:,:2])
	C = getC(T)
	A = getA(V, T)
	middle_button_down = False

	def mouse_move(viewer, mx, my):
		pass
		# print(middle_button_down)
		# if middle_button_down:
		# 	# Cast a ray in the view direction starting from the mouse position
		# 	bc = igl.eigen.MatrixXd()
		# 	fid = igl.eigen.MatrixXi(np.array([-1]))
		# 	coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
		# 	hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
		# 	viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(V), igl.eigen.MatrixXi(T), fid, bc)
		# 	ind = e2p(fid)[0][0]
		# 	muscles[ind] = 0
		# 	return True
	
	def mouse_up(viewer, btn, bbb):
		middle_button_down = False

	def mouse_down(viewer, btn, bbb):
		# Cast a ray in the view direction starting from the mouse position
		bc = igl.eigen.MatrixXd()
		fid = igl.eigen.MatrixXi(np.array([-1]))
		coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
		hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
		viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(V), igl.eigen.MatrixXi(T), fid, bc)
		ind = e2p(fid)[0][0]

		if btn==1:
			middle_button_down = True

		if hit and btn==0:
			# paint hit red
			Fix.append(T[ind][np.argmax(bc)])
			print("fix",T[ind][np.argmax(bc)])
			return True
		
		if hit and btn==2:
			# paint hit red
			Mov.append(T[ind][np.argmax(bc)])
			print("mov",T[ind][np.argmax(bc)])
			return True

		if hit and btn==1:
			muscles[ind] = 0
		
		return False

	def key_down(viewer,aaa, bbb):
		viewer.data().clear()
		viewer.data().set_mesh(V2, T2)
		if (aaa==65):
			#run heat flow on mesh
			u, uvec, eGu, UVECS = heat_method(V, T, Fix, Mov)
			CAx0 = C.dot(A.dot(x0))
			for i in range(len(T)):
				c = np.matrix([CAx0[6*i:6*i+2],CAx0[6*i:6*i+2]])
				alpha = u[i]
				cU, sU = np.cos(alpha), np.sin(alpha)
				U = np.array(((cU,-sU), (sU, cU)))
				scaledU = np.multiply(U, np.array([[.1],[.1]])) + c
				viewer.data().add_edges(igl.eigen.MatrixXd(c[0,:]), igl.eigen.MatrixXd(scaledU[0,:]), black)
		
		if (aaa==65):
			Colors = np.ones(T.shape)
			for t in range(len(T)):
				if(muscles[t]==0):
					Colors[t,:] = black
			viewer.data().set_colors(igl.eigen.MatrixXd(np.array(Colors)))

	def pre_draw(viewer):
		fixed_pts = []
		for i in range(len(Fix)):
			fixed_pts.append(V[Fix[i]])
		viewer.data().add_points(igl.eigen.MatrixXd(np.array(fixed_pts)), red)
		mov_pts = []
		for i in range(len(Mov)):
			mov_pts.append(V[Mov[i]])
		viewer.data().add_points(igl.eigen.MatrixXd(np.array(mov_pts)), green)
		
	key_down(viewer, "b", 123)
	# viewer.callback_mouse_up = mouse_up
	viewer.callback_mouse_move = mouse_move
	viewer.callback_mouse_down = mouse_down
	viewer.callback_key_down = key_down
	viewer.callback_pre_draw = pre_draw
	viewer.core.is_animating = False
	viewer.launch()

display_full_mesh()




