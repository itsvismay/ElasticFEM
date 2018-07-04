#helper functions

import numpy as np
import scipy
from scipy.spatial import Delaunay
from scipy import sparse
import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.3f}'.format(x)})
from iglhelpers import *

def general_eig_solve(A, B=None, modes=None):
	#pass in A = K matrix, and B = M matrix
	print("+General Eig Solve")
	
	if modes is None:
		e, ev = scipy.sparse.linalg.eigsh(A.tocsc(), M=B.tocsc(), which="SM")
	else:
		if(A.shape[0]<= modes):
			print("Too many modes")
			exit()
		e, ev = scipy.sparse.linalg.eigsh(A.tocsc(), M=B.tocsc(), k= modes+2, which="SM")

	print("-Done with Eig Solve")
	return e, ev

def snapshot_basis(filen):
	
	U = []
	for i in range(3,8):
		u = igl.eigen.MatrixXd()
		igl.readDMAT(filen+str(i)+".dmat", u)
		U.append(e2p(u))

	A = np.array(U)
	red, S, V = np.linalg.svd(A[:,:,0].T)
	return red[:,:len(S)]

def generate_euclidean_weights(CAx, handles, others):
	W = np.zeros((len(handles) + len(others), len(handles)))

	for i in range(len(handles)):
		caxi = CAx[6*handles[i]:6*handles[i]+2]
		W[handles[i], i] = 1
		for j in range(len(others)):
			caxj = CAx[6*others[j]:6*others[j]+2 ]
			d = np.linalg.norm(caxi - caxj)

			W[others[j], i] = 1.0/d
	for j in range(len(others)):
		W[others[j],:] /= np.sum(W[others[j],:])

	return np.kron(W, np.eye(3))

def get_area(p1, p2, p3):
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))*0.5

def get_centroid(p1, p2, p3):
	return (np.array(p1)+np.array(p2)+np.array(p3))/3.0

def torus_mesh(r1, r2, r3, step):
	V = []
	T = []
	for theta in range(0, 80):
		angle = theta*np.pi/69
		if(angle<=np.pi):
			V.append([step*r1*np.cos(angle), step*r1*np.sin(angle)])
			V.append([step*r2*np.cos(angle), step*r2*np.sin(angle)])
			V.append([step*r3*np.cos(angle), step*r3*np.sin(angle)])

	V.append([0,0])
	for e in Delaunay(V).simplices:
		if e[0]!=len(V)-1 and e[1]!=len(V)-1 and e[2]!=len(V)-1 and get_area(V[e[0]], V[e[1]], V[e[2]])<5:
			T.append([e[0], e[1], e[2]])
			# T.append(list(e))

	return np.array(V[:len(V)-1]), np.array(T), None

def triangle_mesh():
	V = [[0,0], [1,0], [1,1]]
	T = [[0,1,2]]
	return V, T, [0]

def get_min_max(iV,a, eps=1e-1):
	mov = []
	miny = np.amin(iV, axis=0)[a]
	maxy = np.amax(iV, axis=0)[a]
	for i in range(len(iV)):
		if(abs(iV[i][a] - miny) < eps):
			mov.append(i)

		if(abs(iV[i][a] - maxy)< eps):
			mov.append(i)

	return mov

def get_min(iV, a, eps=1e-1):
	mov = []
	miny = np.amin(iV, axis=0)[a]
	for i in range(len(iV)):
		if(abs(iV[i][a] - miny) < eps):
			mov.append(i)

	return mov

def get_max(iV, a, eps=1e-1):
	mov = []
	maxy = np.amax(iV, axis=0)[a]
	for i in range(len(iV)):
		if(abs(iV[i][a] - maxy)< eps):
			mov.append(i)

	return mov

def get_corners(iV, top=True, eps=1e-1):
	iV = np.array(iV)

	maxs = np.array(get_max(iV, a=1, eps=eps))
	tr = maxs[get_max(iV[maxs,:], a=0, eps = eps)]
	tl = maxs[get_min(iV[maxs,:], a=0, eps = eps)]

	mins = np.array(get_min(iV, a=1, eps=eps))
	br = mins[get_max(iV[mins,:], a=0, eps = eps)]
	bl = mins[get_min(iV[mins,:], a=0, eps = eps)]

	return tr[0], tl[0], br[0], bl[0]

def get_unit_normal(p1, p2, p3):
	n = np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3)))
	return n/np.linalg.norm(n)

def rectangle_mesh(x, y, step=1):
	V = []
	for i in range(0,x+1):
		for j in range(0,y+1):
			V.append([step*i, step*j])
	
	T = Delaunay(V).simplices
	return V, T

def feather_muscle1_test_setup(x = 3, y = 2):
	step = 0.1
	V,T,U = rectangle_mesh(x, y, step=step)
	# V,T, U = torus_mesh(5, 4, 3, step)

	half_x = step*(x)/2.0
	half_y = step*(y)/2.0
	u = []
	for i in range(len(T)):
		e = T[i]
		c = get_centroid(V[e[0]], V[e[1]], V[e[2]])
		if(c[1]<half_y):
			u.append(-0.15)
		else:
			u.append(0.15)

	to_fix =[]
	for i in get_min_max(V,1):
		if(V[i][0]>half_x):
			to_fix.append(i)

	return (V, T, u), to_fix

def feather_muscle2_test_setup(r1 =1, r2=2, r3=3, r4 = 4, p1 = 50, p2 = 25):
	step = 0.1
	V = []
	T = []
	u = []
	
	for theta in range(0, p1):
		angle = theta*np.pi/p2
		# for i in range(2):
		V.append([2*step*r1*np.cos(angle), step*r1*np.sin(angle)])
	print(np.array(V))
	T = Delaunay(V).simplices


	to_fix =get_max(V,0)
	return (V, T, u), to_fix

def heat_method(mesh):
	t = 1e-1
	eLc = igl.eigen.SparseMatrixd()
	igl.cotmatrix(igl.eigen.MatrixXd(mesh.V), igl.eigen.MatrixXi(mesh.T), eLc)
	Lc = e2p(eLc)

	M = mesh.getMassMatrix()
	Mdiag = M.diagonal()[2*np.arange(Lc.shape[0])]
	Mc = sparse.diags(Mdiag)


	#Au = b st. Cu = Cu0
	u0 = np.zeros(len(mesh.V))
	fixed = list(set(mesh.fixed) - set(mesh.mov))
	u0[fixed] = 2
	u0[mesh.mov] = -2

	Id = sparse.eye(len(mesh.V)).tocsc()
	fixedverts = [i for i in range(len(u0)) if u0[i]!=0]
	C = Id[:,fixedverts]

	A = (Mc - t*Lc)
	# u = sparse.linalg.spsolve(A.tocsc(), u0)
	col1 = sparse.vstack((A, C.T))
	col2 = sparse.vstack((C, sparse.csc_matrix((C.shape[1], C.shape[1]))))
	KKT = sparse.hstack((col1, col2))
	lhs = np.concatenate((u0, C.T.dot(u0)))
	u = sparse.linalg.spsolve(KKT.tocsc(), lhs)[0:len(u0)]
	
	eG = igl.eigen.SparseMatrixd()
	nV = np.concatenate((mesh.V, u[:,np.newaxis]), axis=1)
	igl.grad(igl.eigen.MatrixXd(nV), igl.eigen.MatrixXi(mesh.T), eG)
	eu = igl.eigen.MatrixXd(u)
	eGu = (eG*eu).MapMatrix(len(mesh.T), 3)
	Gu = e2p(eGu)

	gradu = np.zeros(len(mesh.T))
	uvecs = np.zeros((len(mesh.T),2))
	for i in range(len(mesh.T)):
		e = mesh.T[i]
		
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