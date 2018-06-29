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

def general_eig_solve(A, B=None, modes=5):
	#pass in A = K matrix, and B = M matrix
	print("+General Eig Solve")
	if(A.shape[0]<= modes):
		print("Too many modes")
		exit()
	# A1 = A.toarray()
	# B1 = B.toarray()
	# e, ev = scipy.linalg.eigh(A1, b=B1, eigvals=(0,modes))
	e, ev = scipy.sparse.linalg.eigsh(A.tocsc(), M=B.tocsc(), k = modes, which="SM")

	eigvals = e[0:modes]
	eigvecs = ev[:, 0:modes]
	print("-Done with Eig Solve")
	return eigvals, eigvecs

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