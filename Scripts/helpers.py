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

def generate_bbw_matrix():
	# List of boundary indices (aka fixed value indices into VV)
	b = igl.eigen.MatrixXi()
	# List of boundary conditions of each weight function
	bc = igl.eigen.MatrixXd()

	igl.boundary_conditions(V, T, C, igl.eigen.MatrixXi(), BE, igl.eigen.MatrixXi(), b, bc)

	bbw_data = igl.BBWData()
	# only a few iterations for sake of demo
	bbw_data.active_set_params.max_iter = 8
	bbw_data.verbosity = 2
	if not igl.bbw(V, T, b, bc, bbw_data, W):
		exit(-1)

	# Normalize weights to sum to one
	igl.normalize_row_sums(W, W)
	# precompute linear blend skinning matrix
	igl.lbs_matrix(V, W, M)

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

def rectangle_mesh(x, y, angle=0, step=1):
	V = []
	for i in range(0,x+1):
		for j in range(0,y+1):
			V.append([step*i, step*j])
	# V.append([0,0])
	# V.append([step*x, 0])
	# V.append([0, step*y])
	# V.append([step*x, step*y])
	# for i in range(100):
	# 	V.append([random.uniform(0,step*x), random.uniform(0, step*y)])
	T = Delaunay(V).simplices
	return V, T, np.array([angle for i in range(len(T))])

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

def featherize(x, y, step=1):
	V,T,U = rectangle_mesh(x, y, step)
	# V,T, U = torus_mesh(5, 4, 3, step)

	half_x = step*(x)/2.0
	half_y = step*(y)/2.0
	u = []
	for i in range(len(T)):
		e = T[i]
		c = get_centroid(V[e[0]], V[e[1]], V[e[2]])
		if(c[0]<half_x):
			u.append((np.pi/2) - 0.1)
		else:
			u.append((np.pi/2) + 0.1)

	return V, T, u

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

def feather_muscle2_test_setup(r1 =1, r2=2, r3=3, r4 = 4, p1 = 10, p2 = 5):
	step = 0.1
	V = []
	T = []
	u = []
	V.append([(r4+1)*step, (r4+1)*step])
	V.append([(r4+1)*step + 1.5*step*r1, (r4+1)*step ])
	V.append([(r4+1)*step - 1.5*step*r1, (r4+1)*step ])
	V.append([(r4+1)*step + 1.75*step*r1, (r4+1)*step ])
	V.append([(r4+1)*step - 1.75*step*r1, (r4+1)*step ])
	for theta in range(0, p1):
		angle = theta*np.pi/p2
		# if(angle<=np.pi):
		V.append([2*step*r1*np.cos(angle) + (r4+1)*step, step*r1*np.sin(angle)+ (r4+1)*step])
		V.append([2*step*r2*np.cos(angle) + (r4+1)*step, step*r2*np.sin(angle)+ (r4+1)*step])
		V.append([2*step*r3*np.cos(angle) + (r4+1)*step, step*r3*np.sin(angle)+ (r4+1)*step])
		V.append([2*step*r4*np.cos(angle) + (r4+1)*step, step*r4*np.sin(angle)+ (r4+1)*step])

	T = Delaunay(V).simplices

	for i in range(len(T)):
		e = T[i]
		c = get_centroid(V[e[0]], V[e[1]], V[e[2]])
		if(c[1]< (step*(r4+1))):
			u.append(-0.15)
		else:
			u.append(0.15)


	to_fix =get_max(V,0)
	print(to_fix)
	return (V, T, u), to_fix
