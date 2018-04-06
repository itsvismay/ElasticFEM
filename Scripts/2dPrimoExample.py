import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import numdifftools as nd
import random
import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.2f}'.format(x)})

V = []
T = []
E = [] #each dg edge element is (nf, T_ind1, T_ind2, V_ind1, V_ind2)

def get_area(p1, p2, p3):
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))*0.5

def get_centroid(p1, p2, p3):
	return (np.array(p1)+np.array(p2)+np.array(p3))/3.0

def rectangle_mesh(x, y):
	for i in range(0,x):
		for j in range(0,y):
			V.append([i, j])

def torus_mesh(r1, r2):
	for theta in range(0, 12):
		angle = theta*np.pi/6.0
		V.append([r1*np.cos(angle), r1*np.sin(angle)])
		V.append([r2*np.cos(angle), r2*np.sin(angle)])

if(True):	
	rectangle_mesh(4, 4)
	T = Delaunay(V).simplices #[e for e in Delaunay(V).simplices if get_area(V[e[0]], V[e[1]], V[e[2]])<3]

else:
	torus_mesh(7, 9)
	for e in Delaunay(V).simplices:
		if get_area(V[e[0]], V[e[1]], V[e[2]])<5:
			T.append(list(e))


print(V)
print(T)
def createP():
	#P = 6Tx2V matrix that creates tet-based positions from V and T
	P = np.zeros((6*len(T), 2*len(V)))
	for i in range(len(T)):
		e = T[i]
		for j in range(len(e)):
			v = e[j]
			P[6*i+2*j, 2*v] = 1
			P[6*i+2*j+1, 2*v+1] = 1
	return P 

def createA():
	#edge matrix for each tet
	A = np.zeros((6*len(T), 6*len(T)))
	sub_A = np.kron(np.matrix([[-1, 1, 0], [0, -1, 1], [-1, 0, 1]]), np.eye(2))
	for i in range(len(T)):
		A[6*i:6*i+6, 6*i:6*i+6] = sub_A
	return A 

AP = np.matmul(createA(), createP())

def internal_edges():
	_n_ = 100
	edges = {}
	for i in range(len(T)):
		p1 = (T[i][0], T[i][1])
		p2 = (T[i][2], T[i][1])
		p3 = (T[i][2], T[i][0])

		if p1 in edges:
			area1 = get_area(V[T[i][0]], V[T[i][1]], V[T[i][2]])
			area2 = get_area(V[T[edges[p1]][0]], V[T[edges[p1]][1]], V[T[edges[p1]][2]])
			nf = _n_*np.linalg.norm(np.array(V[p1[0]]) - np.array(V[p1[1]]))*((1.0/area1) + (1.0/area2))
			E.append((nf, edges[p1], i, p1[0], p1[1]))
		else:
			edges[(T[i][0], T[i][1])] = i 
			edges[(T[i][1], T[i][0])] = i

		if p2 in edges:
			area1 = get_area(V[T[i][0]], V[T[i][1]], V[T[i][2]])
			area2 = get_area(V[T[edges[p2]][0]], V[T[edges[p2]][1]], V[T[edges[p2]][2]])
			nf = _n_*np.linalg.norm(np.array(V[p2[0]]) - np.array(V[p2[1]]))*((1.0/area1) + (1.0/area2)) 
			E.append((nf, edges[p2], i, p2[0], p2[1]))
		else:
			edges[(T[i][2], T[i][1])] = i 
			edges[(T[i][1], T[i][2])] = i

		if p3 in edges:
			area1 = get_area(V[T[i][0]], V[T[i][1]], V[T[i][2]])
			area2 = get_area(V[T[edges[p3]][0]], V[T[edges[p3]][1]], V[T[edges[p3]][2]])
			nf = _n_*np.linalg.norm(np.array(V[p3[0]]) - np.array(V[p3[1]]))*((1.0/area1) + (1.0/area2))
			E.append((nf, edges[p3], i, p3[0], p3[1]))
		else:
			edges[(T[i][0], T[i][2])] = i 
			edges[(T[i][2], T[i][0])] = i

# internal_edges()

q = np.zeros((1+2+4)*len(T)) #1degree for 2d rotation, 2 degree 2d translation, 4 degrees 2d strain
r = np.zeros((1+2)*len(T))

def set_initial_strains(sx, sy):
	for i in range(len(T)):
		q[7*i+3] = sx
		if i==1:
			q[7*i+6] = sy
		else:
			q[7*i+6] = sy
			
set_initial_strains(1, 0.5)

def getU(ind):
	if ind%2== 1:
		alpha = np.pi/4.5
	else:
		alpha = np.pi/4

	cU, sU = np.cos(alpha), np.sin(alpha)
	U = np.array(((cU,-sU), (sU, cU)))
	return U

def getR(ind):
	theta = q[7*ind]
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c,-s), (s, c)))
	# print("R",scipy.linalg.logm(R))
	return R

def getS(ind):
	S = np.array([[q[7*ind+3], q[7*ind+4]], [q[7*ind+5], q[7*ind+6]]])  
	return S

def getF(ind):
	U = getU(ind)
	S = getS(ind)
	R = getR(ind)
	F =  np.matmul(R, np.matmul(U, np.matmul(S, U.transpose())))
	return F

def reduced_to_q(lr):
	for i in range(len(T)):
		q[7*i]   = lr[3*i] 
		q[7*i+1] = lr[3*i+1]
		q[7*i+2] = lr[3*i+2]

def q_to_reduced():
	for i in range(len(T)):
		r[3*i]  =q[7*i]   
		r[3*i+1]=q[7*i+1] 
		r[3*i+2]=q[7*i+2] 

def createGlobalR():
	GR = np.zeros((6*len(T), 6*len(T)))
	for i in range(len(T)):
		F = getF(i)
		F_e = np.kron(np.eye(3), F)
		GR[6*i:6*i+6, 6*i:6*i+6] = F_e
	return GR

A = createA()
P = createP()

def updatePg(ng):
	print(ng)
	Pg = P.dot(ng)
	return Pg

def solve():
	x = np.ravel(V)
	g = np.ravel(V)
	Px = P.dot(x)
	AP = np.matmul(A, P)
	PTAT = np.matmul(P.T, A.T)
	PTATAP = np.matmul(PTAT, AP)
	PTATAPInv = np.linalg.pinv(PTATAP)
	# CholFac, Lower = scipy.linalg.cho_factor(PTATAP)

	def updateR(g):
		Pg = updatePg(g)
		sub_A = np.kron(np.matrix([[-1, 1, 0], [0, -1, 1], [-1, 0, 1]]), np.eye(2))
		for i in range(len(T)):
			e = T[i]
			APx =sub_A.dot(Px[6*i:6*i+6])
			APg =sub_A.dot(Pg[6*i:6*i+6])
			Ue = np.kron(np.eye(3), getU(i))
			Se = np.kron(np.eye(3), getS(i))
			USUAPx = Ue.dot(Se.dot(Ue.T.dot(APx.T)))

			m_APg = np.zeros((3,2))
			m_APg[0:1, 0:2] = APg[0,0:2]
			m_APg[1:2, 0:2] = APg[0,2:4]
			m_APg[2:3, 0:2] = APg[0,4:6]

			m_USUAPx = np.zeros((3,2))
			m_USUAPx[0:1, 0:2] = USUAPx[0:2].T
			m_USUAPx[1:2, 0:2] = USUAPx[2:4].T
			m_USUAPx[2:3, 0:2] = USUAPx[4:6].T
			F = np.matmul(m_APg.T,m_USUAPx)

			U, s, VT = np.linalg.svd(F, full_matrices=True)
			R = np.matmul(VT.T, U.T)
			veca = np.array([1,0])
			vecb = np.dot(R, veca)
			theta = np.arccos(np.dot(veca,vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))
			print("theta", np.linalg.det(R), theta)
			q[7*i] = theta

			# exit()

	def updateT():
		globalR = createGlobalR()
		APx = A.dot(Px)
		GRAPx = globalR.dot(APx)
		PTATGRAPx = PTAT.dot(GRAPx)
		newg = PTATAPInv.dot(PTATGRAPx)
		return newg

	for i in range(10):
		updateR(g)
		g = updateT()


	# RecV = np.zeros((3*len(T), 2))
	# RecT = []
	# for t in range(len(T)):
	# 	tv0 = getF(t).dot(Px[6*t:6*t+2])
	# 	tv1 = getF(t).dot(Px[6*t+2:6*t+4])
	# 	tv2 = getF(t).dot(Px[6*t+4:6*t+6])
	# 	RecV[3*t,   0] = tv0[0]
	# 	RecV[3*t,   1] = tv0[1]
	# 	RecV[3*t+1, 0] = tv1[0]
	# 	RecV[3*t+1, 1] = tv1[1]
	# 	RecV[3*t+2, 0] = tv2[0]
	# 	RecV[3*t+2, 1] = tv2[1]
	# 	RecT.append([3*t, 3*t+1, 3*t+2])

	RecV = np.zeros((2*len(V), 2))
	RecT = T 
	for i in range(len(g)/2):
		RecV[i, 0] = g[2*i]
		RecV[i, 1] = g[2*i+1]


	print("Reconstructed V")
	print(RecV, RecT)
	return RecV, RecT

def display():
	viewer = igl.viewer.Viewer()
	def key_down(viewer, b, c):

		V_new, T1 = solve()

		V2 = igl.eigen.MatrixXd(V_new)
		T2 = igl.eigen.MatrixXi(T1)
		viewer.data.set_mesh(V2, T2)

		red = igl.eigen.MatrixXd([[1,0,0]])
		i =0
		for e in T1:
			centroid = get_centroid(V_new[e[0]], V_new[e[1]], V_new[e[2]])
			C = np.matrix([centroid,centroid])
			U = 0.1*getU(i).transpose()+C
			viewer.data.add_edges(igl.eigen.MatrixXd(C),
							igl.eigen.MatrixXd(U),
							red)
			i+=1

		return True
	key_down(viewer, "a", 1)
	viewer.core.is_animating = False
	viewer.callback_key_down = key_down
	viewer.launch()

display()

