import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import numdifftools as nd
import random
import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.5f}'.format(x)})

V = []
T = []
E = [] #each dg edge element is (nf, T_ind1, T_ind2, V_ind1, V_ind2)

def get_area(p1, p2, p3):
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))

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
	rectangle_mesh(2, 2)
	T = Delaunay(V).simplices #[e for e in Delaunay(V).simplices if get_area(V[e[0]], V[e[1]], V[e[2]])<3]

else:
	torus_mesh(7, 9)
	for e in Delaunay(V).simplices:
		if get_area(V[e[0]], V[e[1]], V[e[2]])<5:
			T.append(list(e))

def createPBlockingMatrix():
	to_fix = [0,1]
	P = np.kron(np.delete(np.eye(len(V)), to_fix, axis =1), np.eye(2))
	return P

#swapped A and P here by accident
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

#swapped A and P by accident
def createA(): 
	#edge matrix for each tet
	A = np.zeros((6*len(T), 6*len(T)))
	sub_A = np.kron(np.matrix([[-1, 1, 0], [0, -1, 1], [-1, 0, 1]]), np.eye(2))
	for i in range(len(T)):
		A[6*i:6*i+6, 6*i:6*i+6] = sub_A
	return A 

AP = np.matmul(createA(), createP())

q = np.zeros((1+2+4)*len(T)) #1degree for 2d rotation, 2 degree 2d translation, 4 degrees 2d strain

def set_initial_strains(sx, sy):
	for i in range(len(T)):
		q[7*i+3] = sx
		if i==0:
			q[7*i+6] = 0.9
		else:
			q[7*i+6] = sy
			
set_initial_strains(1, 1)

def getU(ind):
	if ind%2== 1:
		alpha = np.pi/4
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


def createGlobalR():
	GF = np.zeros((6*len(T), 6*len(T)))
	GR = np.zeros((6*len(T), 6*len(T)))
	GS = np.zeros((6*len(T), 6*len(T)))
	GU = np.zeros((6*len(T), 6*len(T)))
	for i in range(len(T)):
		F = getF(i)
		r = getR(i)
		s = getS(i)
		u = getU(i)
		F_e = np.kron(np.eye(3), F)
		r_e = np.kron(np.eye(3), r)
		s_e = np.kron(np.eye(3), s)
		u_e = np.kron(np.eye(3), u)
		GF[6*i:6*i+6, 6*i:6*i+6] = F_e
		GR[6*i:6*i+6, 6*i:6*i+6] = r_e
		GS[6*i:6*i+6, 6*i:6*i+6] = s_e
		GU[6*i:6*i+6, 6*i:6*i+6] = u_e
	# print(GF)
	# print(GR.dot(GU.dot(GS.dot(GU.T))))
	# exit()
	return GF, GR, GS, GU

A = createA()
P = createP()
BLOCK = createBlockingMatrix()

def updatePg(ng):
	print(ng)
	Pg = P.dot(ng)
	return Pg


def FiniteDifferences(g):
	x = np.ravel(V)
	print(len(V), len(T))
	GF, GR, GS, GU = createGlobalR()

	def E(_g, _bigR, _bigS, _bigU):
		totalE = 0
		PAg = A.dot(P.dot(_g))
		FPAx = _bigR.dot(_bigU.dot(_bigS.dot(_bigU.T.dot(A.dot(P.dot(x))))))
		totalE += 0.5*np.inner(PAg, PAg)
		totalE -= np.inner(PAg, FPAx)
		totalE += 0.5*np.inner(FPAx, FPAx)
		return totalE

	def dEdR():
		print("q", q)
		bigF, bigR, bigS, bigU = createGlobalR()
		USUT = bigU.dot(bigS.dot(bigU.T))
		PAg = A.dot(P.dot(g))
		return -1*np.outer(PAg, USUT.dot(A.dot(P.dot(x))))

	def dEdg():
		t1 = P.T.dot(A.T.dot(A.dot(P.dot(g))))
		t2 = P.T.dot(A.T.dot(GF.dot(A.dot(P.dot(x)))))
		return t1-t2

	def dEdS():
		bigF, bigR, bigS, bigU = createGlobalR()
		UPAx = bigU.T.dot(A.dot(P.dot(x)))
		RU = bigR.dot(bigU)
		PAg = A.dot(P.dot(g))

		return np.multiply.outer(bigS.dot(UPAx), UPAx) - np.multiply.outer(np.dot(RU.T, PAg), UPAx)

	default_E = E(g, GR, GS, GU)
	# print("Earap", default_E)

	#Wiggle g
	#pretty sure this is right, because I use it in ARAP
	# print("Wiggle G")
	# wiggleg = np.zeros(len(g)) + g
	# wiggleg[0] += 0.0001
	# print("Same?",dEdg()[0], (E(wiggleg, GR) - default_E)/0.0001)
	

	#Wiggle R
	# print("Wiggle R")
	# q[0] += 0.0001
	# wiggleGF, wiggleGR, wiggleGS, wiggleGU = createGlobalR()
	# u_r_h = E(g, wiggleGF)
	# print( dEdR())
	# print((GR - wiggleGR)/0.0001)
	# print(dEdR().dot((GR - wiggleGR)/0.0001))
	# print("Same?", (u_r_h - default_E)/0.0001)
	# q[0] = 0

	#Wiggle S
	print("dEdS")
	# q[7*0+3] += 0.001
	wiggleGF, wiggleGR, wiggleGS, wiggleGU = createGlobalR()
	wiggleGS[1,1] += 0.0001
	print(dEdS())
	u_s_h = E(g, GR, wiggleGS, GU)
	print("Same?", (u_s_h - default_E)/0.0001)

# FiniteDifferences(g = np.ravel(V))
# exit()


def solve():
	x = np.ravel(V)
	g = np.ravel(V)
	Px = P.dot(x)
	AP = np.matmul(A, P)
	PTAT = np.matmul(P.T, A.T)
	PTATAP = np.matmul(PTAT, AP)
	BPAAPB = BLOCK.T.dot(PTATAP.dot(BLOCK))
	print(PTATAP)
	exit()
	if(False):
		PTATAPInv = np.linalg.pinv(BPAAPB)
	else:
		KKT_matrix1 = np.concatenate((np.eye(BPAAPB.shape[0]), BPAAPB), axis=0)
		KKT_matrix2 = np.concatenate((BPAAPB.T, np.zeros(BPAAPB.shape)), axis=0)
		KKT = np.concatenate((KKT_matrix1, KKT_matrix2), axis=1)
		#print("KKT", scipy.linalg.eig(KKT)[0])
		CholFac, Lower = scipy.linalg.lu_factor(KKT)

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
			# print("theta", np.linalg.det(R), theta)
			q[7*i] = theta

			# exit()

	def updateT():
		globalR = createGlobalR()[0]
		APx = A.dot(Px)
		GRAPx = globalR.dot(APx)
		BPTATGRAPx = BLOCK.T.dot(PTAT.dot(GRAPx))

		if(False):
			newg = BLOCK.dot(PTATAPInv.dot(BPTATGRAPx))
			# print(newg)
		else:
			ob = np.concatenate((np.zeros(len(BPTATGRAPx)), BPTATGRAPx))
			# print(ob)
			gu = scipy.linalg.lu_solve((CholFac, Lower), ob)
			# print(gu)
			newg = BLOCK.dot(gu[0:len(BPTATGRAPx)])
		return newg

	for i in range(10):
		updateR(g)
		g = updateT()


	RecV = np.zeros((3*len(T), 2))
	RecT = []
	for t in range(len(T)):
		tv0 = getF(t).dot(Px[6*t:6*t+2])
		tv1 = getF(t).dot(Px[6*t+2:6*t+4])
		tv2 = getF(t).dot(Px[6*t+4:6*t+6])
		RecV[3*t,   0] = tv0[0]
		RecV[3*t,   1] = tv0[1]
		RecV[3*t+1, 0] = tv1[0]
		RecV[3*t+1, 1] = tv1[1]
		RecV[3*t+2, 0] = tv2[0]
		RecV[3*t+2, 1] = tv2[1]
		RecT.append([3*t, 3*t+1, 3*t+2])

	FiniteDifferences(g)
	# RecV = np.zeros((2*len(V), 2))
	# RecT = T 
	# for i in range(len(g)/2):
	# 	RecV[i, 0] = g[2*i]
	# 	RecV[i, 1] = g[2*i+1]


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

