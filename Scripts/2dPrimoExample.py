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


def get_area(p1, p2, p3):
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))*0.5

def get_centroid(p1, p2, p3):
	return (np.array(p1)+np.array(p2)+np.array(p3))/3.0

def rectangle_mesh(x, y):
	for i in range(0,x):
		for j in range(0,y):
			V.append([i, j])
rectangle_mesh(8, 8)

def torus_mesh(r1, r2):
	for theta in range(0, 12):
		angle = theta*np.pi/6.0
		V.append([r1*np.cos(angle), r1*np.sin(angle)])
		V.append([r2*np.cos(angle), r2*np.sin(angle)])
# torus_mesh(7, 9)

T = []
# for e in Delaunay(V).simplices:
# 	if get_area(V[e[0]], V[e[1]], V[e[2]])<5:
# 		T.append(list(e))

T = Delaunay(V).simplices #[e for e in Delaunay(V).simplices if get_area(V[e[0]], V[e[1]], V[e[2]])<3]
E = [] #each dg edge element is (nf, T_ind1, T_ind2, V_ind1, V_ind2)

# exit()

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
internal_edges()

q = np.zeros((1+2+4)*len(T)) #1degree for 2d rotation, 2 degree 2d translation, 4 degrees 2d strain
r = np.zeros((1+2)*len(T))

def set_initial_strains(s):
	for i in range(len(T)):
		q[7*i+3] = s
		q[7*i+6] = 0.3*s
			
set_initial_strains(1)

def getF(ind):
	theta = q[7*ind]
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c,-s), (s, c)))

	alpha = np.pi/4
	cU, sU = np.cos(alpha), np.sin(alpha)
	U = np.array(((cU,-sU), (sU, cU)))
	S = np.array([[q[7*ind+3], q[7*ind+4]], [q[7*ind+5], q[7*ind+6]]])  
	F =  np.matmul(R, np.matmul(U, np.matmul(S, U.transpose())))
	return F

def get_new_x():
	return getF(t).dot()
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

def solve():
	newV = np.zeros((len(V), len(V[0])))
	RecV = np.zeros((3*len(T), 2))
	RecT = []

	def func(r_k):
		reduced_to_q(r_k)

		totalE = 0
		for e in E:
			t1 = e[1]
			t2 = e[2]
			u1 = getF(t1).dot(np.array(V[e[3]]) - get_centroid(V[T[t1][0]], V[T[t1][1]], V[T[t1][2]])) + get_centroid(V[T[t1][0]], V[T[t1][1]], V[T[t1][2]]) + np.array([r[3*t1+1], r[3*t1+2]])
			u2 = getF(t1).dot(np.array(V[e[4]]) - get_centroid(V[T[t1][0]], V[T[t1][1]], V[T[t1][2]])) + get_centroid(V[T[t1][0]], V[T[t1][1]], V[T[t1][2]]) + np.array([r[3*t1+1], r[3*t1+2]])
			u3 = getF(t2).dot(np.array(V[e[3]]) - get_centroid(V[T[t2][0]], V[T[t2][1]], V[T[t2][2]])) + get_centroid(V[T[t2][0]], V[T[t2][1]], V[T[t2][2]]) + np.array([r[3*t2+1], r[3*t2+2]])
			u4 = getF(t2).dot(np.array(V[e[4]]) - get_centroid(V[T[t2][0]], V[T[t2][1]], V[T[t2][2]])) + get_centroid(V[T[t2][0]], V[T[t2][1]], V[T[t2][2]]) + np.array([r[3*t2+1], r[3*t2+2]])

			u_left = np.concatenate((u1, u2))
			u_right = np.concatenate((u3, u4))
			totalE += e[0]*np.dot((u_left - u_right), (u_left - u_right))
		return totalE

	def dfunc(r_k):
		J = nd.Gradient(func)(r_k)
		return J.ravel()


	# res = minimize(func, r, method='Nelder-Mead', options={'disp': True})
	for t in range(len(T)):
		tv0 = getF(t).dot(np.array(V[T[t][0]]) - get_centroid(V[T[t][0]], V[T[t][1]], V[T[t][2]])) + get_centroid(V[T[t][0]], V[T[t][1]], V[T[t][2]]) + np.array([q[7*t+1], q[7*t+2]])
		tv1 = getF(t).dot(np.array(V[T[t][1]]) - get_centroid(V[T[t][0]], V[T[t][1]], V[T[t][2]])) + get_centroid(V[T[t][0]], V[T[t][1]], V[T[t][2]]) + np.array([q[7*t+1], q[7*t+2]])
		tv2 = getF(t).dot(np.array(V[T[t][2]]) - get_centroid(V[T[t][0]], V[T[t][1]], V[T[t][2]])) + get_centroid(V[T[t][0]], V[T[t][1]], V[T[t][2]]) + np.array([q[7*t+1], q[7*t+2]])
		RecV[3*t,   0] = tv0[0]
		RecV[3*t,   1] = tv0[1]
		RecV[3*t+1, 0] = tv1[0]
		RecV[3*t+1, 1] = tv1[1]
		RecV[3*t+2, 0] = tv2[0]
		RecV[3*t+2, 1] = tv2[1]
		RecT.append([3*t, 3*t+1, 3*t+2])

		# newV[T[t][0], 0] = tv0[0]
		# newV[T[t][0], 1] = tv0[1]
		# newV[T[t][1], 0] = tv1[0]
		# newV[T[t][1], 1] = tv1[1]
		# newV[T[t][2], 0] = tv2[0]
		# newV[T[t][2], 1] = tv2[1]

	print("Reconstructed V")
	return RecV, RecT

def display():
	viewer = igl.viewer.Viewer()
	def key_down(viewer, b, c):

		V_new, T1 = solve()

		V1 = igl.eigen.MatrixXd(V_new)
		T2 = igl.eigen.MatrixXi(T1)
		viewer.data.set_mesh(V1, T2)

		return True
	key_down(viewer, "a", 1)
	viewer.core.is_animating = False
	viewer.callback_key_down = key_down
	viewer.launch()

display()

