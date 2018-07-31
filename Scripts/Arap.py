import numpy as np
import math
from collections import defaultdict
from sets import Set
import json
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy import sparse

import datetime
import random
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.3f}'.format(x)})
from iglhelpers import *

temp_png = os.path.join(os.getcwd(),"out.png")

		
class ARAP:
	#class vars
	def __init__(self, imesh, filen=None):
		print("Init ARAP")
		self.mesh = imesh
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix()

		#these are the fixed vertices which stay constant
		A = self.mesh.getA()
		P = self.mesh.getP()
		B = self.BLOCK
		C = self.ANTI_BLOCK.T
		AtPtPA = A.T.dot(P.T.dot(P.dot(A)))
		self.PAx = P.dot(A.dot(self.mesh.x0))
		self.Egg = None

		self.DSDs = None
		self.sparseDSds()
		
		#LU inverse
		self.Egg = self.mesh.G.T.dot(A.T.dot(P.T.dot(P.dot(A.dot(self.mesh.G)))))
		CQ = C.dot(self.mesh.G)
		col1 = sparse.vstack((self.Egg, CQ))
		col2 = sparse.vstack((CQ.T, sparse.csc_matrix((CQ.shape[0], CQ.shape[0]))))
		KKT = sparse.hstack((col1, col2))
		# print(self.mesh.G.shape)
		# print(self.Egg)
		# exit()
		self.CholFac = scipy.sparse.linalg.splu(KKT.tocsc())

		r_size = len(self.mesh.red_r)
		z_size = len(self.mesh.z)
		s_size = len(self.mesh.red_s)
		t_size = len(self.mesh.T)

		self.Erg = np.zeros((z_size, r_size))
		self.Err = np.zeros((r_size, r_size))
		self.Egs = np.zeros((z_size, s_size))
		self.Ers = np.zeros((r_size, s_size))
		self.Ers_mid = np.zeros(( 6*len(self.mesh.T), r_size ))
		self.Ers_first_odd = sparse.diags([np.ones(6*t_size-1), np.ones(6*t_size), np.ones(6*t_size-1)],[-1,0,1]).tocsc()
		self.Ers_first_even = sparse.diags([np.ones(6*t_size-1), np.ones(6*t_size), np.ones(6*t_size-1)],[-1,0,1]).tocsc()
		
		self.PA = P.dot(A)
		self.PAG = self.PA.dot(self.mesh.G)
		self.USUtPAx_E = []

		self.constErTerms = []
		self.constErTerms = []
		self.constErs_midTerms = []
		self.constErs_Terms = []
		self.constEgsTerms = []
		self.constUUtPaxTerms = []
		self.constEsTerms = []
		self.constItRTerms = []

		self.setupConstErTerms()
		self.setupConstErsTerms()
		self.setupConstEgsTerms()
		self.setupConstEsTerms()
		self.setupConstUSUtPAxTerms()
		self.setupConstItRTerms()
		print("Done with ARAP init")

	def energy(self, _z, _R, _S, _U):
		# self.updateConstUSUtPAx()
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.G.dot(_z) + self.mesh.x0)) 
		FPAx = _R.dot(_U.dot(_S.dot(_U.T.dot(self.PAx))))
		# print(PAg)
		# print(self.mesh.red_r)
		# print(self.mesh.red_s)
		return 0.5*(np.dot(PAg - FPAx, PAg - FPAx))

	def Energy(self):
		PAg = self.PA.dot(self.mesh.getg())
		FPAx0 = self.constTimeFPAx() #self.mesh.GF.dot(self.PAx)
		en = 0.5*(np.dot(PAg - FPAx0, PAg - FPAx0))
		return en

	def Jacobian(self, block = False, kkt= True, useSparse=True):
		print("		Jacobian")
		# self.mesh.getGlobalF(updateR=True, updateS=True, updateU=False)
		Egg, Erg, Err, Egs, Ers = self.Hessians(useSparse=useSparse)

		lhs_left = np.vstack((Egg, Erg.T))
		lhs_right = np.concatenate((Erg , Err))

		rhs = -1*np.concatenate((Egs, Ers))


		#Constraining Rotation
		R = np.array([[] for i in self.mesh.red_r]).T

		#NEW KKT SOLVE
		if kkt:
			C = self.ANTI_BLOCK.T.dot(self.mesh.G)
			r_size = C.shape[1]
			gb_size = C.shape[0]
			r_size = R.shape[1]
			rb_size = R.shape[0]

			
				
			col1 = np.vstack((lhs_left, C))
			# print("col", col1.shape, col1.nnz)
			col2 = np.vstack((lhs_right, np.zeros((gb_size, r_size))))
			# print("col", col2.shape, col2.nnz)
			col3 = np.vstack(( C.T,
				np.vstack((np.zeros((r_size, gb_size)),
							np.zeros((gb_size, gb_size))))))
			# print("col", col3.shape, col3.nnz)
			jacKKT = np.hstack((col1, col2, col3))

			KKT_constrains = np.vstack((rhs, np.zeros((gb_size+rb_size, rhs.shape[1]))))
			
			Jac_s = scipy.linalg.solve(jacKKT, KKT_constrains)


			results = Jac_s[0:rhs.shape[0], :]


		dgds = results[0:lhs_left.shape[1],:]
		drds = results[lhs_left.shape[1]:,:]
		Eg,Er,Es = self.Gradients()
		dEds = np.matmul(Eg, dgds) + np.matmul(Er, drds) + Es
		print("		Jacobian")
		return dEds, dgds, drds

	def Hessians(self, useSparse=True):
		print("			Hessians")
		# PAg = self.PA.dot(self.mesh.getg())
		# USUt = self.mesh.GU.dot(self.mesh.GS.dot(self.mesh.GU.T))
		# USUtPAx = USUt.dot(self.PAx)
		# UtPAx = self.mesh.GU.T.dot(self.PAx)
		# r_size = len(self.mesh.red_r)
		# z_size = len(self.mesh.z)
		# s_size = len(self.mesh.red_s)
		# DR = self.sparseDRdr()
		# DS = self.sparseDSds()
		Ezz = self.Egg

		
		# ###############ERG
		# sample = self.sparseErg_first(-1*sparse.csc_matrix(self.PAG), USUtPAx)
		# for j in range(self.Erg.shape[1]):
		# 	for i in range(self.Erg.shape[0]):
		# 		self.Erg[i,j] = DR[j].multiply(sample[i]).sum()
		self.Erg = self.constTimeErz()


		###############ERR
		# negPAg_USUtPAx = self.sparseOuterProdDiags(-1*PAg, USUtPAx)
		# DDR = self.sparseDDRdrdr_diag()
		# for i in range(self.Err.shape[0]):
		# 	spD = DDR[i]
		# 	if spD.nnz>0:
		# 		self.Err[i,i] = spD.multiply(negPAg_USUtPAx).sum()
		self.Err = self.constTimeErr()
		
		# ###############EGS
		# PAGtRU = self.PAG.T.dot(self.mesh.GR.dot(self.mesh.GU))
		# d_gEgdS = self.sparseEgs_first(-1*PAGtRU, UtPAx)
		# for i in range(self.Egs.shape[0]):
		# 	for j in range(self.Egs.shape[1]):
		# 		self.Egs[i,j] = DS[j].multiply(d_gEgdS[i]).sum()
		# print(self.Egs)
		self.Egs = self.constTimeEgs()
		
		# print(self.Egs)
		# exit()
		mid = self.constTimeErs_mid()
		self.Ers = self.constTimeErs_second(mid)
			
		print("			Hessians")
		return Ezz, self.Erg, self.Err, self.Egs, self.Ers

	def sparseErg_first(self, nPAT, USUtPAx):
		first = []
		cdd = np.zeros(len(USUtPAx))
		cdl = np.zeros(len(USUtPAx)-1)
		cdu = np.zeros(len(USUtPAx)-1)
		for c in range(nPAT.shape[1]):
			PATc = nPAT.getcol(c)
			dd = PATc[:,0].T.multiply(USUtPAx)
			du = PATc[:-1,0].T.multiply(USUtPAx[1:])
			dl = PATc[1:,0].T.multiply(USUtPAx[:-1])
			cdd[:] = dd.toarray()
			cdl[:] = dl.toarray()
			cdu[:] = du.toarray()
			sp = sparse.diags([cdl, cdd, cdu],[-1,0,1]).tocsc()
			first.append(sp)

		return first

	def sparseEgs_first(self, PAtRU, UtPAx):
		first = []
		cdd = np.zeros(len(UtPAx))
		cdl = np.zeros(len(UtPAx)-1)
		cdu = np.zeros(len(UtPAx)-1)
		for r in range(PAtRU.shape[0]):
			matr = PAtRU.getrow(r)
			dd = matr.multiply(UtPAx)
			du = matr[0,:-1].multiply(UtPAx[1:])
			dl = matr[0,1:].multiply(UtPAx[:-1])
			cdd[:] = dd.toarray()
			cdl[:] = dl.toarray()
			cdu[:] = du.toarray()
			sp = sparse.diags([cdl, cdd, cdu],[-1,0,1])
			first.append(sp)

		return first

	def sparseErs_first(self, nPAg):
		for t in range(len(self.mesh.T)):
			u = self.mesh.getU(t)
			self.Ers_first_odd[6*t + 0:6*t + 2, 6*t + 0:6*t + 2] = np.multiply.outer(u[:,0], nPAg[6*t + 0:6*t + 2])
			self.Ers_first_odd[6*t + 2:6*t + 4, 6*t + 2:6*t + 4] = np.multiply.outer(u[:,0], nPAg[6*t + 2:6*t + 4])
			self.Ers_first_odd[6*t + 4:6*t + 6, 6*t + 4:6*t + 6] = np.multiply.outer(u[:,0], nPAg[6*t + 4:6*t + 6])

			self.Ers_first_even[6*t + 0:6*t + 2, 6*t + 0:6*t + 2] = np.multiply.outer(u[:,1], nPAg[6*t + 0:6*t + 2])
			self.Ers_first_even[6*t + 2:6*t + 4, 6*t + 2:6*t + 4] = np.multiply.outer(u[:,1], nPAg[6*t + 2:6*t + 4])
			self.Ers_first_even[6*t + 4:6*t + 6, 6*t + 4:6*t + 6] = np.multiply.outer(u[:,1], nPAg[6*t + 4:6*t + 6])

		return self.Ers_first_odd, self.Ers_first_even

	def sparseErs_second(self, UtPAx, mid):
		#no odd/even necessary here because
		#each row of mid only has 1 non-zero entry
		second = []

		for c in range(mid.shape[1]):
			midc = mid[:,c]
			dd = np.multiply(UtPAx, midc)
			du = np.multiply(UtPAx[:-1], midc[1:])
			dl = np.multiply(UtPAx[1:], midc[:-1])

			second .append(sparse.diags([dl, dd, du], [-1,0,1]))

		return second

	def sparseOuterProdDiags(self, vec1, vec2):
		dd = np.multiply(vec1, vec2)
		du = np.multiply(vec1[1:], vec2[:-1])
		dl = np.multiply(vec1[:-1],vec2[1:])
		sp = sparse.diags([dl, dd, du], [-1, 0, 1]).tocsc()
		return sp

	def setupConstErTerms(self):
		UU = sparse.lil_matrix((3*len(self.mesh.T), 3*len(self.mesh.T))) 

		for t in range(len(self.mesh.T)):
			u = self.mesh.getU(t)
			u1, u2 = u[0,0], u[0,1]
			UU[3*t+0, 3*t+0] = u1*u1
			UU[3*t+0, 3*t+1] = u2*u2
			UU[3*t+0, 3*t+2] = 2*u1*u2

			UU[3*t+1, 3*t+0] = -u1*u2
			UU[3*t+1, 3*t+1] = u1*u2
			UU[3*t+1, 3*t+2] = u1*u1 - u2*u2

			UU[3*t+2, 3*t+0] = u2*u2
			UU[3*t+2, 3*t+1] = u1*u1
			UU[3*t+2, 3*t+2] = -2*u1*u2

		UUW = UU.tocsc().dot(self.mesh.sW)

		MPAx = sparse.lil_matrix((6*len(self.mesh.T), 3*len(self.mesh.T)))
		for t in range(len(self.mesh.T)):
			x1, x2, x3 = self.PAx[6*t+0], self.PAx[6*t+1], self.PAx[6*t+2]
			x4, x5, x6 = self.PAx[6*t+3], self.PAx[6*t+4], self.PAx[6*t+5]

			MPAx[6*t+0, 3*t+0] = x1
			MPAx[6*t+0, 3*t+1] = x2
			MPAx[6*t+1, 3*t+1] = x1
			MPAx[6*t+1, 3*t+2] = x2

			MPAx[6*t+2, 3*t+0] = x3
			MPAx[6*t+2, 3*t+1] = x4
			MPAx[6*t+3, 3*t+1] = x3
			MPAx[6*t+3, 3*t+2] = x4

			MPAx[6*t+4, 3*t+0] = x5
			MPAx[6*t+4, 3*t+1] = x6
			MPAx[6*t+5, 3*t+1] = x5
			MPAx[6*t+5, 3*t+2] = x6

		PAxUUtW = MPAx.tocsc().dot(UUW)


		for r in range(len(self.mesh.red_r)):
			B = self.mesh.RotationBLOCK[r]
			BU = B.T.dot(PAxUUtW)
			BPAG = B.T.dot(-1*self.PAG)
			BPAx0 = B.T.dot(-1*self.PAx)

			self.constErTerms.append((BU, BPAG, BPAx0))

	def setupConstErsTerms(self):
		def getOddEvenUMatrix(BUB):
			oddU = np.zeros((BUB.shape[0], 2))
			evenU = np.zeros((BUB.shape[0], 2))
			for i in range(BUB.shape[0]/2):
				u11 = self.mesh.GU[2*i, 2*i]
				u12 = self.mesh.GU[2*i, 2*i+1]
				u21 = self.mesh.GU[2*i+1, 2*i]
				u22 = self.mesh.GU[2*i+1, 2*i+1]
				oddU[2*i, 0] = u11
				oddU[2*i, 1] = u12
				oddU[2*i+1, 0] = u12
				oddU[2*i+1, 1] = u11

				evenU[2*i, 0] = u21
				evenU[2*i, 1] = -u22
				evenU[2*i+1, 0] = u22
				evenU[2*i+1, 1] = -u21
				
			return oddU, evenU
		

		for i in range(len(self.mesh.red_r)):
			B = self.mesh.RotationBLOCK[i]
			BPAG = B.T.dot(self.PAG) 
			BPAx = B.T.dot(self.PAx)
			c = np.array([-np.sin(self.mesh.red_r[i]), -np.cos(self.mesh.red_r[i])])
			BtUB = B.T.dot(self.mesh.GU.dot(B))
			oddU, evenU = getOddEvenUMatrix(BtUB)

			spdiag_oddcU = sparse.diags(oddU.dot(c)).tocsc()
			spdiag_evencU = sparse.diags(evenU.dot(c)).tocsc()
			oddcUPAG = spdiag_oddcU*BPAG
			evencUPAG = spdiag_evencU*BPAG
			oddcUPAx = spdiag_oddcU.dot(BPAx)
			evencUPAx = spdiag_evencU.dot(BPAx)
			self.constErs_midTerms.append((B.dot(oddcUPAG), B.dot(evencUPAG), B.dot(oddcUPAx), B.dot(evencUPAx) ))
		
		UtPAx = self.mesh.GU.T.dot(self.PAx)
		UtPAxDS = np.zeros((len(UtPAx), len(self.mesh.red_s)))
		for t in range(len(self.mesh.red_s)/3):
				sWx = self.mesh.sW[:,3*t]
				sWy = self.mesh.sW[:,3*t+1]
				sWo = self.mesh.sW[:,3*t+2]

				#for each handle, figure out the weight on other elements
				diag_x = np.kron(sWx[3*np.arange(len(sWx)/3)], np.array([1,0,1,0,1,0]))
				diag_y = np.kron(sWy[3*np.arange(len(sWy)/3)+1], np.array([0,1,0,1,0,1]))
				diag_o = np.kron(sWo[3*np.arange(len(sWo)/3)+2], np.array([1,0,1,0,1,0]))
				UtPAxDS[:,3*t+0] += np.multiply(diag_x, UtPAx)
				UtPAxDS[:,3*t+1] += np.multiply(diag_y, UtPAx)
				UtPAxDS[1:,3*t+2] += np.multiply(diag_o[:-1], UtPAx[:-1]) 
				UtPAxDS[:-1, 3*t+2] += np.multiply(diag_o[:-1], UtPAx[1:])
		self.constErs_Terms.append(UtPAxDS)

	def setupConstEgsTerms(self):
		wr_cols = []
		for i in range(len(self.mesh.red_r)):
			ce_map = np.array(self.mesh.r_cluster_element_map[i])
			wr_c1 = np.zeros(2*len(self.mesh.T))
			wr_c2 = np.zeros(2*len(self.mesh.T))
			wr_c1[2*ce_map] = 1
			wr_c2[2*ce_map+1] = 1
			wr_cols.append(wr_c1)
			wr_cols.append(wr_c2)
		Wr = np.vstack(wr_cols).T

		UU = sparse.lil_matrix((2*len(self.mesh.T), 2*len(self.mesh.T)))
		zsize = len(self.mesh.z)
		for t in range(len(self.mesh.T)):
			u1, u2 = np.cos(self.mesh.u[t]), np.sin(self.mesh.u[t])

			UU[2*t, 2*t]    = u1
			UU[2*t, 2*t+1]  = -u2
			UU[2*t+1, 2*t]  = -u2
			UU[2*t+1, 2*t+1]= -u1 

		UWr = UU.dot(Wr)
		repeat3 = sparse.kron(sparse.eye(len(self.mesh.T)), np.array([[1,0],[0,1], [1,0],[0,1], [1,0],[0,1]]))
		repeatUWr = repeat3.dot(UWr)
		
		#This is the tricky part
		if sparse.issparse(self.PAG):
			QAP = -1*self.PAG.T.toarray()
		else:
			QAP = -1*self.PAG.T
	
		# PAGtRU = -self.PAG.T.dot(self.mesh.GR.dot(self.mesh.GU))
		# print("PAG")
		# print(self.mesh.GU)
		# print(self.mesh.u)
		# # print(UWr.dot(c_vec))
		# exit()
	
		_6T_ = 6*len(self.mesh.T)
		for m in range(0,len(self.mesh.z)):
			evens = QAP[m, 2*np.arange(_6T_/2)] #first, third, fifth, etc...
			diag = np.kron(evens, np.array([1,1]))

			odds = QAP[m, 2*np.arange(_6T_/2)+1]
			off_diag = np.kron(odds, np.array([1,0]))
			
			PAGm = sparse.diags([off_diag[:-1], diag, -1*off_diag[:-1]], [-1,0,1])
	
			PAGmUWrUtPAxdS = (PAGm.dot(repeatUWr)).T.dot(self.constErs_Terms[0])
			self.constEgsTerms.append(PAGmUWrUtPAxdS)
	
	def setupConstEsTerms(self):
		wr_cols = []
		for i in range(len(self.mesh.red_r)):
			ce_map = np.array(self.mesh.r_cluster_element_map[i])
			wr_c1 = np.zeros(2*len(self.mesh.T))
			wr_c2 = np.zeros(2*len(self.mesh.T))
			wr_c1[2*ce_map] = 1
			wr_c2[2*ce_map+1] = 1
			wr_cols.append(wr_c1)
			wr_cols.append(wr_c2)
		Wr = np.vstack(wr_cols).T

		UtPAx = self.mesh.GU.T.dot(self.PAx)
		MUtPAx = sparse.lil_matrix((6*len(self.mesh.T), 3*len(self.mesh.T)))
		for t in range(len(self.mesh.T)):
			x = UtPAx[6*t:6*t+6]

			MUtPAx[6*t+0, 3*t+0] = x[0]
			MUtPAx[6*t+0, 3*t+2] = x[1]
			MUtPAx[6*t+1, 3*t+1] = x[1]
			MUtPAx[6*t+1, 3*t+2] = x[0]

			MUtPAx[6*t+2, 3*t+0] = x[2]
			MUtPAx[6*t+2, 3*t+2] = x[3]
			MUtPAx[6*t+3, 3*t+1] = x[3]
			MUtPAx[6*t+3, 3*t+2] = x[2]
			
			MUtPAx[6*t+4, 3*t+0] = x[4]
			MUtPAx[6*t+4, 3*t+2] = x[5]
			MUtPAx[6*t+5, 3*t+1] = x[5]
			MUtPAx[6*t+5, 3*t+2] = x[4]
		MUtPAx.tocsc()
		MUtPAxsW = MUtPAx.dot(self.mesh.sW)
		DSUtPAxMUtPAxsW = MUtPAxsW.T.dot(self.constErs_Terms[0])
		self.constEsTerms.append(DSUtPAxMUtPAxsW)
		

		# print(MUtPAxsW.dot(self.mesh.red_s) - self.mesh.GS.dot(UtPAx))
		# DS = self.sparseDSds()
		# out = self.sparseOuterProdDiags(self.mesh.GS.dot(UtPAx), UtPAx)
		# print(DS[0].multiply(out).sum())
		# exit()

		U_PAx = sparse.lil_matrix((6*len(self.mesh.T), 2*len(self.mesh.T)))
		for t in range(len(self.mesh.T)):
			u1, u2 = np.cos(self.mesh.u[t]), np.sin(self.mesh.u[t])
			p = self.PAx[6*t:6*t+6]
			U_PAx[6*t+0, 2*t+0] = p[0]*u1 + p[1]*u2
			U_PAx[6*t+0, 2*t+1] = p[1]*u1 - p[0]*u2
			U_PAx[6*t+1, 2*t+0] = p[1]*u1 - p[0]*u2
			U_PAx[6*t+1, 2*t+1] = -p[0]*u1 - p[1]*u2

			U_PAx[6*t+2, 2*t+0] = p[2]*u1 + p[3]*u2
			U_PAx[6*t+2, 2*t+1] = p[3]*u1 - p[2]*u2
			U_PAx[6*t+3, 2*t+0] = p[3]*u1 - p[2]*u2
			U_PAx[6*t+3, 2*t+1] = -p[2]*u1 - p[3]*u2

			U_PAx[6*t+4, 2*t+0] = p[4]*u1 + p[5]*u2
			U_PAx[6*t+4, 2*t+1] = p[5]*u1 - p[4]*u2
			U_PAx[6*t+5, 2*t+0] = p[5]*u1 - p[4]*u2
			U_PAx[6*t+5, 2*t+1] = -p[4]*u1 - p[5]*u2

		self.constEsTerms.append((U_PAx.dot(Wr)).T.dot(self.constErs_Terms[0]))

		U_PAG = []
		
		for i in range(len(self.mesh.z)):
			U_PAG_i = sparse.lil_matrix((6*len(self.mesh.T), 2*len(self.mesh.T)))
			for t in range(len(self.mesh.T)):
				u1, u2 = np.cos(self.mesh.u[t]), np.sin(self.mesh.u[t])
				p = self.PAG[6*t:6*t+6, i]
				U_PAG_i[6*t+0, 2*t+0] = p[0]*u1 + p[1]*u2
				U_PAG_i[6*t+0, 2*t+1] = p[1]*u1 - p[0]*u2
				U_PAG_i[6*t+1, 2*t+0] = p[1]*u1 - p[0]*u2
				U_PAG_i[6*t+1, 2*t+1] = -p[0]*u1 - p[1]*u2

				U_PAG_i[6*t+2, 2*t+0] = p[2]*u1 + p[3]*u2
				U_PAG_i[6*t+2, 2*t+1] = p[3]*u1 - p[2]*u2
				U_PAG_i[6*t+3, 2*t+0] = p[3]*u1 - p[2]*u2
				U_PAG_i[6*t+3, 2*t+1] = -p[2]*u1 - p[3]*u2

				U_PAG_i[6*t+4, 2*t+0] = p[4]*u1 + p[5]*u2
				U_PAG_i[6*t+4, 2*t+1] = p[5]*u1 - p[4]*u2
				U_PAG_i[6*t+5, 2*t+0] = p[5]*u1 - p[4]*u2
				U_PAG_i[6*t+5, 2*t+1] = -p[4]*u1 - p[5]*u2
			U_PAG_i.tocsc()
			U_PAG.append((U_PAG_i.dot(Wr)).T.dot(self.constErs_Terms[0]))


		self.constEsTerms.append(U_PAG)

	def setupConstUSUtPAxTerms(self):
		UtPAx = self.mesh.GU.T.dot(self.PAx)

		wr_cols = []
		for i in range(len(self.mesh.red_r)):
			ce_map = np.array(self.mesh.r_cluster_element_map[i])
			wr_c1 = np.zeros(2*len(self.mesh.T))
			wr_c2 = np.zeros(2*len(self.mesh.T))
			wr_c1[2*ce_map] = 1
			wr_c2[2*ce_map+1] = 1
			wr_cols.append(wr_c1)
			wr_cols.append(wr_c2)
		Wr = np.vstack(wr_cols).T

		UUtPAx_6tx3 = np.zeros((6*len(self.mesh.T), 3))
		for t in range(len(self.mesh.T)):
			u1, u2 = np.cos(self.mesh.u[t]), np.sin(self.mesh.u[t])
			x = UtPAx[6*t:6*t+6]

			UUtPAx_6tx3[6*t+0, 0] = u1*x[0]
			UUtPAx_6tx3[6*t+0, 1] = -u2*x[1]
			UUtPAx_6tx3[6*t+0, 2] = u1*x[1] - u2*x[0]
			UUtPAx_6tx3[6*t+1, 0] = u2*x[0]
			UUtPAx_6tx3[6*t+1, 1] = u1*x[1]
			UUtPAx_6tx3[6*t+1, 2] = u1*x[0] + u2*x[1]

			UUtPAx_6tx3[6*t+2, 0] = u1*x[2]
			UUtPAx_6tx3[6*t+2, 1] = -u2*x[3]
			UUtPAx_6tx3[6*t+2, 2] = u1*x[3] - u2*x[2]
			UUtPAx_6tx3[6*t+3, 0] = u2*x[2]
			UUtPAx_6tx3[6*t+3, 1] = u1*x[3]
			UUtPAx_6tx3[6*t+3, 2] = u1*x[2] + u2*x[3]

			UUtPAx_6tx3[6*t+4, 0] = u1*x[4]
			UUtPAx_6tx3[6*t+4, 1] = -u2*x[5]
			UUtPAx_6tx3[6*t+4, 2] = u1*x[5] - u2*x[4]
			UUtPAx_6tx3[6*t+5, 0] = u2*x[4]
			UUtPAx_6tx3[6*t+5, 1] = u1*x[5]
			UUtPAx_6tx3[6*t+5, 2] = u1*x[4] + u2*x[5]

		UUtPAxWr = []
		transform_matrix = np.kron(np.eye(3), np.array([[0,-1],[1,0]]))
		for i in range(3):
			P_matrix_i = sparse.lil_matrix((6*len(self.mesh.T), 2*len(self.mesh.T)))
			for t in range(len(self.mesh.T)):

				p123456 = UUtPAx_6tx3[6*t:6*t+6, i]
				p_21_43_65 = transform_matrix.dot(p123456)
				P_matrix_i[6*t:6*t+6, 2*t+0] = p123456[:,np.newaxis]
				P_matrix_i[6*t:6*t+6, 2*t+1] = p_21_43_65[:,np.newaxis]
			P_matrix_i.tocsc()
			UUtPAxWr.append(P_matrix_i.dot(Wr))
		

		S123_matrices = []
		S_1 = sparse.kron(sparse.eye(len(self.mesh.T)), np.kron(np.ones((6,1)), np.array([1,0,0])))
		S_2 = sparse.kron(sparse.eye(len(self.mesh.T)), np.kron(np.ones((6,1)), np.array([0,1,0])))
		S_3 = sparse.kron(sparse.eye(len(self.mesh.T)), np.kron(np.ones((6,1)), np.array([0,0,1])))
		S123_matrices.append(S_1)
		S123_matrices.append(S_2)
		S123_matrices.append(S_3)

		self.constUUtPaxTerms.append(UUtPAxWr)
		self.constUUtPaxTerms.append(S123_matrices)
		self.constUUtPaxTerms.append(None)
		self.updateConstUSUtPAx()
		
	def setupConstItRTerms(self):
		
		UtPAx = self.mesh.GU.T.dot(self.PAx)
		MUtPAx = sparse.lil_matrix((6*len(self.mesh.T), 3*len(self.mesh.T)))
		for t in range(len(self.mesh.T)):
			x = UtPAx[6*t:6*t+6]
			u1, u2 = np.cos(self.mesh.u[t]), np.sin(self.mesh.u[t])

			MUtPAx[6*t+0, 3*t+0] = x[0]*u1
			MUtPAx[6*t+0, 3*t+1] = -x[1]*u2
			MUtPAx[6*t+0, 3*t+2] = x[1]*u1 - x[0]*u2
			MUtPAx[6*t+1, 3*t+0] = x[0]*u2
			MUtPAx[6*t+1, 3*t+1] = x[1]*u1
			MUtPAx[6*t+1, 3*t+2] = x[0]*u1 + x[1]*u2

			MUtPAx[6*t+2, 3*t+0] = x[2]*u1
			MUtPAx[6*t+2, 3*t+1] = -x[3]*u2
			MUtPAx[6*t+2, 3*t+2] = x[3]*u1 - x[2]*u2
			MUtPAx[6*t+3, 3*t+0] = x[2]*u2
			MUtPAx[6*t+3, 3*t+1] = x[3]*u1
			MUtPAx[6*t+3, 3*t+2] = x[2]*u1 + x[3]*u2
			
			MUtPAx[6*t+4, 3*t+0] = x[4]*u1
			MUtPAx[6*t+4, 3*t+1] = -x[5]*u2
			MUtPAx[6*t+4, 3*t+2] = x[5]*u1 - x[4]*u2
			MUtPAx[6*t+5, 3*t+0] = x[4]*u2
			MUtPAx[6*t+5, 3*t+1] = x[5]*u1
			MUtPAx[6*t+5, 3*t+2] = x[4]*u1 + x[5]*u2

		MUtPAx.tocsc()
		MUtPAxsW = MUtPAx.dot(self.mesh.sW)
		
		for i in range(len(self.mesh.red_r)):
			B = self.mesh.RotationBLOCK[i]
			self.constItRTerms.append(B.T.dot(MUtPAxsW))

	def updateConstUSUtPAx(self):
		s = self.mesh.sW.dot(self.mesh.red_s)
		s0 = self.constUUtPaxTerms[1][0].dot(s)
		s1 = self.constUUtPaxTerms[1][1].dot(s)
		s2 = self.constUUtPaxTerms[1][2].dot(s)

		USUtPAxWr = np.multiply(self.constUUtPaxTerms[0][0], s0[:,np.newaxis])
		USUtPAxWr += np.multiply(self.constUUtPaxTerms[0][1], s1[:,np.newaxis])
		USUtPAxWr += np.multiply(self.constUUtPaxTerms[0][2], s2[:,np.newaxis])
		self.constUUtPaxTerms[2] = USUtPAxWr

	def constTimeFPAx(self):
		c_vec = []
		for i in range(len(self.mesh.red_r)):
			c1, c2 = np.cos(self.mesh.red_r[i]), np.sin(self.mesh.red_r[i])
			c_vec.append(c1)
			c_vec.append(c2)
		c = np.array(c_vec)
		FPAx = self.constUUtPaxTerms[2].dot(c)
		return FPAx
	
	def constTimeEs(self):
		c_vec = []
		for i in range(len(self.mesh.red_r)):
			c1, c2 = np.cos(self.mesh.red_r[i]), np.sin(self.mesh.red_r[i])
			c_vec.append(c1)
			c_vec.append(c2)
		c = np.array(c_vec)

		ans = self.constEsTerms[1].T.dot(c)
		for i in range(len(self.mesh.z)):
			ans += self.constEsTerms[2][i].T.dot(c)*self.mesh.z[i]

		t1 =  self.constEsTerms[0].T.dot(self.mesh.red_s)
		return t1- ans

	def constTimeEgs(self):
		c_vec = []
		for i in range(len(self.mesh.red_r)):
			c1, c2 = np.cos(self.mesh.red_r[i]), np.sin(self.mesh.red_r[i])
			c_vec.append(c1)
			c_vec.append(c2)
		c = np.array(c_vec)

		aa = datetime.datetime.now()
		Egs = np.zeros((len(self.mesh.z), len(self.mesh.red_s)))
		bb = datetime.datetime.now()
		for i in range(len(self.mesh.z)):
			UtRtPAGi = self.constEgsTerms[i].T.dot(c_vec)
			Egs[i,:] = UtRtPAGi

		cc = datetime.datetime.now()
		# print("TIME, ", (bb-aa).microseconds, (cc-bb).microseconds)
		return Egs

	def constTimeErs_mid(self):
		Ers_mid = np.zeros(self.Ers_mid.shape)
		for i in range(Ers_mid.shape[1]):
			cUPAG = self.constErs_midTerms[i]
			odd_spR = cUPAG[0].dot(self.mesh.z) + cUPAG[2]
			even_spR = cUPAG[1].dot(self.mesh.z) + cUPAG[3]
			odd_i = odd_spR.reshape((odd_spR.shape[0]/2, 2)).sum(axis=1)
			even_i = even_spR.reshape((even_spR.shape[0]/2, 2)).sum(axis=1)

			col_i = np.zeros(Ers_mid.shape[0])
			col_i[2*np.arange(Ers_mid.shape[0]/2)] = odd_i
			col_i[2*np.arange(Ers_mid.shape[0]/2)+1] = even_i
			Ers_mid[:, i] = col_i

		return Ers_mid
	
	def constTimeErs_second(self, mid):
		Ers = mid.T.dot(self.constErs_Terms[0])
		return Ers

	def constTimeErr(self):
		rDDR = self.redSparseDDRdrdr()
		Err = np.zeros((len(self.mesh.red_r), len(self.mesh.red_r)))
		for i in range(len(self.mesh.red_r)):
			BUSUtPAx0 = self.constErTerms[i][0].dot(self.mesh.red_s)
			BPAGz = self.constErTerms[i][1].dot(self.mesh.z)
			BPAx0 = self.constErTerms[i][2]
			res = BUSUtPAx0.T.dot(rDDR[i].dot(BPAGz+BPAx0))
			Err[i,i] = res
		return Err	

	def constTimeErz(self):
		Erz = np.zeros((len(self.mesh.z), len(self.mesh.red_r)))

		rDR = self.redSparseDRdr()
		for i in range(len(self.mesh.red_r)):
			BUSUtPAx0 = self.constErTerms[i][0].dot(self.mesh.red_s)
			BPAG = self.constErTerms[i][1]
			res = BPAG.T.dot(rDR[i][0].dot(BUSUtPAx0))
			Erz[:,i] = res

		return Erz

	def constTimeEr(self):
		dEdr = np.zeros(len(self.mesh.red_r))
		rDR = self.redSparseDRdr()
		for i in range(len(dEdr)):
			BUSUtPAx0 = self.constErTerms[i][0].dot(self.mesh.red_s)
			BPAGz = self.constErTerms[i][1].dot(self.mesh.z)
			BPAx0 = self.constErTerms[i][2]
			res = BUSUtPAx0.T.dot(rDR[i][0].dot(BPAGz+BPAx0))
			dEdr[i] = res

		return dEdr

	def Gradients(self):
		PAg = self.PA.dot(self.mesh.getg())
		# USUt = self.mesh.GU.dot(self.mesh.GS.dot(self.mesh.GU.T))
		# DR = self.sparseDRdr()
		# DS = self.sparseDSds()

		# dEdR = -1*self.sparseOuterProdDiags(PAg, USUt.dot(self.PAx))
		# dEdr = np.zeros(len(self.mesh.red_r))
		# for i in range(len(dEdr)):
		# 	dEdr[i] = DR[i].multiply(dEdR).sum()
		dEdr = self.constTimeEr()

		FPAx = self.constTimeFPAx() #self.mesh.GF.dot(self.PAx)
		dEdg = self.PAG.T.dot(PAg - FPAx)

		# UtPAx = self.mesh.GU.T.dot(self.PAx)
		# RU = self.mesh.GR.dot(self.mesh.GU)
		# dEdS = self.sparseOuterProdDiags(self.mesh.GS.dot(UtPAx), UtPAx) -self.sparseOuterProdDiags(RU.T.dot(PAg), UtPAx)
		# dEds = np.zeros(len(self.mesh.red_s))
		# for i in range(len(dEds)):
		# 	dEds[i] = DS[i].multiply(dEdS).sum()
		
		# print("deds")
		# print(dEds)
		dEds =self.constTimeEs()
		# print(dEds)
		# exit()

		return dEdg, dEdr, dEds

	def redSparseDRdr(self):
		DR = []

		for t in range(len(self.mesh.red_r)):
			c, s = np.cos(self.mesh.red_r[t]), np.sin(self.mesh.red_r[t])
			dRdr_e = np.array(((-s,-c), (c, -s)))
			blocked = sparse.kron(sparse.eye(3*len(self.mesh.r_cluster_element_map[t])), dRdr_e)
			DR.append((blocked, dRdr_e))

		return DR

	def sparseDRdr(self):
		DR = []

		for t in range(len(self.mesh.red_r)):
			c, s = np.cos(self.mesh.red_r[t]), np.sin(self.mesh.red_r[t])
			dRdr_e = np.array(((-s,-c), (c, -s)))
			blocked = sparse.kron(sparse.eye(3*len(self.mesh.r_cluster_element_map[t])), dRdr_e)

			B = self.mesh.RotationBLOCK[t]

			placeHolder = B.dot(blocked.dot(B.T))
			DR.append(placeHolder)

		return DR

	def sparseDDRdrdr_diag(self):
		diagD2R = []

		for t in range(len(self.mesh.red_r)):
			B = self.mesh.RotationBLOCK[t]

			c, s = np.cos(self.mesh.red_r[t]), np.sin(self.mesh.red_r[t])
			ddR_e = np.array(((-c,s), (-s, -c)))
			blocked = sparse.kron(sparse.eye(3*len(self.mesh.r_cluster_element_map[t])), ddR_e)
			placeHolder = B.dot(blocked.dot(B.T))

			diagD2R.append(placeHolder)
		return diagD2R

	def redSparseDDRdrdr(self):
		diagD2R = []

		for t in range(len(self.mesh.red_r)):
			B = self.mesh.RotationBLOCK[t]
			c, s = np.cos(self.mesh.red_r[t]), np.sin(self.mesh.red_r[t])
			ddR_e = np.array(((-c,s), (-s, -c)))
			blocked = sparse.kron(sparse.eye(3*len(self.mesh.r_cluster_element_map[t])), ddR_e)

			diagD2R.append(blocked)
		return diagD2R

	def sparseDSds(self):
		if self.DSDs == None:
			DS = []
			s = self.mesh.sW.dot(self.mesh.red_s)

			#iterate through all handles
			for t in range(len(self.mesh.red_s)/3):
				sWx = self.mesh.sW[:,3*t]
				sWy = self.mesh.sW[:,3*t+1]
				sWo = self.mesh.sW[:,3*t+2]


				#for each handle, figure out the weight on other elements
				diag_x = np.kron(sWx[3*np.arange(len(sWx)/3)], np.array([1,0,1,0,1,0]))
				diag_y = np.kron(sWy[3*np.arange(len(sWy)/3)+1], np.array([0,1,0,1,0,1]))
				diag_o = np.kron(sWo[3*np.arange(len(sWo)/3)+2], np.array([1,0,1,0,1,0]))

				gdSdsx = sparse.diags(diag_x,0).tocsc()
				gdSdsy = sparse.diags(diag_y,0).tocsc()
				gdSdso = sparse.diags([diag_o[:-1], diag_o[:-1]], [-1,1]).tocsc()

				DS.append(gdSdsx)
				DS.append(gdSdsy)
				DS.append(gdSdso)

			self.DSDs = DS

		return self.DSDs

	def dEdr(self):
		g, r, s = self.Gradients()
		return None, r

	def dEdg(self):
		PAg = self.PA.dot(self.mesh.getg())
		FPAx = self.constTimeFPAx() #self.mesh.GF.dot(self.PAx)
		res =  self.PAG.T.dot(PAg - FPAx)
		return res

	def dEds(self):
		g, r, s = self.Gradients()
		return None, s

	def Hess_Egg(self, block =False):
		return self.Hessians()[0]

	def Hess_Erg(self, block=False):
		return self.Hessians()[1]

	def Hess_Err(self, block=False):
		return self.Hessians()[2]

	def Hess_Egs(self, block=False):
		return self.Hessians()[3]

	def Hess_Ers(self, block=False):
		return self.Hessians()[4]

	def itR(self):
		PA_Gzx = self.PA.dot(self.mesh.getg())
		for i in range(len(self.mesh.red_r)):
			one = self.mesh.RotationBLOCK[i].T.dot(PA_Gzx)
			two = self.USUtPAx_E[i]

			m_PAg = np.reshape(one, (len(one)/2,2))
			m_USUPAx = np.reshape(two, (len(two)/2,2))
			F = np.matmul(m_PAg.T,m_USUPAx)
			u, s, vt = np.linalg.svd(F, full_matrices=True)
			R = np.matmul(vt.T, u.T)

			veca = np.array([1,0])
			vecb = np.dot(R, veca)

			#check angle sign
			theta = np.arccos(np.dot(veca,vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))
			c, s = np.cos(theta), np.sin(theta)
			R_check = np.array(((c,-s), (s, c)))
			if(np.linalg.norm(R-R_check)<1e-8):
				theta *=-1

			self.mesh.red_r[i] = theta

		return 1

	def itT(self):
		FPAx = self.constTimeFPAx() #self.mesh.GF.dot(self.PAx)
		
		deltaAbtg = self.ANTI_BLOCK.T.dot(self.mesh.g)
		GtAtPtFPAx = self.PAG.T.dot(FPAx)
		GtAtPtPAx = self.PAG.T.dot(self.PA.dot(self.mesh.x0))
		gb =  GtAtPtFPAx - GtAtPtPAx

		gd = np.concatenate((gb, deltaAbtg))
		gu = self.CholFac.solve(gd)
		self.mesh.z = gu[0:len(gb)]


		# else:
		# 	Abtg = self.ANTI_BLOCK.T.dot(self.mesh.getg() - self.mesh.x0)
		# 	AtPtFPAx = self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx))
		# 	AtPtPAx0 = self.Egg.dot(self.mesh.x0)
		# 	gb = AtPtFPAx - AtPtPAx0
		# 	gd = np.concatenate((gb, Abtg))
		# 	gu = self.CholFac.solve(gd)
		# 	self.mesh.g = gu[0:AtPtFPAx.size]

		return 1

	def iterate(self, its=100):
		print("	+ARAP iterate")
		# self.USUtPAx_E = []
		# for i in range(len(self.mesh.red_r)):
		# 	B = self.mesh.RotationBLOCK[i]
		# 	PAx_e = B.T.dot(self.PAx)
		# 	Ue = B.T.dot(self.mesh.GU.dot(B))
		# 	Se = B.T.dot(self.mesh.GS.dot(B))
		# 	USUPAx = Ue.dot(Se.dot(Ue.T.dot(PAx_e.T)))

		# 	self.USUtPAx_E.append(USUPAx)


		self.USUtPAx_E = []
		for i in range(len(self.mesh.red_r)):
			USUPAx = self.constItRTerms[i].dot(self.mesh.red_s)
			self.USUtPAx_E.append(USUPAx)

			
		Eg0 = self.dEdg()
		for i in range(its):
			g = self.itT()
			r = self.itR()
			self.mesh.getGlobalF(updateR=True, updateS=False)
			Eg = self.dEdg()

			if(1e-7 > np.linalg.norm(Eg-Eg0)):
				# print("En",self.Energy(), self.mesh.red_r, self.mesh.z)
				# self.mesh.getGlobalF(updateR=True, updateS=False, updateU=False)
				print("	-ARAP iterate "+str(i))
				# print("ARAP converged", np.linalg.norm(Eg))
				return
			Eg0 = Eg