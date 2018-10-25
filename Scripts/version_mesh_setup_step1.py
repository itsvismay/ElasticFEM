#running the code version 3
import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
from iglhelpers import *
from scipy.spatial import Delaunay
from collections import defaultdict
import scipy
from scipy import sparse
from scipy.sparse import linalg
from scipy.cluster.vq import vq, kmeans, whiten
import Helpers
import random

FOLDER = "./MeshSetups/"+"TestArm/"
# os.mkdir(FOLDER)
# os.mkdir(FOLDER+"muscles")
# os.mkdir(FOLDER+"bones")
# os.mkdir(FOLDER+"muscle_bone")

print("Writing to folder: "+FOLDER)

mesh1 = {}
mesh2 = {}
mesh3 = {}
mesh4 = {}

# mesh1 = Helpers.rectangle_mesh(x=5, y=1, step=1.0, offset=(0,0))
# mesh2 = Helpers.rectangle_mesh(x=5, y=1, step=1.0, offset=(5,0))
# mesh3 = Helpers.torus_mesh(r1=2, r2=3, r3=5, step=1.0, offset=(5,1))
# print(mesh2["V"])
# print(mesh2["T"])
# print("")

ev1, et1 = igl.eigen.MatrixXd(), igl.eigen.MatrixXi()
igl.readOBJ(FOLDER+"1mesh.obj", ev1, et1)
mesh1["V"] = np.array(e2p(ev1)[:, :2])
mesh1["T"] = np.array(e2p(et1))
mesh1["u"] = np.zeros(len(mesh1["T"]))
ev2, et2 = igl.eigen.MatrixXd(), igl.eigen.MatrixXi()
igl.readOBJ(FOLDER+"2mesh.obj", ev2, et2)
mesh2["V"] = np.array(e2p(ev2)[:, :2])
mesh2["T"] = np.array(e2p(et2))
print(mesh2["V"])
print(mesh2["T"])
mesh2["u"] = np.zeros(len(mesh2["T"]))
ev4, et4 = igl.eigen.MatrixXd(), igl.eigen.MatrixXi()
igl.readOBJ(FOLDER+"4mesh.obj", ev4, et4)
mesh4["V"] = np.array(e2p(ev4)[:, :2])
mesh4["T"] = np.array(e2p(et4))
mesh4["u"] = np.zeros(len(mesh4["T"]))
ev3, et3 = igl.eigen.MatrixXd(), igl.eigen.MatrixXi()
igl.readOBJ(FOLDER+"3mesh.obj", ev3, et3)
mesh3["V"] = np.array(e2p(ev3)[:, :2])
mesh3["T"] = np.array(e2p(et3))
mesh3["u"] = np.zeros(len(mesh3["T"]))

mesh1["isMuscle"]= False
mesh1["Mov"] = []
mesh1["Fix"] = []
mesh1["nrc"] = 1
mesh1["nsh"] = 1
mesh2["isMuscle"]= False
mesh2["Mov"] = []
mesh2["Fix"] = []
mesh2["nrc"] = 1
mesh2["nsh"] = 1
mesh4["isMuscle"]= False
mesh4["Mov"] = []
mesh4["Fix"] = []
mesh4["nrc"] = 1
mesh4["nsh"] = 1
mesh3["isMuscle"]= True
mesh3["Mov"] = Helpers.get_max(mesh3["V"], a=1, eps=1e-2)
mesh3["Fix"] = Helpers.get_min(mesh3["V"], a=1, eps=1e-2)
mesh3["nrc"] = 2
mesh3["nsh"] = 2


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
	C = sparse.kron(sparse.eye(len(iT)), sparse.kron(np.ones((3,3))/3 , np.eye(2)))
	return C

def getP(iT):
	sub_P = np.kron(np.matrix([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]), np.eye(2))/3.0
	# sub_P = np.kron(np.matrix([[-1, 1, 0], [0, -1, 1], [1, 0, -1]]), np.eye(2))
	P = sparse.kron(sparse.eye(len(iT)), sub_P).tocsc()
	return P

def get_area(p1, p2, p3):
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))*0.5

def getVertexWiseMassDiags(iV, iT):
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

	return mass_diag

def getBlockingMatrices(iV, iFix):
	if(len(iFix) == len(iV)):
		BLOCK =  sparse.csc_matrix(np.array([[]]))
		ANTI_BLOCK = sparse.eye(2*len(iV)).tocsc()
		return BLOCK, ANTI_BLOCK

	if(len(iFix) == 0):
		BLOCK = sparse.eye(2*len(iV)).tocsc()
		ANTI_BLOCK= sparse.csc_matrix((2*len(iV), (2*len(iV))))
		return BLOCK, ANTI_BLOCK

	Id = sparse.eye(len(iV)).tocsc()
	anti_b = sparse.kron(Id[:,iFix], np.eye(2))
	ab = np.zeros(len(iV))
	ab[iFix] = 1
	notfix = [i for i in range(0, len(ab)) if ab[i] == 0]
	b = sparse.kron(Id[:,notfix], np.eye(2))
	BLOCK = b.tocsc()
	ANTI_BLOCK = anti_b.tocsc()
	return BLOCK, ANTI_BLOCK

def heat_method(iV, iT, iFix, iMov):
	t = 1e-1
	eLc = igl.eigen.SparseMatrixd()
	igl.cotmatrix(igl.eigen.MatrixXd(iV), igl.eigen.MatrixXi(iT), eLc)
	Lc = e2p(eLc)
	Mdiag = getVertexWiseMassDiags(iV, iT)[2*np.arange(Lc.shape[0])]
	Mc = sparse.diags(Mdiag)

	# #Au = b st. Cu = Cu0
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
	igl.grad(igl.eigen.MatrixXd(nV), igl.eigen.MatrixXi(iT), eG)
	eu = igl.eigen.MatrixXd(u)
	eGu = (eG*eu).MapMatrix(len(iT), 3)
	Gu = e2p(eGu)
	gradu = np.zeros(len(iT))
	for i in range(len(iT)):
		e = iT[i]
	
		uvec = Gu[i,0:2]
		veca = uvec
		vecb = np.array([1,0])
		# theta = np.arccos(np.dot(veca,vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))
		x1 = np.cross(veca, vecb).dot(np.array([0,0,1]))
		x2 = np.linalg.norm(veca)*np.linalg.norm(vecb) + veca.dot(vecb)
		theta = 2*np.arctan2(x1, x2)[2]
		# print(theta)
		gradu[i] = theta

	return gradu

def modal_analysis(mesh, num_modes=None):
	K = mesh["A"].T.dot(mesh["P"].T.dot(mesh["P"].dot(mesh["A"])))

	eig, ev = Helpers.general_eig_solve(A = K, B = mesh["M"], modes=num_modes)
	eig = eig[2:]
	ev = ev[:,2:]
	ev = np.divide(ev, eig*eig)
	ev = sparse.csc_matrix(ev)
	############handle modes KKT solve#####
	C = mesh["ANTI_BLOCK"].T
	col1 = sparse.vstack((K, C))
	col2 = sparse.vstack((C.T, sparse.csc_matrix((C.shape[0], C.shape[0]))))
	KKT = sparse.hstack((col1, col2))
	eHconstrains = sparse.vstack((sparse.csc_matrix((K.shape[0], C.shape[0])), sparse.eye(C.shape[0])))
	eH = sparse.linalg.spsolve(KKT.tocsc(), eHconstrains.tocsc())[0:K.shape[0]]
	###############QR get orth basis#######
	eHeV = sparse.hstack((eH, ev))
	Q1, QR1 = np.linalg.qr(eHeV.toarray(), mode="reduced")
	return Q1

def kmeans_clustering(mesh, clusters=3):
		A = mesh["A"]
		C = mesh["C"]
		G = np.add(mesh["G"].T, np.ravel(mesh["V"]))
		CAG = C.dot(A.dot(G.T))
		Data = np.zeros((len(mesh["T"]), 2*mesh["G"].shape[1]))

		for i in range(len(mesh["T"])):
			point = CAG[6*i:6*i+2, :]
			Data[i,:] = np.ravel(point) #triangle by x1,y1,x2,y2, x3,y3....

		print(clusters, Data.shape)
		centroids,_ = kmeans(Data, clusters)
		idx,_ = vq(Data,centroids)
		return idx

def rotation_clusters(mesh, nrc =1):
	if nrc==len(mesh["T"]):
		element_to_cluster = np.arange(nrc)
	else:
		element_to_cluster = kmeans_clustering(mesh, nrc)
	return element_to_cluster

def skinning_handles(mesh, nsh=1):
	e_c = kmeans_clustering(mesh, clusters = nsh)
	c_e = defaultdict(list)

	for i in range(len(mesh["T"])):
		c_e[e_c[i]].append(i)

	shandles = []
	CAx0 = mesh["C"].dot(mesh["A"].dot(np.ravel(mesh["V"])))
	for k in range(len(c_e.keys())):
		els = np.array(c_e[k], dtype='int32')
		centx = CAx0[6*els]
		centy = CAx0[6*els+1]
		avc = np.array([np.sum(centx)/len(els), np.sum(centy)/len(els)]) 
		minind = els[0]
		mindist = np.linalg.norm(avc-np.array([centx[0], centy[0]]))
		for i in range(1,len(els)):
			dist = np.linalg.norm(avc-np.array([centx[i], centy[i]]))
			if dist<=mindist:
				mindist = dist 
				minind = els[i]

		shandles.append(minind)
	return np.array(shandles)

def bbw_skinning_matrix(mesh, handles):
	vertex_handles = mesh["T"][handles]
	unique_vert_handles = np.unique(vertex_handles)
	helper = np.add(np.zeros(unique_vert_handles[-1]+1), -1)

	for i in range(len(unique_vert_handles)):
		helper[unique_vert_handles[i]] = i 

	vert_to_tet = np.zeros((len(handles), 3), dtype="int32")
	for i in range(vertex_handles.shape[0]):
		vert_to_tet[i,:] = helper[vertex_handles[i]]

	C = mesh["V"][unique_vert_handles]
	P = np.array([np.arange(len(C))], dtype="int32").T

	V = igl.eigen.MatrixXd(mesh["V"])
	T = igl.eigen.MatrixXi(mesh["T"])
	M = igl.eigen.MatrixXd()
	W = igl.eigen.MatrixXd()
	C = igl.eigen.MatrixXd(C)
	P = igl.eigen.MatrixXi(P)
	# List of boundary indices (aka fixed value indices into VV)
	b = igl.eigen.MatrixXi()
	# List of boundary conditions of each weight function
	bc = igl.eigen.MatrixXd()
	# print(unique_vert_handles)
	# print(mesh["V"][np.array([989, 1450, 1610])])
	# print(C)
	# exit()
	igl.boundary_conditions(V, T, C, P, igl.eigen.MatrixXi(), igl.eigen.MatrixXi(), b, bc)	

	
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
	
	vW = e2p(W) #v x verts of handles

	tW = np.zeros((len(mesh["T"]), len(handles))) #T x handles
	#get average of vertices for each triangle
	for i in range(len(mesh["T"])):
		e = mesh["T"][i]
		for h in range(len(handles)):
			if i== handles[h]:
				tW[i,:] *= 0
				tW[i,h] = 1

				break
			p0 = vW[e[0],vert_to_tet[h,:]].sum()
			p1 = vW[e[1],vert_to_tet[h,:]].sum()
			p2 = vW[e[2],vert_to_tet[h,:]].sum()
			tW[i,h] = (p0+p1+p2)/3.

	tW /= np.sum(tW, axis =1)[:, np.newaxis] #normalize rows to sum to 1
	return np.kron(tW, np.eye(3))

def setup_meshes(meshes):
	#For muscles:
	# - Fix points, Mov points
	# -> Heat flow (u)
	# -> Modal Analysis (G)
	# -> Kmeans Rotation Clusters ()
	# -> Skinning handles, BBW

	#For bones:
	# - One rotation cluster per bone
	# - One skinning handle per bone (for now, one total later)
	# - Skinning weights all 1
	for mesh in meshes:
		if mesh["isMuscle"]:
			# muscle
			mesh["C"] = getC(mesh["T"])
			mesh["A"] = getA(mesh["V"], mesh["T"])
			mesh["P"] = getP(mesh["T"])
			mesh["M"] = sparse.diags(getVertexWiseMassDiags(mesh["V"], mesh["T"]))
			mesh["BLOCK"], mesh["ANTI_BLOCK"] = getBlockingMatrices(mesh["V"], mesh["Fix"]) 
			mesh["u"] = np.ones(len(mesh["T"]))*np.pi/2#heat_method(mesh["V"], mesh["T"], mesh["Fix"], mesh["Mov"])
			mesh["G"] = modal_analysis(mesh)
			mesh["e_to_c"] = rotation_clusters(mesh, nrc=mesh["nrc"])
			mesh["shandles_ind"] = skinning_handles(mesh, nsh = mesh["nsh"])
			mesh["sW"] = bbw_skinning_matrix(mesh, handles = mesh["shandles_ind"])

		else:
			# bone
			mesh["C"] = getC(mesh["T"])
			mesh["A"] = getA(mesh["V"], mesh["T"])
			mesh["P"] = getP(mesh["T"])
			mesh["e_to_c"] = np.zeros(len(mesh["T"]), dtype='int32')
			mesh["shandles_ind"] = np.array([0])
			mesh["sW"] = np.kron(np.ones((len(mesh["T"]), 1)), np.eye(3))#bbw_skinning_matrix(mesh, handles = mesh["shandles_ind"])

def output_meshes(meshes):
	#Output all individually.
	#For muscles:
	# - mesh V, F
	# - Heat flow (u)
	# - Kmeans Rotation Clusters ()
	# - Skinning handles, BBW

	#For bones:
	# - One rotation cluster per bone
	# - One skinning handle per bone (for now, one total later)
	# - Skinning weights all 1
	muscle_count = 0
	bone_count = 0
	for mesh in meshes:
		if mesh["isMuscle"]:
			igl.writeDMAT(FOLDER+"muscles/"+str(muscle_count)+"V.dmat", igl.eigen.MatrixXd(np.array(mesh["V"])), True)
			igl.writeDMAT(FOLDER+"muscles/"+str(muscle_count)+"F.dmat", igl.eigen.MatrixXi(mesh["T"]), True)
			igl.writeOBJ(FOLDER+"muscles/"+str(muscle_count)+"muscle.obj",igl.eigen.MatrixXd(np.array(np.hstack((mesh["V"], np.zeros((len(mesh["V"]),1)))))),igl.eigen.MatrixXi(mesh["T"]))
			igl.writeDMAT(FOLDER+"muscles/"+str(muscle_count)+"u.dmat", igl.eigen.MatrixXd(np.array([mesh["u"]])), True)
			igl.writeDMAT(FOLDER+"muscles/"+str(muscle_count)+"e_to_c.dmat", igl.eigen.MatrixXi(mesh["e_to_c"]), True)
			igl.writeDMAT(FOLDER+"muscles/"+str(muscle_count)+"s_handles.dmat", igl.eigen.MatrixXi(np.array([mesh["shandles_ind"]], dtype='int32')), True)
			igl.writeDMAT(FOLDER+"muscles/"+str(muscle_count)+"skinning_weights.dmat", igl.eigen.MatrixXd(mesh["sW"]), True)
			muscle_count += 1

		else:
			igl.writeDMAT(FOLDER+"bones/"+str(bone_count)+"V.dmat", igl.eigen.MatrixXd(np.array(mesh["V"])), True)
			igl.writeDMAT(FOLDER+"bones/"+str(bone_count)+"F.dmat", igl.eigen.MatrixXi(mesh["T"]), True)
			igl.writeOBJ(FOLDER+"bones/"+str(bone_count)+"bone.obj",igl.eigen.MatrixXd(np.array(np.hstack((mesh["V"], np.zeros((len(mesh["V"]),1)))))),igl.eigen.MatrixXi(mesh["T"]))
			igl.writeDMAT(FOLDER+"bones/"+str(bone_count)+"u.dmat", igl.eigen.MatrixXd(np.array([mesh["u"]])), True)
			igl.writeDMAT(FOLDER+"bones/"+str(bone_count)+"e_to_c.dmat", igl.eigen.MatrixXi(mesh["e_to_c"]), True)
			igl.writeDMAT(FOLDER+"bones/"+str(bone_count)+"s_handles.dmat", igl.eigen.MatrixXi(np.array([mesh["shandles_ind"]], dtype='int32')), True)
			igl.writeDMAT(FOLDER+"bones/"+str(bone_count)+"skinning_weights.dmat", igl.eigen.MatrixXd(mesh["sW"]), True)
			bone_count += 1

	#Output combined
	# - [V1 V2, V3] -> remove duplicates in Meshlab
	# - [T1, T2, T3]
	# - [u1, u2, u3]
	# - [[s1.1, s1.2, 0, 0],[0, 0, 1, 0],[0,0,0,1]]

	V = meshes[0]["V"]
	T = meshes[0]["T"]
	for im in range(1,len(meshes)):
		T = np.vstack((T, np.add(meshes[im]["T"], len(V))))
		V = np.vstack((V, meshes[im]["V"]))
	V = np.hstack((V, np.zeros((len(V),1))))
	V2 = igl.eigen.MatrixXd(V)
	T2 = igl.eigen.MatrixXi(T)

	u = meshes[0]["u"]
	sW_blocks = [meshes[0]["sW"]]
	shandles = meshes[0]["shandles_ind"]
	v = meshes[0]["T"].shape[0]
	e_to_c = meshes[0]["e_to_c"]
	tot_clusters = meshes[0]["nrc"]
	elem_material = np.ones(meshes[0]["T"].shape[0])*meshes[0]["isMuscle"]
	shandle_for_muscle = np.ones(meshes[0]["nsh"])
	if not meshes[0]["isMuscle"]:
		shandle_for_muscle*=0

	for im in range(1, len(meshes)):
		u = np.concatenate((u, meshes[im]["u"]))
		sW_blocks.append(meshes[im]["sW"])
		shandles = np.concatenate((shandles, np.add(meshes[im]["shandles_ind"], v)))
		e_to_c = np.concatenate((e_to_c, np.add(meshes[im]["e_to_c"], tot_clusters)))
		elem_material = np.concatenate((elem_material, np.ones(meshes[im]["T"].shape[0])*meshes[im]["isMuscle"]))
		v += meshes[im]["T"].shape[0]
		tot_clusters += meshes[im]["nrc"]
		if meshes[im]["isMuscle"]:
			shandle_for_muscle = np.concatenate((shandle_for_muscle, np.ones(meshes[im]["nsh"])))
		else:
			shandle_for_muscle = np.concatenate((shandle_for_muscle, np.zeros(meshes[im]["nsh"])))
	
	sW = scipy.linalg.block_diag(*sW_blocks)

	igl.writeOBJ(FOLDER + "muscle_bone/" + "combined.obj",V2,T2)
	igl.writeDMAT(FOLDER + "muscle_bone/" + "u.dmat", igl.eigen.MatrixXd(np.array(u)), True)
	igl.writeDMAT(FOLDER + "muscle_bone/" + "shandles.dmat", igl.eigen.MatrixXi(np.array(shandles, dtype="int32")), True)
	igl.writeDMAT(FOLDER + "muscle_bone/" + "e_to_c.dmat", igl.eigen.MatrixXi(np.array(e_to_c, dtype="int32")), True)
	igl.writeDMAT(FOLDER + "muscle_bone/" + "sW.dmat", igl.eigen.MatrixXd(sW), True)
	igl.writeDMAT(FOLDER + "muscle_bone/" + "elem_material.dmat", igl.eigen.MatrixXd(elem_material), True)
	igl.writeDMAT(FOLDER + "muscle_bone/" + "is_shandle_for_muscle.dmat", igl.eigen.MatrixXd(shandle_for_muscle), True)
	return

def display_mesh(meshes):
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
		for im in range(len(meshes)):
			bc = igl.eigen.MatrixXd()
			fid = igl.eigen.MatrixXi(np.array([-1]))
			coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
			hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
			viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(meshes[im]["V"]), igl.eigen.MatrixXi(meshes[im]["T"]), fid, bc)
			ind = e2p(fid)[0][0]
			
			if hit and btn==0:
				# paint hit red
				meshes[im]["Fix"].append(meshes[im]["T"][ind][np.argmax(bc)])
				print("fix", im, ind, np.argmax(bc))
				return True

			if hit and btn==2:
				# paint hit red
				meshes[im]["Mov"].append(meshes[im]["T"][ind][np.argmax(bc)])
				print("mov", im, ind, np.argmax(bc))
				return True

		return False

	def key_down(viewer,aaa, bbb):
		viewer.data().clear()
		if aaa==48:
			for im in range(len(meshes)):
				for e in meshes[im]["T"]:
					P = meshes[im]["V"][e]
					DP = np.array([P[1], P[2], P[0]])
					viewer.data().add_edges(igl.eigen.MatrixXd(P), igl.eigen.MatrixXd(DP), purple)
		
		if aaa>48 and aaa<=48+len(meshes):
			im = aaa-49
			viewer.data().set_mesh(igl.eigen.MatrixXd(meshes[im]["V"]), igl.eigen.MatrixXi(meshes[im]["T"]))
			CAx0 = meshes[im]["C"].dot(meshes[im]["A"].dot(np.ravel(meshes[im]["V"])))
			for i in range(len(meshes[im]["T"])):
				c = np.matrix([CAx0[6*i:6*i+2],CAx0[6*i:6*i+2]])
				alpha = meshes[im]["u"][i]
				cU, sU = np.cos(alpha), np.sin(alpha)
				U = np.array(((cU,-sU), (sU, cU)))
				scaledU = np.multiply(U, np.array([[.1],[.1]])) + c
				viewer.data().add_edges(igl.eigen.MatrixXd(c[0,:]), igl.eigen.MatrixXd(scaledU[0,:]), black)

			Colors = np.ones(meshes[im]["T"].shape)
			for i in range(len(meshes[im]["T"])): 
				color = black
				Colors[i,:] = randc[meshes[im]["e_to_c"][i]]
			Colors[np.array([meshes[im]["shandles_ind"]]),:] = np.array([0,0,0])
			viewer.data().set_colors(igl.eigen.MatrixXd(np.array(Colors)))

		if aaa ==65:
			#run the setup code
			setup_meshes(meshes)

		if aaa==83:
			#output mesh
			output_meshes(meshes)

	def pre_draw(viewer):
		fixed_pts = []
		mov_pts = []
		for mesh in meshes:
			for i in range(len(mesh["Fix"])):
				fixed_pts.append(mesh["V"][mesh["Fix"][i]])
			for i in range(len(mesh["Mov"])):
				mov_pts.append(mesh["V"][mesh["Mov"][i]])
		
		viewer.data().add_points(igl.eigen.MatrixXd(np.array(fixed_pts)), red)
		viewer.data().add_points(igl.eigen.MatrixXd(np.array(mov_pts)), green)


		
	key_down(viewer, "b", 123)
	viewer.callback_mouse_down = mouse_down
	viewer.callback_key_down = key_down
	viewer.callback_pre_draw = pre_draw
	viewer.core.is_animating = False
	viewer.launch()


meshes = [mesh1, mesh2, mesh4, mesh3]
display_mesh(meshes)





