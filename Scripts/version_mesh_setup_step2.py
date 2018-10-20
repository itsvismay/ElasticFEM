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

FOLDER = "./MeshSetups/"+"TestArm/"
print("reading from: "+FOLDER)

eV = igl.eigen.MatrixXd()
eT = igl.eigen.MatrixXi()
eu = igl.eigen.MatrixXd()
es_ind = igl.eigen.MatrixXi()
er_ind = igl.eigen.MatrixXi()
esW = igl.eigen.MatrixXd()

igl.readOBJ(FOLDER+"muscle_bone/"+"combined.obj", eV, eT)
igl.readDMAT(FOLDER+"muscle_bone/"+"u.dmat", eu)
igl.readDMAT(FOLDER+"muscle_bone/"+"shandles.dmat", es_ind)
igl.readDMAT(FOLDER+"muscle_bone/"+"e_to_c.dmat", er_ind)
igl.readDMAT(FOLDER+"muscle_bone/"+"sW.dmat", esW)



V = e2p(eV)
T = e2p(eT)
u = e2p(eu)[:, 0]
s_ind = e2p(es_ind)[:, 0]
r_ind = e2p(er_ind)[:, 0]
sW = e2p(esW)









