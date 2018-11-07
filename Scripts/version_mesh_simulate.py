#running the code version 3
import Meshwork
import Mesh
import Arap
import Display
import Solvers
import Neo
import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
from iglhelpers import *
from scipy.spatial import Delaunay


#Mesh creation
# #Read-in mesh: read V, T, U, rot-clusters, skinning handles, Modes (if reduced)
# VTU = Meshwork.rectangle_mesh(x=5, y=5, step=1.0)
# mw = Meshwork.Preprocessing(_VT = VTU)
# mw.Fix = Meshwork.get_max(mw.V, a=1, eps=1e-2)	
# mw.Mov = Meshwork.get_min(mw.V, a=1, eps=1e-2)
# mw.display()
# exit()

# V = igl.eigen.MatrixXd()
# F = igl.eigen.MatrixXi()
# igl.readOBJ("./MeshSetups/TestArm/muscle_bone/combined.obj", V, F)
# V1 = e2p(V)[:,:]
# F1 = e2p(F)
# VTU = {"V": np.array(V1[:,:2]), "T": F1}
# mw = Meshwork.Preprocessing(_VT = VTU, modes_used=None)
# mw.Fix = [10,1,0]
# mw.Mov = [25]
# mw.display()
# exit()

# mw = Meshwork.Preprocessing()
# mesh = mw.getMesh(name= "test2x2", modes_used=30)
# emat = igl.eigen.MatrixXd()
# igl.readDMAT("./MeshSetups/TestArm/muscle_bone/"+"elem_material.dmat", emat)
# elem_material = e2p(emat)[:, 0]
# mesh.elem_youngs = np.array([600000 if e<0.5 else 6e10 for e in elem_material])
# mesh.elem_poisson = np.array([0.45 if e<0.5 else 0.45 for e in elem_material])
# mesh.u_toggle = elem_material

FOLDER = "./MeshSetups/"+"TestArm/"
print("reading from: "+FOLDER)
eV = igl.eigen.MatrixXd()
eT = igl.eigen.MatrixXi()
eu = igl.eigen.MatrixXd()
es_ind = igl.eigen.MatrixXi()
er_ind = igl.eigen.MatrixXi()
esW = igl.eigen.MatrixXd()
emat = igl.eigen.MatrixXd()
eshandle_muscle = igl.eigen.MatrixXd()
igl.readOBJ(FOLDER+"muscle_bone/"+"combined.obj", eV, eT)
igl.readDMAT(FOLDER+"muscle_bone/"+"u.dmat", eu)
igl.readDMAT(FOLDER+"muscle_bone/"+"shandles.dmat", es_ind)
igl.readDMAT(FOLDER+"muscle_bone/"+"e_to_c.dmat", er_ind)
igl.readDMAT(FOLDER+"muscle_bone/"+"sW.dmat", esW)
igl.readDMAT(FOLDER+"muscle_bone/"+"elem_material.dmat", emat)
igl.readDMAT(FOLDER+"muscle_bone/"+"is_shandle_for_muscle.dmat", eshandle_muscle)
V = e2p(eV)[:,:2]
T = e2p(eT)
u = e2p(eu)[:, 0]
s_ind = e2p(es_ind)[:, 0]
r_ind = e2p(er_ind)[:, 0]
sW = e2p(esW)
elem_material = e2p(emat)[:, 0]
shandle_muscle = e2p(eshandle_muscle)[:,0]



mesh = Mesh.Mesh(read_in = True)
mesh.init_muscle_bone(V, T, u, s_ind, r_ind, sW, elem_material,[0,1], [], shandle_muscle, modes_used=None)

#ARAP setup
arap = Arap.ARAP(imesh=mesh, filen="snapshots/")

#Elasticity setup
neo = Neo.NeohookeanElastic(imesh = mesh)

#Solver setup
ti = Solvers.TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neo)

# #Running
disp = Display.Display(isolve = ti)


disp.display_dynamics()

