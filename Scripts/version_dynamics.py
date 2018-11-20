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
# VTU = Meshwork.rectangle_mesh(x=10, y=10, step=1.0)
# mw = Meshwork.Preprocessing(_VT = VTU)
# mw.Fix = Meshwork.get_max(mw.V, a=1, eps=1e-2)	
# mw.Mov = Meshwork.get_min(mw.V, a=1, eps=1e-2)
# mesh = mw.getMesh(modes_used=20)
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

V = igl.eigen.MatrixXd()
F = igl.eigen.MatrixXi()
igl.readOFF("./MeshSetups/3koval/3koval.off", V, F)
V1 = e2p(V)[:,:]
F1 = e2p(F)
VTU = {"V": np.array(V1[:,:2]*100.0), "T": F1}
mw = Meshwork.Preprocessing(_VT = VTU, modes_used=50)
mw.Fix = Meshwork.get_max(mw.V, a=0, eps=2e-1)
# for v in range(len(mw.V)):
# 	if mw.V[v][0] > 6 and mw.V[v][1]<1e-1 and mw.V[v][1]>-1e-1:
# 		mw.Fix.append(v)
mw.Mov = Meshwork.get_min(mw.V, a=0, eps=2e-1)
mesh = mw.getMesh(name= "3koval", modes_used=30)

# mw.Mov = mw.Mov + Meshwork.get_min(mw.V, a=1, eps=1e-3)
# mw.Mov = mw.Mov + Meshwork.get_max(mw.V, a=1, eps=1e-3)

#ARAP setup
arap = Arap.ARAP(imesh=mesh, filen="snapshots/")

#Elasticity setup
neo = Neo.NeohookeanElastic(imesh = mesh)

#Solver setup
ti = Solvers.TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neo)

# #Running
disp = Display.Display(isolve = ti)


disp.display_statics()

