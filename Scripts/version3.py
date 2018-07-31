#running the code version 3
import Meshwork
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
V = igl.eigen.MatrixXd()
F = igl.eigen.MatrixXi()
igl.readOFF("./MeshSetups/3koval/3koval.off", V, F)
V1 = e2p(V)
T = Delaunay(V1[:,:2]).simplices
# print(e2p(F)[:10,:])
# print(T[:10,:])
# exit()
VTU = [np.array(V1[:,:2]*100.0), T]
mw = Meshwork.Preprocessing(_VT = VTU, modes_used=50)
mw.Fix = Meshwork.get_max(mw.V, a=0, eps=1e-3)
# for v in range(len(mw.V)):
# 	if mw.V[v][0] > 0 and mw.V[v][1]<1e-5 and mw.V[v][1]>-1e-5:
# 		mw.Fix.append(v)
mw.Mov = Meshwork.get_min(mw.V, a=0, eps=1e-3)
# mw.Mov = mw.Mov + Meshwork.get_min(mw.V, a=1, eps=1e-3)
# mw.Mov = mw.Mov + Meshwork.get_max(mw.V, a=1, eps=1e-3)
mw.display()
exit()



#Read-in mesh: read V, T, U, rot-clusters, skinning handles, Modes (if reduced)
# mw = Meshwork.Preprocessing(_VT = None)
# mesh = mw.getMesh(name="woman", modes_used=15)
mw = Meshwork.Preprocessing()
mesh = mw.getMesh(name= "3koval/Unipennate", modes_used=10)

#ARAP setup
arap = Arap.ARAP(imesh=mesh, filen="snapshots/")

#Elasticity setup
neo = Neo.NeohookeanElastic(imesh = mesh)

#Solver setup
ti = Solvers.TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neo)

#Running
disp = Display.Display(isolve = ti)
disp.headless()
# disp.display()

#Save mesh
name = "./MeshSetups/3koval/Unipennate/moreDOFS_V.dmat"
# g = mesh.getg()
# igl.writeDMAT(name, igl.eigen.MatrixXd(np.array([g])), True)


#Compare mesh
g = mesh.getg()
eg = igl.eigen.MatrixXd()
igl.readDMAT(name, eg)
g2 = e2p(eg)[0,:]
print(np.linalg.norm(g-g2))

