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
# V = igl.eigen.MatrixXd()
# F = igl.eigen.MatrixXi()
# igl.readOFF("./MeshSetups/3koval/3koval.off", V, F)
# V1 = e2p(V)[:,:]
# # print(V1.shape)
# F1 = e2p(F)
# T = Delaunay(V1[:,:2], incremental=True).simplices[:-2,:]
# # print(e2p(F)[:10,:])
# # print(T[:10,:])
# # exit()
# VTU = [np.array(V1[:,:2]*100.0), F1]
# mw = Meshwork.Preprocessing(_VT = VTU, modes_used=50)
# mw.Fix = Meshwork.get_max(mw.V, a=0, eps=2e-1)
# # for v in range(len(mw.V)):
# # 	if mw.V[v][0] > 6 and mw.V[v][1]<1e-1 and mw.V[v][1]>-1e-1:
# # 		mw.Fix.append(v)
# mw.Mov = Meshwork.get_min(mw.V, a=0, eps=2e-1)

# # mw.Mov = mw.Mov + Meshwork.get_min(mw.V, a=1, eps=1e-3)
# # mw.Mov = mw.Mov + Meshwork.get_max(mw.V, a=1, eps=1e-3)
# mw.display()
# exit()



#Read-in mesh: read V, T, U, rot-clusters, skinning handles, Modes (if reduced)
# mw = Meshwork.Preprocessing(_VT = None)
# mesh = mw.getMesh(name="woman", modes_used=15)
mw = Meshwork.Preprocessing()
mesh = mw.getMesh(name= "3koval", modes_used=30)

#ARAP setup
arap = Arap.ARAP(imesh=mesh, filen="snapshots/")

#Elasticity setup
neo = Neo.NeohookeanElastic(imesh = mesh)

#Solver setup
ti = Solvers.TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neo)

#Running
disp = Display.Display(isolve = ti)
# disp.headless()
disp.display()

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

# Simple example of rigid bodies in 3d
# Catalog all fast methods for deformation. 
# Projective dynamics (dont handle anisotropy - can't use for muscles), 
# 		multigrids, sparse meshless methods,
# 		Dextrous manipulation (juggling) coarsened muscles, they're super stiff, 
#  		using corated elasticity (bad artifacts like Breannan Smith's paper) so don't use corated elasaticity
# With proper neohookean elasticity, you gotta use Tian Tian Liu's paper (which is a Quasi-Newton solve with warm start of Hessian)
# 		And Dave can break this because of some manually set parameters

# Francois's frame based paper. 
# Ignores cross-coupling between rotational degrees of freedom. Our paper computes frames via ARAP, Faure explicitly parametrizes frames.
# Hard to incorporate Faure into existing things
# Our method is easy to apply into tet-meshes
# Borrow their idea of material aware shape functions?

#ACM art instead of ACM siggraph latex template, on github
# Related works table - build table of features that papers support or not
# -- arbitrary material model (biomechanically correct)
# -- Easy model reduction for performance
# -- Anisotropy
# -- Performance
# -- Supports complex geometry, supports skeltal structures bones/rigid body incorporation

# Look up projective dynamics and Position based dynamics
# Real-time Large-deformation Substructuring by Jernej Barbic
# Number the equations
# Small rigid body chain - shoulder elbow, etc...
# Flex muscle and drive the bone (dynamics or statics) or move the bone and solve for the muscle shape (statics)
# 
# Figure out what the mass matrix would look like for Backwards Euler energy minization
# Start implementing dynamic rigid bodies if mass matrix is reasonable
# 
#Sept 11th
#Generating Muscle Geometries:
#Given a vector field, generate geometry for the tendon
#No vector field, generate muscle fiber field and tendon geometry
#Jernej's Barbic Emersion stuff (paper read it)
#Houdini had a software where you can click two attachments and it creates
#	a muscle connection
#Sheet muscle has interesting geometric properties