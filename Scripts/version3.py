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


#Mesh creation
# VTU,tofix = Meshwork.feather_muscle2_test_setup()
# V = igl.eigen.MatrixXd()
# F = igl.eigen.MatrixXi()
# igl.readOFF("./MeshSetups/woman/mid_woman.off", V, F)
# VTU = [np.array(e2p(V)[:,:2]/100.0), np.array(e2p(F))]
# VTU = Meshwork.rectangle_mesh(x=100, y=100, step=0.1)
# mw = Meshwork.Preprocessing(_VT = VTU, modes_used=15)
# mw.display()
# exit()


#Read-in mesh: read V, T, U, rot-clusters, skinning handles, Modes (if reduced)
mw = Meshwork.Preprocessing(_VT = None)
mesh = mw.getMesh(name="woman", modes_used=15)

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

