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
#Read-in mesh: read V, T, U, rot-clusters, skinning handles, Modes (if reduced)
VTU = Meshwork.rectangle_mesh(x=5, y=5, step=1.0)
mw = Meshwork.Preprocessing(_VT = VTU)
mw.Fix = Meshwork.get_max(mw.V, a=1, eps=1e-2)	
mw.Mov = Meshwork.get_min(mw.V, a=1, eps=1e-2)
mw.display()
exit()

mw = Meshwork.Preprocessing()
mesh = mw.getMesh(name= "test2x2", modes_used=30)

#ARAP setup
arap = Arap.ARAP(imesh=mesh, filen="snapshots/")

#Elasticity setup
neo = Neo.NeohookeanElastic(imesh = mesh)

#Solver setup
ti = Solvers.TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neo)

#Running
disp = Display.Display(isolve = ti)

# disp.headless()
disp.display_dynamics()

