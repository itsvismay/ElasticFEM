#running the code version 3
import Meshwork
import Arap
import Neo
import Display
import Solvers

#Mesh creation
# VTU,tofix = Meshwork.feather_muscle2_test_setup()
VTU = Meshwork.rectangle_mesh(x=10, y=10, step=0.1)
mw = Meshwork.Preprocessing(_VT = VTU)
# d.display()

#Read-in mesh: read V, T, U, rot-clusters, skinning handles, Modes (if reduced)
mesh = mw.getMesh()
print(mesh.fixed)

#ARAP setup
arap = Arap.ARAP(imesh=mesh, filen="snapshots/")

#Elasticity setup
neo = Neo.NeohookeanElastic(imesh = mesh)

#Solver setup
ti = Solvers.TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neo)

#Running
disp = Display.Display(imesh = mesh, iarap = arap, ineo = neo, isolve = ti)
disp.display()

