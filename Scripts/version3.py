#running the code version 3
import Meshwork 

#Mesh creation
VTU = Meshwork.rectangle_mesh(30, 10, step=0.1)
d = Meshwork.Preprocessing(_VT = VTU)
d.display()

#Read-in mesh 
#read V, T, U, rot-clusters, skinning handles, Modes (if reduced)

#ARAP setup

#Elasticity setup

#Solver setup

#Running

