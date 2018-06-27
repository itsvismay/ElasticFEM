#running the code version 3
import Meshwork 

#Mesh creation
VTU,tofix = Meshwork.feather_muscle2_test_setup()
d = Meshwork.Preprocessing(_VT = VTU)
d.display()

#Read-in mesh 
#read V, T, U, rot-clusters, skinning handles, Modes (if reduced)

#ARAP setup

#Elasticity setup

#Solver setup

#Running

