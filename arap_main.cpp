#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>

#include <json.hpp>

#include "newmark.h"
#include "euler.h"

using json = nlohmann::json;

using namespace Eigen;
using namespace std;
json j_input;

int main(int argc, char *argv[])
{
    std::cout<<"3d ARAP\n";
    json j_config_parameters;
    std::ifstream i("../input/input.json");
    i >> j_input;

    double youngs_mod = j_input["youngs"];
    double poisson = j_input["poissons"];
    double gravity = j_input["gravity"];

	MatrixXd V;
	MatrixXi T;
	MatrixXi F;
	igl::readMESH(j_input["mesh_file"], V, T, F);


    Mesh* SM = new Mesh(T, V, youngs_mod, poisson, gravity);
    SM->initializeMesh();
    // SM->initializeRSCoords();

    std::cout<< "STEPPING"<<std::endl;

	igl::opengl::glfw::Viewer viewer;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer)
    {   
        if(viewer.core.is_animating)
        {
            MatrixXd newV = SM->getCurrentVerts();
            viewer.data().set_vertices(newV);
    	}
        return false;
    };

	viewer.data().set_mesh(V,F);
    viewer.data().show_lines = true;
    viewer.data().invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data().face_based = true;

    viewer.launch();
    return EXIT_SUCCESS;

}
