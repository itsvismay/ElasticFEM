#include <igl/viewer/Viewer.h>
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

void rep(const real_1d_array &x, double func, void *ptr)
{}

void func(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    Euler* euler = (Euler*) ptr;
    VectorXd& xbfgs = euler->get_xbfgs();
    for(unsigned int i=0; i<xbfgs.size(); ++i)
    {
        xbfgs(i) = x[i];
    }

    func = euler->getZeroOrder(xbfgs);
    std::cout<<"F: "<<func<<std::endl;
    VectorXd& g = euler->get_g();
    std::cout<<g<<std::endl;
    euler->getFirstOrder(xbfgs, g);
    for(unsigned i=0; i<g.size(); ++i)
    {
        grad[i] = g(i);
    }
}

int main(int argc, char *argv[])
{
    std::cout<<"3d Neohookean Muscle\n";
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
    SM->initializeRSCoords();

    //Set Muscle stuff here
        // Eigen::VectorXd Zc;
        // Zc.resize(SM->getColwiseV().cols());
        // Zc.setZero();
        // double mag = j_input["fibre_mag"];
        // int axis = j_input["axis"];
        // Eigen::Vector3d maxs = SM->getColwiseV().rowwise().maxCoeff();
        // Eigen::Vector3d mins = SM->getColwiseV().rowwise().minCoeff();
        // double t = j_input["thresh"];
        // Eigen::Vector3d thresh = mins + t*(maxs - mins);
        // // std::cout<<maxs.transpose()<<std::endl;
        // // std::cout<<mins.transpose()<<std::endl;
        // // std::cout<<thresh.transpose()<<"\naaa"<<std::endl;
        // Eigen::Vector3d fibre_dir(j_input["fibre_dir"][0], j_input["fibre_dir"][1], j_input["fibre_dir"][2]);
        // #pragma omp parallel for
        // for(auto& tet: SM->getTets())
        // {   
        //     int i0 = tet.getIndices()(0);
        //     int i1 = tet.getIndices()(1);
        //     int i2 = tet.getIndices()(2);
        //     int i3 = tet.getIndices()(3);

        //     if(SM->getColwiseV().col(i0)(axis)< thresh(axis))
        //     {
        //         tet.set_fibre_mag(mag);
        //         tet.set_fibre_dir(fibre_dir);
        //         Zc(i0/3) = mag;
        //         Zc(i1/3) = mag;
        //         Zc(i2/3) = mag;
        //         Zc(i3/3) = mag;
        //     }
        // }
        // Eigen::MatrixXd C;
        // igl::jet(Zc, true, C);
        // //Swap RGB colors because libigl background is blue
        // C.col(2).swap(C.col(1));
    //---

    std::cout<< "STEPPING"<<std::endl;
    // Newmark* nmrk = new Newmark(SM, j_input["timestep"]);
    // nmrk->step();
    Euler *elr = new Euler(SM, j_input["timestep"]);
    elr->bfgs_step(func, rep);
	igl::viewer::Viewer viewer;
    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {   
        if(viewer.core.is_animating)
        {
            elr->bfgs_step(func, rep);
            MatrixXd newV = SM->getCurrentVerts();
            viewer.data.set_vertices(newV);
    	}
        return false;
    };

	viewer.data.set_mesh(V,F);
    viewer.core.show_lines = true;
    viewer.core.invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data.face_based = true;
    // viewer.data.set_colors(C);

    viewer.launch();
    return EXIT_SUCCESS;

}
