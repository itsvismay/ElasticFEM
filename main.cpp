#include <stdio.h>

#include <igl/viewer/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>
#include <json.hpp>

using json = nlohmann::json;


int main(int argc, char *argv[])
{
	if(argc<2){
		
	}else{
		std::cout<<"No config file specified. Using defaults."
		json j_config_parameters;
		std::ifstream  config_json_file(std::string(argv[1]));
		config_json_file >> j_config_parameters;
	}
	
	return 0;

}