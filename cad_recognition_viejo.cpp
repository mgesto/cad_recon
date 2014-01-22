#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
//#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/apps/render_views_tesselated_sphere.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr;

std::string model_filename_;
std::string scene_filename_;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);

float model_ss_ (0.01f);
float scene_ss_ (0.05f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);

void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*           Cad Model Recognition - Usage Guide                           *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.stl scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene filenames
 std::vector<int> modelname;
 std::vector<int> scenename;
  modelname = pcl::console::parse_file_extension_argument (argc, argv, ".stl");
  if (modelname.size () != 1)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }
    scenename = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (scenename.size () != 1)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }

  model_filename_ = argv[modelname[0]];
  scene_filename_ = argv[scenename[0]];

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
}


void calculatepointsofview (std::string path,std::vector<CloudPtr>& views)
{
	//Inicializate the models
	vtkSmartPointer<vtkPolyData> polygonPolyData = vtkSmartPointer<vtkPolyData>::New();
	pcl::PolygonMesh mesh;
	//load from file the model and convert to vtk format
	pcl::io::loadPolygonFileSTL("C:\\Users\\tidop\\Desktop\\CuchilloCADmetros.stl",mesh);
	pcl::VTKUtils::convertToVTK(mesh,polygonPolyData);
	// create the 80 points of view of the model
	pcl::apps::RenderViewsTesselatedSphere render_views;
	render_views.setResolution (150);
	////render_views.setViewAngle(43.0); //field of view of kinect vertical
	render_views.setViewAngle(57.0); // field of view of kinect horizontal I use this, because shot can deal with oclusions in the real mode, more information it will be better???
	////ask about the previous question
	render_views.setGenOrganized(false);
	render_views.setTesselationLevel (1); //80 views
	render_views.addModelFromPolyData(polygonPolyData); //vtk model
	render_views.generateViews ();
	////std::vector< CloudPtr > views; in the header of the funcion
	//// if i want the poses I need to do the same but with .getposes
	//// this funcion is ready, we can think in storage the 80 points of view,
	render_views.getViews (views);

}
bool findmodelonebyone(pcl::PointCloud<PointType>::Ptr model,pcl::PointCloud<PointType>::Ptr scene,std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& rototranslations)
{

	pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
	pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
	pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<int> sampled_indices;
	// I need to downsample the scene to the model resolution (150*150)
	pcl::UniformSampling<PointType> uniform_sampling;
	uniform_sampling.setInputCloud (scene);
	uniform_sampling.setRadiusSearch (scene_ss_);// the resolution of the model is 150*150, the resolution of the scene is 640*480, 
	// I can supose that the resolution of the scene have 10 times less points
	// actually is 14, but the scene is not a square, and to have enough numbers.

	uniform_sampling.compute (sampled_indices);
	pcl::copyPointCloud (*scene, sampled_indices.points, *scene_keypoints);
	//  Compute Normals
	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
	norm_est.setKSearch (8);// look at this value can be changued
	norm_est.setInputCloud (model);
	norm_est.compute (*model_normals);
	norm_est.setNumberOfThreads(8);
	norm_est.setInputCloud (scene);
	norm_est.compute (*scene_normals);


	pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
    pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
 

	//
	//  Compute Descriptor for keypoints
	//
	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
	descr_est.setNumberOfThreads(8);
	descr_est.setRadiusSearch (descr_rad_);// ?? set up with the best option

	descr_est.setInputCloud (model);
	descr_est.setInputNormals (model_normals);
	//descr_est.setSearchSurface (model);
	descr_est.compute (*model_descriptors);

	descr_est.setInputCloud (scene_keypoints);
	descr_est.setInputNormals (scene_normals);
	descr_est.setSearchSurface (scene);
	descr_est.compute (*scene_descriptors);


	// changue the code, compute only 1 time the scene descriptors
	//
	//  Find Model-Scene Correspondences with KdTree
	//
	pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

	pcl::KdTreeFLANN<DescriptorType> match_search;
	match_search.setInputCloud (model_descriptors);
	//  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	//This is used for 
	for (size_t i = 0; i < scene_descriptors->size (); ++i)
	{
		std::vector<int> neigh_indices (1);
		std::vector<float> neigh_sqr_dists (1);
		if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
		{
			continue;
		}
		int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
		if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
		{
			pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			model_scene_corrs->push_back (corr);
		}
	}

	//
	//  Actual Clustering
	//
	
	std::vector<pcl::Correspondences> clustered_corrs;

	//  Using Hough3D
	//
	//  Compute (Keypoints) Reference Frames only for Hough
	//

	pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
	pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

	pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
	rf_est.setFindHoles (true);
	//rf_est.setRadiusSearch (rf_rad_);
	rf_est.setRadiusSearch (rf_rad_);
	// i use model and not model_keypoints because in the model i didnt do a subsample
	rf_est.setInputCloud (model);
	rf_est.setInputNormals (model_normals);
	rf_est.setSearchSurface (model);
	rf_est.compute (*model_rf);

	rf_est.setInputCloud (scene_keypoints);
	rf_est.setInputNormals (scene_normals);
	rf_est.setSearchSurface (scene);
	rf_est.compute (*scene_rf);

	//  Clustering
	// I take the number of models in the scene, Here in my test I will only have one result
	pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
	clusterer.setHoughBinSize (cg_size_);
	clusterer.setHoughThreshold (cg_thresh_);
	clusterer.setUseInterpolation (true);
	clusterer.setUseDistanceWeight (false);
	// i use model and not model_keypoints because in the model i didnt do a subsample
	clusterer.setInputCloud (model);
	clusterer.setInputRf (model_rf);
	clusterer.setSceneCloud (scene_keypoints);
	clusterer.setSceneRf (scene_rf);
	clusterer.setModelSceneCorrespondences (model_scene_corrs);

	clusterer.recognize (rototranslations, clustered_corrs);

	if(rototranslations.size()==0) return false;
	else if(rototranslations.size()>0) return true;// maybe here is better ==1
	else return true;
}

int
main (int argc, char *argv[])
{
 // parseCommandLine (argc, argv);
  std::vector<CloudPtr> views;
  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  // create the vector of point clouds with the views of the object
  // try to reduce the number of points of view in symmetric objects( camera roll histogram, look)
  calculatepointsofview(model_filename_,views);

  //load the scene from asus xtion

  if (pcl::io::loadPCDFile ("C:\\Users\\tidop\\Desktop\\cuccu.pcd", *scene) < 0)
  {
    std::cout << "Error loading scene cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }
  // this is the vector with the results and the model obtained( in my case only 1)
   std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
   // check for all the points of view 
   int numeroencontrado=0;
  for ( int i=0; i<views.size();i++)
  {
	  pcl::PointCloud<PointType>::Ptr actualview (new pcl::PointCloud<PointType> ());
	  actualview= views[i];

	  
	  bool find=findmodelonebyone(actualview,scene,rototranslations);
	  if (find ==true)
	  {
		  model=views[i];
		  numeroencontrado++;
		 // if we find we see the result to see if this is correct
		  break;
	  }

  }


  //
  //  Output results
  //
  std::cout << "Model instances found: " << numeroencontrado << std::endl;

  //
  //  Visualization
  ////
  //pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
  //viewer.addPointCloud (scene, "scene_cloud");


  //for (size_t i = 0; i < rototranslations.size (); ++i)
  //{
  //  pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
  //  pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

  //  std::stringstream ss_cloud;
  //  ss_cloud << "instance" << i;

  //  pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
  //  viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

  //}

  //while (!viewer.wasStopped ())
  //{
  //  viewer.spinOnce ();
  //}

  return (0);
}
