#include "robot_teleop/hdf5_saver.hpp"
#include <H5Cpp.h>
#include <H5DataSet.h>
#include <H5DataSpace.h>
#include <H5File.h>
#include <H5Group.h>
#include <H5PredType.h>
#include <H5Tpublic.h>
#include <H5public.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

HDF5Saver::HDF5Saver(const std::string &path) : path(path) {
	boost::filesystem::create_directories(path);
}

void HDF5Saver::new_episode() {
	file_num += 1;
	frame_num = -1;

	file = H5::H5File(path + "/" + std::format("{:04}", file_num) + ".hdf5",
					  H5F_ACC_TRUNC);
}

void HDF5Saver::write_data(const std::vector<uint8_t> &main_img,
						   const std::vector<uint8_t> &wrist_img,
						   const std::vector<float> &main_depth,
						   const std::vector<float> &wrist_depth, const int W,
						   const int &H, const std::array<mjtNum, 8> &ee_pose,
						   const std::array<mjtNum, 8> &state) {
	frame_num += 1;
	H5::Group frame{file.createGroup("/" + std::format("{:04}", frame_num))};

	std::vector<hsize_t> dims_rgb{static_cast<hsize_t>(H),
								  static_cast<hsize_t>(W), 3};
	std::vector<hsize_t> dims_depth{static_cast<hsize_t>(H),
									static_cast<hsize_t>(W)};
	std::vector<hsize_t> dims_ee_pose{static_cast<hsize_t>(ee_pose.size())};
	std::vector<hsize_t> dims_state{static_cast<hsize_t>(state.size())};

	write_img_data(frame, main_img, dims_rgb, H5::PredType::NATIVE_UINT8,
				   "main_img");
	write_img_data(frame, wrist_img, dims_rgb, H5::PredType::NATIVE_UINT8,
				   "wrist_img");
	write_img_data(frame, main_depth, dims_depth, H5::PredType::NATIVE_FLOAT,
				   "main_depth");
	write_img_data(frame, wrist_depth, dims_depth, H5::PredType::NATIVE_FLOAT,
				   "wrist_depth");

	write_vector_data(frame, ee_pose, dims_ee_pose, H5::PredType::NATIVE_DOUBLE,
					  "ee_pose");
	write_vector_data(frame, state, dims_state, H5::PredType::NATIVE_DOUBLE,
					  "state");

	frame.close();
}

template <typename T>
void HDF5Saver::write_img_data(const H5::Group &frame,
							   const std::vector<T> &img,
							   const std::vector<hsize_t> &dims,
							   const H5::PredType &datatype,
							   const std::string &dataset_name) {
	H5::DataSpace dataspace{static_cast<int>(dims.size()), dims.data()};
	H5::DataSet dataset =
		frame.createDataSet(dataset_name, datatype, dataspace);

	dataset.write(img.data(), datatype);
}

template <std::size_t N>
void HDF5Saver::write_vector_data(const H5::Group &frame,
								  const std::array<mjtNum, N> &data,
								  const std::vector<hsize_t> &dims,
								  const H5::PredType &datatype,
								  const std::string &dataset_name) {
	H5::DataSpace dataspace{static_cast<int>(dims.size()), dims.data()};
	H5::DataSet dataset =
		frame.createDataSet(dataset_name, datatype, dataspace);

	dataset.write(&data, datatype);
}

void HDF5Saver::close() { file.close(); }
