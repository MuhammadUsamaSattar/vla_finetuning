#pragma once

#include <H5Cpp.h>
#include <mujoco/mujoco.h>
#include <string>
#include <vector>

class HDF5Saver {
  public:
	HDF5Saver(const std::string &path);

	void new_episode();

	void write_data(const std::vector<uint8_t> &main_img,
					const std::vector<uint8_t> &wrist_img,
					const std::vector<float> &main_depth,
					const std::vector<float> &wrist_depth, const int W,
					const int &H, const std::array<mjtNum, 8> &ee_pose,
					const std::array<mjtNum, 8> &state);

	template <typename T>
	void write_img_data(const H5::Group &frame, const std::vector<T> &img,
						const std::vector<hsize_t> &dims,
						const H5::PredType &datatype,
						const std::string &dataset_name);

	template <std::size_t N>
	void write_vector_data(const H5::Group &frame,
						   const std::array<mjtNum, N> &data,
						   const std::vector<hsize_t> &dims,
						   const H5::PredType &datatype,
						   const std::string &dataset_name);

	void close();

  private:
	H5::H5File file;
	std::string path;
	int frame_num{-1};
	int file_num{-1};
	int H;
	int W;
};
