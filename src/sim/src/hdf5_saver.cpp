#include "sim/hdf5_saver.hpp"

#include <H5DataSet.h>
#include <H5PredType.h>
#include <H5Spublic.h>

#include <boost/filesystem.hpp>

HDF5Saver::HDF5Saver(const std::string& path) : path(path) {
    boost::filesystem::create_directories(path);
}

void HDF5Saver::new_episode() {
    file_num += 1;
    frame_num = -1;

    file = H5::H5File(path + "/" + std::format("{:05}", file_num) + ".hdf5", H5F_ACC_TRUNC);
}

void HDF5Saver::close() {
    file.close();
}
