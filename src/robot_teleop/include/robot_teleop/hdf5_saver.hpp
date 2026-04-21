#pragma once

#include <H5Cpp.h>
#include <H5public.h>
#include <mujoco/mujoco.h>

#include <format>
#include <string>
#include <vector>

class HDF5Saver {
   public:
    HDF5Saver(const std::string& path);

    void new_episode();

    template <std::size_t N>
    void write_data(const std::vector<uint8_t>& main_img,
                    const std::vector<uint8_t>& wrist_img,
                    const std::vector<float>& main_depth,
                    const std::vector<float>& wrist_depth,
                    const int& W,
                    const int& H,
                    const std::array<mjtNum, N>& ee_pose,
                    const std::array<mjtNum, 8>& state,
                    const std::string& task) {
        // std::cout << ee_pose.size() << std::endl;
        frame_num += 1;
        H5::Group frame{file.createGroup("/" + std::format("{:05}", frame_num))};

        std::vector<hsize_t> dims_rgb{static_cast<hsize_t>(H), static_cast<hsize_t>(W), 3};
        std::vector<hsize_t> dims_depth{static_cast<hsize_t>(H), static_cast<hsize_t>(W)};
        std::vector<hsize_t> dims_ee_pose{static_cast<hsize_t>(ee_pose.size())};
        std::vector<hsize_t> dims_state{static_cast<hsize_t>(state.size())};
        std::vector<hsize_t> dims_task{static_cast<hsize_t>(1)};

        write_img_data(frame, main_img, dims_rgb, H5::PredType::NATIVE_UINT8, "main_img");
        write_img_data(frame, wrist_img, dims_rgb, H5::PredType::NATIVE_UINT8, "wrist_img");
        write_img_data(frame, main_depth, dims_depth, H5::PredType::NATIVE_FLOAT, "main_depth");
        write_img_data(frame, wrist_depth, dims_depth, H5::PredType::NATIVE_FLOAT, "wrist_depth");

        write_vector_data(frame, ee_pose, dims_ee_pose, H5::PredType::NATIVE_DOUBLE, "ee_pose");
        write_vector_data(frame, state, dims_state, H5::PredType::NATIVE_DOUBLE, "state");

        write_task_data(frame, task, "task");

        frame.close();
    }

    void close();

   private:
    H5::H5File file;
    std::string path;
    int frame_num{-1};
    int file_num{-1};

    template <typename T>
    void write_img_data(const H5::Group& frame,
                        const std::vector<T>& img,
                        const std::vector<hsize_t>& dims,
                        const H5::PredType& datatype,
                        const std::string& dataset_name) {
        H5::DataSpace dataspace{static_cast<int>(dims.size()), dims.data()};
        H5::DataSet dataset = frame.createDataSet(dataset_name, datatype, dataspace);

        dataset.write(img.data(), datatype);
    }

    template <std::size_t N>
    void write_vector_data(const H5::Group& frame,
                           const std::array<mjtNum, N>& data,
                           const std::vector<hsize_t>& dims,
                           const H5::PredType& datatype,
                           const std::string& dataset_name) {
        H5::DataSpace dataspace{static_cast<int>(dims.size()), dims.data()};
        H5::DataSet dataset = frame.createDataSet(dataset_name, datatype, dataspace);

        dataset.write(data.data(), datatype);
    }

    void write_task_data(const H5::Group& frame,
                         const std::string& task,
                         const std::string& dataset_name) {
        H5::DataSpace dataspace{H5S_SCALAR};
        H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);

        H5::DataSet dataset = frame.createDataSet(dataset_name, strType, dataspace);

        dataset.write(task, strType);
    }
};
