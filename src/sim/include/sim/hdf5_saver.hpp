#pragma once

#include <H5Cpp.h>
#include <H5public.h>
#include <mujoco/mujoco.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

struct SaveData {
    std::vector<uint8_t> main_img;
    std::vector<uint8_t> wrist_img;
    std::vector<float> main_depth;
    std::vector<float> wrist_depth;
    int W;
    int H;
    std::array<mjtNum, 7> ee_pose{};
    std::array<mjtNum, 8> state{};
    std::string task;
};
class HDF5Saver {
   public:
    std::atomic<bool> running{true};
    HDF5Saver(const std::string& path);

    void new_episode();

    void write_data();
    void run_write_loop();

    void push(std::vector<uint8_t>&& main_img,
              std::vector<uint8_t>&& wrist_img,
              std::vector<float>&& main_depth,
              std::vector<float>&& wrist_depth,
              int W,
              int H,
              std::array<mjtNum, 7>&& ee_pose,
              std::array<mjtNum, 8>&& state,
              std::string task);
    void close();

   private:
    H5::H5File file;
    std::string path;
    int frame_num{-1};
    int file_num{-1};
    std::queue<SaveData> queue;
    std::mutex mtx;

    template <typename T>
    void write_img_data(const H5::Group& frame,
                        const std::vector<T>& img,
                        const std::vector<hsize_t>& dims,
                        const H5::PredType& datatype,
                        const std::string& dataset_name) {
        H5::DataSpace dataspace{static_cast<int>(dims.size()), dims.data()};
        H5::DataSet dataset = frame.createDataSet(dataset_name, datatype, dataspace);

        // If image is 2D or 3D (H x W x C or H x W)
        const size_t H = dims[0];
        const size_t row_size = img.size() / H;

        std::vector<T> flipped(img.size());

        for (size_t i = 0; i < H; i++) {
            const size_t src_row = i;
            const size_t dst_row = H - 1 - i;

            std::copy(img.begin() + src_row * row_size,
                      img.begin() + (src_row + 1) * row_size,
                      flipped.begin() + dst_row * row_size);
        }

        dataset.write(flipped.data(), datatype);
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
