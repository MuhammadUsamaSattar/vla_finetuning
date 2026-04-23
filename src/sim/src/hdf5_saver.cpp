#include "sim/hdf5_saver.hpp"

#include <H5DataSet.h>
#include <H5PredType.h>
#include <H5Spublic.h>

#include <boost/filesystem.hpp>
#include <format>
#include <mutex>
#include <queue>

HDF5Saver::HDF5Saver(const std::string& path) : path(path) {
    boost::filesystem::create_directories(path);
}

void HDF5Saver::new_episode() {
    while (!queue.empty()) {
        write_data();
    }
    file_num += 1;
    frame_num = -1;

    file = H5::H5File(path + "/" + std::format("{:05}", file_num) + ".hdf5", H5F_ACC_TRUNC);
}

void HDF5Saver::run_write_loop() {
    while (running.load()) write_data();
}

void HDF5Saver::write_data() {
    SaveData data;
    H5::Group frame;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty())
            return;

        data = std::move(queue.front());
        queue.pop();
        frame_num += 1;
        frame = H5::Group{file.createGroup("/" + std::format("{:05}", frame_num))};
    }

    std::vector<hsize_t> dims_rgb{static_cast<hsize_t>(data.H), static_cast<hsize_t>(data.W), 3};
    std::vector<hsize_t> dims_depth{static_cast<hsize_t>(data.H), static_cast<hsize_t>(data.W)};
    std::vector<hsize_t> dims_ee_pose{static_cast<hsize_t>(data.ee_pose.size())};
    std::vector<hsize_t> dims_state{static_cast<hsize_t>(data.state.size())};
    std::vector<hsize_t> dims_task{static_cast<hsize_t>(1)};

    write_img_data(frame, data.main_img, dims_rgb, H5::PredType::NATIVE_UINT8, "main_img");
    write_img_data(frame, data.wrist_img, dims_rgb, H5::PredType::NATIVE_UINT8, "wrist_img");
    write_img_data(frame, data.main_depth, dims_depth, H5::PredType::NATIVE_FLOAT, "main_depth");
    write_img_data(frame, data.wrist_depth, dims_depth, H5::PredType::NATIVE_FLOAT, "wrist_depth");

    write_vector_data(frame, data.ee_pose, dims_ee_pose, H5::PredType::NATIVE_DOUBLE, "ee_pose");
    write_vector_data(frame, data.state, dims_state, H5::PredType::NATIVE_DOUBLE, "state");

    write_task_data(frame, data.task, "task");

    frame.close();
}

void HDF5Saver::push(std::vector<uint8_t>&& main_img,
                     std::vector<uint8_t>&& wrist_img,
                     std::vector<float>&& main_depth,
                     std::vector<float>&& wrist_depth,
                     int W,
                     int H,
                     std::array<mjtNum, 7>&& ee_pose,
                     std::array<mjtNum, 8>&& state,
                     std::string task) {
    std::lock_guard<std::mutex> lock(mtx);
    queue.push(SaveData{std::move(main_img),
                        std::move(wrist_img),
                        std::move(main_depth),
                        std::move(wrist_depth),
                        W,
                        H,
                        std::move(ee_pose),
                        std::move(state),
                        std::move(task)});
}

void HDF5Saver::close() {
    file.close();
}
