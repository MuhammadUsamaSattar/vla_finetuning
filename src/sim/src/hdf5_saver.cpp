#include "sim/hdf5_saver.hpp"

#include <H5Attribute.h>
#include <H5DataSet.h>
#include <H5Exception.h>
#include <H5PredType.h>
#include <H5Spublic.h>
#include <H5public.h>

#include <array>
#include <boost/filesystem.hpp>
#include <cstddef>
#include <ctime>
#include <format>
#include <iostream>
#include <mutex>
#include <queue>

HDF5Saver::HDF5Saver(const std::string& path) : path(path) {
    boost::filesystem::create_directories(path);
    file = H5::H5File(path + "/dataset.hdf5", H5F_ACC_TRUNC);
}

void HDF5Saver::new_episode(int episode_increment) {
    clock_t t{clock()};
    std::cout << "Queue size: " << queue.size() << ", ";
    while (!queue.empty()) {
        continue;
    }
    std::cout << "Emptying Queue: " << ((float)(clock() - t) / CLOCKS_PER_SEC) * 1000 << " ms, ";
    t = clock();

    episode_num += episode_increment;
    frame_num = -1;

    std::string name = std::format("{:05}", episode_num);

    if (H5Lexists(file.getId(), name.c_str(), H5P_DEFAULT) > 0) {
        file.unlink(name);
    }

    episode = file.createGroup(name);
    std::cout << "Creating New File: " << ((float)(clock() - t) / CLOCKS_PER_SEC) * 1000 << " ms, ";
}

void HDF5Saver::run_write_loop() {
    while (running.load()) write_data();
}

size_t HDF5Saver::get_queue_size() {
    std::lock_guard<std::mutex> lock(mtx);
    return queue.size();
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
        frame = episode.createGroup(std::format("{:05}", frame_num));
    }

    std::vector<hsize_t> dims_rgb{static_cast<hsize_t>(data.H), static_cast<hsize_t>(data.W), 3};
    std::vector<hsize_t> dims_depth{static_cast<hsize_t>(data.H), static_cast<hsize_t>(data.W)};
    std::vector<hsize_t> dims_ee_pose{static_cast<hsize_t>(data.ee_pose.size())};
    std::vector<hsize_t> dims_state{static_cast<hsize_t>(data.state.size())};
    std::vector<hsize_t> dims_task{static_cast<hsize_t>(1)};
    std::vector<hsize_t> dim_attr{static_cast<hsize_t>(1)};

    write_img_data(frame, data.main_img, dims_rgb, H5::PredType::NATIVE_UINT8, "main_img");
    write_img_data(frame, data.wrist_img, dims_rgb, H5::PredType::NATIVE_UINT8, "wrist_img");
    write_img_data(frame, data.main_depth, dims_depth, H5::PredType::NATIVE_FLOAT, "main_depth");
    write_img_data(frame, data.wrist_depth, dims_depth, H5::PredType::NATIVE_FLOAT, "wrist_depth");

    write_vector_data(frame, data.ee_pose, dims_ee_pose, H5::PredType::NATIVE_DOUBLE, "ee_pose");
    write_vector_data(frame, data.state, dims_state, H5::PredType::NATIVE_DOUBLE, "state");

    write_task_data(frame, data.task, "task");

    H5::DataSpace attr_dataspace{1, dim_attr.data()};
    H5::Attribute attr{frame.createAttribute("time", H5::PredType::NATIVE_DOUBLE, attr_dataspace)};
    attr.write(H5::PredType::NATIVE_DOUBLE, &data.time);

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
                     std::string task,
                     double time) {
    std::lock_guard<std::mutex> lock(mtx);
    queue.push(SaveData{std::move(main_img),
                        std::move(wrist_img),
                        std::move(main_depth),
                        std::move(wrist_depth),
                        W,
                        H,
                        std::move(ee_pose),
                        std::move(state),
                        std::move(task),
                        time});
}

void HDF5Saver::close() {
    file.close();
}
