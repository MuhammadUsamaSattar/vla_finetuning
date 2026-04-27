#include "sim/mujoco_sim.hpp"

#include <math.h>

#include <Eigen/Dense>
#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "mujoco/mjtnum.h"
#include "mujoco/mujoco.h"
#include "sim/controller.hpp"
#include "sim/hdf5_saver.hpp"

namespace fs = std::filesystem;

bool Sim::key_tx_pos = false, Sim::key_tx_neg = false;  // I/K → X
bool Sim::key_ty_pos = false, Sim::key_ty_neg = false;  // J/L → Y
bool Sim::key_tz_pos = false, Sim::key_tz_neg = false;  // U/P → Z

// Rotation keys (XYZ)
bool Sim::key_rx_pos = false, Sim::key_rx_neg = false;  // W/S → X
bool Sim::key_ry_pos = false, Sim::key_ry_neg = false;  // A/D → Y
bool Sim::key_rz_pos = false, Sim::key_rz_neg = false;  // Q/E → Z

bool Sim::key_gripper_open = false;   // F
bool Sim::key_gripper_close = false;  // H

// EE control
double Sim::EE_STEP = 0.000625;
double Sim::ROT_STEP = 0.0025;
const double Sim::EE_STEP_STEP = EE_STEP / 10;
const double Sim::ROT_STEP_STEP = ROT_STEP / 10;

// Reset episdoe
bool Sim::reset_episode = false;

Sim::Sim(Controller& controller,
         int video_frame_rate,
         int save_frame_rate,
         std::array<std::string, 3> tasks,
         bool save)
  : controller(controller) {
    setup_env(video_frame_rate, save_frame_rate, tasks, save);
}
// ─── Main
// ───────────────────────────────────────────────────────────────────
void Sim::run_sim() {
    if (saver != nullptr)
        saver_thread = std::thread(&HDF5Saver::run_write_loop, saver.get());

    std::srand(std::time({}));
    std::array<std::string, 6> task_desc{
        "Pick the ",
        " and place it on the platform.",
        "Take the ",
        " to the platform.",
        "Move the ",
        " to the platform.",
    };

    while (!glfwWindowShouldClose(window)) {
        double prev_save_time = 0.0;
        double prev_video_time = 0.0;
        std::array<mjtNum, 7> ee_pose{};
        double prev_gripper_ctrl = gripper_ctrl;
        std::array<double, 6> delta{};

        if (saver != nullptr)
            saver->new_episode();

        while (!reset_episode && !glfwWindowShouldClose(window)) {
            // Reset episode
            if (reset_episode)
                break;

            double sim_time = mj_data->time;
            double real_time = glfwGetTime();

            if (sim_time > real_time) {
                std::this_thread::sleep_for(std::chrono::duration<double>(sim_time - real_time));
            }

            auto [dx, dy, dz, drx, dry, drz] = controller.get_deltas(key_tx_pos,
                                                                     key_tx_neg,
                                                                     key_ty_pos,
                                                                     key_ty_neg,
                                                                     key_tz_pos,
                                                                     key_tz_neg,
                                                                     key_rx_pos,
                                                                     key_rx_neg,
                                                                     key_ry_pos,
                                                                     key_ry_neg,
                                                                     key_rz_pos,
                                                                     key_rz_neg,
                                                                     EE_STEP,
                                                                     ROT_STEP);
            auto new_delta = applyEEDelta(dx, dy, dz, drx, dry, drz);
            for (unsigned int i = 0; i < (unsigned int)delta.size(); i++) {
                delta[i] += new_delta[i];
            }

            for (int i = 0; i < 7; i++) mj_data->ctrl[i] = q_target[i];

            if (key_gripper_close)
                gripper_ctrl = std::max(0.0, gripper_ctrl - GRIPPER_STEP);
            if (key_gripper_open)
                gripper_ctrl = std::min(255.0, gripper_ctrl + GRIPPER_STEP);
            mj_data->ctrl[7] = gripper_ctrl;

            mj_step(mj_model, mj_data);

            // ── Rendering
            // ──────────────────────────────────────────────────────
            if ((mj_data->time - prev_video_time) > (1.0 / (double)video_frame_rate)) {
                int W, H;
                glfwGetFramebufferSize(window, &W, &H);
                int half_W = W / 2;
                int half_H = H / 2;

                // 1. Main view — left half
                mjrRect main_view = {0, 0, half_W, half_H};
                auto [main_rgb, main_depth] = renderCamera("main", main_view);

                // 2. Side view — right half
                mjrRect side_view = {half_W, 0, half_W, half_H};
                renderCamera("side", side_view);

                // 3. Side view — front half
                mjrRect front_view = {0, half_H, half_W, half_H};
                renderCamera("front", front_view);

                // 4. Wrist inset — bottom-right corner of left half
                mjrRect inset = {half_W, half_H, half_W, half_H};
                auto [wrist_rgb, wrist_depth] = renderCamera("wrist_mount", inset);

                // 5. HUD on main view
                // char info[512];
                std::array<char, 512U> info;
                int hand_id = mj_name2id(mj_model, mjOBJ_BODY, "hand");
                snprintf(info.data(),
                         info.size(),
                         "Episode: %d\n"
                         "Task: %s\n"
                         "Step Rates:  Increase: =  Decrease: -\n"
                         "Translation  I/K: X  J/L: Y  U/O: Z\n"
                         "Rotation     W/S: X  A/D: Y  Q/E: Z\n"
                         "Gripper      F: Open  H: Close  (%.0f)\n"
                         "Reset Episode: Spacebar\n"
                         "Time: %.2f\n"
                         "X: %f  Y: %f  Z: %f\n"
                         "w: %f  x: %f  y: %f  z: %f",
                         task_idx,
                         tasks[task_idx % tasks.size()].c_str(),
                         gripper_ctrl,
                         mj_data->time,
                         mj_data->xpos[hand_id * 3 + 0],
                         mj_data->xpos[hand_id * 3 + 1],
                         mj_data->xpos[hand_id * 3 + 2],
                         mj_data->xquat[hand_id * 4 + 0],
                         mj_data->xquat[hand_id * 4 + 1],
                         mj_data->xquat[hand_id * 4 + 2],
                         mj_data->xquat[hand_id * 4 + 3]);

                // 5. Labels
                mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, main_view, info.data(), NULL, &con);
                mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, main_view, "Main", NULL, &con);
                mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, side_view, "Side", NULL, &con);
                mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, front_view, "Front", NULL, &con);
                mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, inset, "Wrist Camera", NULL, &con);

                // 6. Save images
                if (saver != nullptr &&
                    ((mj_data->time - prev_save_time) >= (1.0 / (double)save_frame_rate))) {
                    std::copy(delta.begin(), delta.end(), ee_pose.begin());
                    ee_pose[6] = gripper_ctrl - prev_gripper_ctrl;
                    prev_gripper_ctrl = gripper_ctrl;
                    int task_desc_idx = rand() % (task_desc.size() / 2);

                    std::array<mjtNum, 8> state{mj_data->ctrl[0],
                                                mj_data->ctrl[1],
                                                mj_data->ctrl[2],
                                                mj_data->ctrl[3],
                                                mj_data->ctrl[4],
                                                mj_data->ctrl[5],
                                                mj_data->ctrl[6],
                                                mj_data->ctrl[7]};
                    saver->push(std::move(main_rgb),
                                std::move(wrist_rgb),
                                std::move(main_depth),
                                std::move(wrist_depth),
                                half_W,
                                half_H,
                                std::move(ee_pose),
                                std::move(state),
                                task_desc[2 * task_desc_idx + 0] + tasks[task_idx % tasks.size()] +
                                    task_desc[2 * task_desc_idx + 1],
                                mj_data->time);
                    prev_save_time = mj_data->time;
                }

                prev_video_time = mj_data->time;
                glfwSwapBuffers(window);
            }
            glfwPollEvents();
        }
        task_idx += 1;
        resetEpisode();
    }
    if (saver_thread.joinable()) {
        saver->running.store(false);
        saver_thread.join();
    }
}

Sim::~Sim() {
    if (saver != nullptr)
        saver->close();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);
    mj_deleteVFS(&vfs);
    glfwTerminate();
}

// ─── VFS helper ─────────────────────────────────────────────────────────────
void Sim::loadMeshFilesToVFS(mjVFS& vfs, const std::string& assets_dir) {
    if (!fs::exists(assets_dir))
        throw std::runtime_error("Assets directory does not exist: " + assets_dir);

    for (const auto& entry : fs::directory_iterator(assets_dir)) {
        if (!entry.is_regular_file())
            continue;
        std::string ext = entry.path().extension().string();
        if (ext == ".stl" || ext == ".obj" || ext == ".urdf") {
            std::string filename = entry.path().filename().string();
            int result = mj_addFileVFS(&vfs, assets_dir.c_str(), filename.c_str());
            if (result != 0)
                fprintf(stderr,
                        "Warning: Could not add %s to VFS (err %d)\n",
                        filename.c_str(),
                        result);
        }
    }
}

// ─── IK: Damped pseudoinverse (6DOF) ────────────────────────────────────────
std::array<double, 6> Sim::applyEEDelta(
    double dx, double dy, double dz, double drx, double dry, double drz) {
    int hand_id = mj_name2id(mj_model, mjOBJ_BODY, "hand");
    // if (hand_id < 0)
    // 	return;

    // Get hand rotation matrix (3x3, row-major) from world frame
    const mjtNum* R = mj_data->xmat + 9 * hand_id;

    // Transform translation delta from hand local frame to world frame
    // world_delta = R * local_delta
    double wx = R[0] * dx + R[1] * dy + R[2] * dz;
    double wy = R[3] * dx + R[4] * dy + R[5] * dz;
    double wz = R[6] * dx + R[7] * dy + R[8] * dz;

    // Transform rotation delta from hand local frame to world frame
    double wrx = R[0] * drx + R[1] * dry + R[2] * drz;
    double wry = R[3] * drx + R[4] * dry + R[5] * drz;
    double wrz = R[6] * drx + R[7] * dry + R[8] * drz;

    // Initialize jacp and jacr with 21x3 = 81 elements with 0 value
    int nv = mj_model->nv;
    std::vector<mjtNum> jacp(3 * nv, 0);
    std::vector<mjtNum> jacr(3 * nv, 0);
    mj_jacBody(mj_model, mj_data, jacp.data(), jacr.data(), hand_id);

    const int n_arm = 7;
    Eigen::MatrixXd J(6, n_arm);  // Initialize vector 6 x 7
    // std::cout << J.size() << std::endl;

    // Assign Jacobian with 6 DOFs and 7 joints
    // Iterate over (0 - 2) x (0 - 6) = 3 x 7
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < n_arm; j++) J(i, j) = jacp[i * nv + j];
    // Iterate over (0 - 2) x (0 - 6) = 3 x 7
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < n_arm; j++) J(i + 3, j) = jacr[i * nv + j];

    // Use world-frame deltas
    Eigen::VectorXd v(6);
    v << wx, wy, wz, wrx, wry, wrz;

    const double lambda = 0.01;
    // Damped Psuedo-Inverse
    // JJT = J * Jt
    // J+ = J_perose_inv = Jt * (JJT + lamda2 * I)-1
    auto JJT = J * J.transpose();  // 6 x 6
    // 7 x 6
    auto J_pinv =
        J.transpose() * (JJT + lambda * lambda * Eigen::MatrixXd::Identity(6, 6)).inverse();

    // 7 x 1
    // dq = J+ * ee_pose
    Eigen::VectorXd dq = J_pinv * v;

    const double gain{375};
    for (int i = 0; i < n_arm; i++) {
        q_target[i] += dq(i) * mj_model->opt.timestep * gain;
        q_target[i] = std::clamp(q_target[i],
                                 (double)mj_model->actuator_ctrlrange[2 * i],
                                 (double)mj_model->actuator_ctrlrange[2 * i + 1]);
    }

    return {wx, wy, wz, wrx, wry, wrz};
}

// ─── Camera render helper ────────────────────────────────────────────────────
std::pair<std::vector<unsigned char>, std::vector<float>> Sim::renderCamera(
    const std::string_view cam_name, const mjrRect& rect) {
    mjvCamera c;
    mjv_defaultCamera(&c);
    int id = mj_name2id(mj_model, mjOBJ_CAMERA, cam_name.data());
    if (id >= 0) {
        c.type = mjCAMERA_FIXED;
        c.fixedcamid = id;
    }
    mjv_updateScene(mj_model, mj_data, &opt, NULL, &c, mjCAT_ALL, &scn);
    mjr_render(rect, &scn, &con);

    std::vector<unsigned char> rgb(rect.width * rect.height * 3);
    std::vector<float> depth(rect.width * rect.height);
    mjr_readPixels(rgb.data(), depth.data(), rect, &con);

    return {rgb, depth};
}

// ─── Callbacks ──────────────────────────────────────────────────────────────

void Sim::resetEpisode() {
    reset_episode = false;

    int key_id = mj_name2id(mj_model, mjOBJ_KEY, "home");
    if (key_id >= 0) {
        mj_resetDataKeyframe(mj_model, mj_data, key_id);
        for (int i = 0; i < 7; i++) q_target[i] = mj_data->ctrl[i];
    }

    std::array<int, 3> body_ids{mj_name2id(mj_model, mjOBJ_BODY, "cylinder"),
                                mj_name2id(mj_model, mjOBJ_BODY, "ball"),
                                mj_name2id(mj_model, mjOBJ_BODY, "box")};

    std::array<std::pair<double, double>, 3> locs{};
    for (size_t i = 0; i < body_ids.size(); i++) {
        int joint_id = mj_model->body_jntadr[body_ids[i]];  // joint index
        int q_address = mj_model->jnt_qposadr[joint_id];    // qpos start

        double x, y;
        do {
            auto [x_new, y_new] = get_random_position(0.3, 0.5);
            x = x_new;
            y = y_new;
        } while ((i != 0) && (get_dist(std::pair<double, double>{x, y}, locs) < 0.05));
        locs[i].first = x;
        locs[i].second = y;

        mj_data->qpos[q_address + 0] = x;
        mj_data->qpos[q_address + 1] = y;
        mj_data->qpos[q_address + 2] = 0.025;

        // keep orientation valid (unit quaternion!)
        // mj_data->qpos[qadr + 3] = 1.0;
        // mj_data->qpos[qadr + 4] = 0.0;
        // mj_data->qpos[qadr + 5] = 0.0;
        // mj_data->qpos[qadr + 6] = 0.0;
    }
}

void Sim::get_model_and_data() {
    std::string pkg_path = ament_index_cpp::get_package_share_directory("sim");
    std::string xml_path = pkg_path + "/mujoco/scene.xml";
    std::string assets_path = pkg_path + "/mujoco/franka_emika_panda/assets";
    std::string plugin_dir = std::getenv("HOME");
    plugin_dir += "/mujoco/bin/mujoco_plugin";

    // Load mesh decoder plugins
    std::string stl_plugin = plugin_dir + "/libstl_decoder.so";
    std::string obj_plugin = plugin_dir + "/libobj_decoder.so";
    mj_loadPluginLibrary(stl_plugin.c_str());
    mj_loadPluginLibrary(obj_plugin.c_str());

    mj_defaultVFS(&vfs);
    try {
        loadMeshFilesToVFS(vfs, assets_path);
    } catch (const std::exception& e) {
        mj_deleteVFS(&vfs);
        throw;
    }

    // Load model
    char errstr[1000];
    mj_model = mj_loadXML(xml_path.c_str(), &vfs, errstr, sizeof(errstr));
    if (!mj_model) {
        mj_deleteVFS(&vfs);
        throw std::runtime_error("Failed to load model: " + std::string(errstr));
    }
    mj_data = mj_makeData(mj_model);
}

void Sim::setup_env(int video_frame_rate,
                    int save_frame_rate,
                    std::array<std::string, 3> tasks,
                    bool save) {
    get_model_and_data();

    // Set initial pose from keyframe
    resetEpisode();

    // Init GLFW
    if (!glfwInit())
        throw std::runtime_error("Failed to init GLFW");

    window = glfwCreateWindow(1920, 1080, "MuJoCo Sim", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, keyboard);

    // MuJoCo rendering
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(mj_model, &scn, 2000);
    mjr_makeContext(mj_model, &con, mjFONTSCALE_150);

    // Main camera
    int cam_id = mj_name2id(mj_model, mjOBJ_CAMERA, "main");
    if (cam_id >= 0) {
        cam.type = mjCAMERA_FIXED;
        cam.fixedcamid = cam_id;
    } else {
        fprintf(stderr, "Warning: camera 'main' not found, using free camera\n");
    }

    this->video_frame_rate = video_frame_rate;
    this->save_frame_rate = save_frame_rate;
    this->tasks = tasks;

    if (save)
        saver = std::make_unique<HDF5Saver>("temp_data");
}

std::array<double, 2> get_random_position(double a, double b) {
    double r = ((static_cast<double>(std::rand()) / RAND_MAX) * (b - a)) + a;
    double theta = (static_cast<double>(std::rand()) / RAND_MAX) * M_PI * 2;

    return std::array<double, 2>{r * cos(theta), r * sin(theta)};
}

double get_dist(std::pair<double, double> new_loc, std::array<std::pair<double, double>, 3>& locs) {
    double min_dist = 10000000;
    for (auto p : locs) {
        min_dist = std::min(
            min_dist, sqrt(pow(new_loc.first - p.first, 2) + pow(new_loc.second - p.second, 2)));
    }
    return min_dist;
}
