#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <filesystem>

#include "mujoco/mjmodel.h"
#include "mujoco/mjtnum.h"
#include "mujoco/mjvisualize.h"
#include "robot_teleop/hdf5_saver.hpp"

namespace fs = std::filesystem;

// Global sim objects
mjModel* mj_model = nullptr;
mjData* mj_data = nullptr;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

// Mouse state
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx{0}, lasty{0};

// Translation keys (XYZ)
bool key_tx_pos = false, key_tx_neg = false;  // I/K → X
bool key_ty_pos = false, key_ty_neg = false;  // J/L → Y
bool key_tz_pos = false, key_tz_neg = false;  // U/P → Z

// Rotation keys (XYZ)
bool key_rx_pos = false, key_rx_neg = false;  // W/S → X
bool key_ry_pos = false, key_ry_neg = false;  // A/D → Y
bool key_rz_pos = false, key_rz_neg = false;  // Q/E → Z

// Gripper keys
bool key_gripper_open = false;   // F
bool key_gripper_close = false;  // H

// EE control
double gripper_ctrl = 255.0;
double EE_STEP = 0.0005;
double ROT_STEP = 0.0005;
const double EE_STEP_STEP = EE_STEP / 10;
const double ROT_STEP_STEP = ROT_STEP / 10;
const double GRIPPER_STEP = 0.1;
double q_target[7] = {0, 0, 0, -1.57079, 0, 1.57079, -0.7853};

// Reset episdoe
bool reset_episode = false;

// ─── VFS helper ─────────────────────────────────────────────────────────────
void loadMeshFilesToVFS(mjVFS& vfs, const std::string& assets_dir) {
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
std::array<double, 6> applyEEDelta(
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

    const double gain{100};
    for (int i = 0; i < n_arm; i++) {
        q_target[i] += dq(i) * mj_model->opt.timestep * gain;
        q_target[i] = std::clamp(q_target[i],
                                 (double)mj_model->actuator_ctrlrange[2 * i],
                                 (double)mj_model->actuator_ctrlrange[2 * i + 1]);
    }

    return {wx, wy, wz, wrx, wry, wrz};
}

// ─── Camera render helper ────────────────────────────────────────────────────
std::pair<std::vector<unsigned char>, std::vector<float>> renderCamera(
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
void keyboard(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GLFW_TRUE);

    bool on = (action == GLFW_PRESS || action == GLFW_REPEAT);
    bool off = (action == GLFW_RELEASE);

    switch (key) {
        // Translation
        case GLFW_KEY_J:
            key_tx_pos = on;
            if (off)
                key_tx_pos = false;
            break;
        case GLFW_KEY_L:
            key_tx_neg = on;
            if (off)
                key_tx_neg = false;
            break;
        case GLFW_KEY_I:
            key_ty_pos = on;
            if (off)
                key_ty_pos = false;
            break;
        case GLFW_KEY_K:
            key_ty_neg = on;
            if (off)
                key_ty_neg = false;
            break;
        case GLFW_KEY_O:
            key_tz_pos = on;
            if (off)
                key_tz_pos = false;
            break;
        case GLFW_KEY_U:
            key_tz_neg = on;
            if (off)
                key_tz_neg = false;
            break;
        // Rotation
        case GLFW_KEY_S:
            key_rx_pos = on;
            if (off)
                key_rx_pos = false;
            break;
        case GLFW_KEY_W:
            key_rx_neg = on;
            if (off)
                key_rx_neg = false;
            break;
        case GLFW_KEY_A:
            key_ry_pos = on;
            if (off)
                key_ry_pos = false;
            break;
        case GLFW_KEY_D:
            key_ry_neg = on;
            if (off)
                key_ry_neg = false;
            break;
        case GLFW_KEY_E:
            key_rz_pos = on;
            if (off)
                key_rz_pos = false;
            break;
        case GLFW_KEY_Q:
            key_rz_neg = on;
            if (off)
                key_rz_neg = false;
            break;
        // Gripper
        case GLFW_KEY_H:
            key_gripper_open = on;
            if (off)
                key_gripper_open = false;
            break;
        case GLFW_KEY_F:
            key_gripper_close = on;
            if (off)
                key_gripper_close = false;
            break;
        // Change steps
        case GLFW_KEY_EQUAL:
            EE_STEP += EE_STEP_STEP;
            ROT_STEP += ROT_STEP_STEP;
            break;
        case GLFW_KEY_MINUS:
            EE_STEP = std::max(EE_STEP - EE_STEP_STEP, 0.0);
            ROT_STEP = std::max(ROT_STEP - ROT_STEP_STEP, 0.0);
            break;
        // Reset episode
        case GLFW_KEY_SPACE:
            if (action == GLFW_PRESS)
                reset_episode = true;
            break;
    }
}

void resetEpisode() {
    int key_id = mj_name2id(mj_model, mjOBJ_KEY, "home");
    if (key_id >= 0) {
        mj_resetDataKeyframe(mj_model, mj_data, key_id);
        for (int i = 0; i < 7; i++) q_target[i] = mj_data->ctrl[i];
    }
}

template <typename Callable>
class ScopeGuard {
   public:
    explicit ScopeGuard(Callable&& callable) : callable_{callable} {}

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard(ScopeGuard&&) = delete;
    ScopeGuard operator=(const ScopeGuard&) = delete;
    ScopeGuard operator=(ScopeGuard&&) = delete;

    ~ScopeGuard() { callable_(); }

   private:
    Callable callable_;
};

template <typename Callable>
ScopeGuard(Callable&&) -> ScopeGuard<Callable>;

// ─── Main
// ───────────────────────────────────────────────────────────────────
int main() {
    std::string pkg_path = ament_index_cpp::get_package_share_directory("robot_teleop");
    std::string xml_path = pkg_path + "/mujoco/scene.xml";
    std::string assets_path = pkg_path + "/mujoco/franka_emika_panda/assets";
    std::string plugin_dir = std::getenv("HOME");
    plugin_dir += "/mujoco/bin/mujoco_plugin";
    HDF5Saver saver{"data"};

    // Load mesh decoder plugins
    std::string stl_plugin = plugin_dir + "/libstl_decoder.so";
    std::string obj_plugin = plugin_dir + "/libobj_decoder.so";
    mj_loadPluginLibrary(stl_plugin.c_str());
    mj_loadPluginLibrary(obj_plugin.c_str());

    // VFS
    mjVFS vfs;
    const ScopeGuard at_exit{[&vfs]() {
        mjv_freeScene(&scn);
        mjr_freeContext(&con);
        mj_deleteData(mj_data);
        mj_deleteModel(mj_model);
        mj_deleteVFS(&vfs);
        glfwTerminate();
    }};

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

    // Set initial pose from keyframe
    resetEpisode();

    // Init GLFW
    if (!glfwInit())
        throw std::runtime_error("Failed to init GLFW");

    GLFWwindow* window = glfwCreateWindow(1920, 1080, "MuJoCo Sim", NULL, NULL);
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

    std::string tasks[3]{"Pick the green cube and place it on the platform.",
                         "Pick the blue cylinder and place it on the platform.",
                         "Pick the red sphere and place it on the platform."};

    // Main loop
    int task_idx = 0;

    double video_frame_rate{60};
    double save_frame_rate{30};
    while (!glfwWindowShouldClose(window)) {
        double prev_save_time = 0.0;
        double prev_video_time = 0.0;
        std::array<mjtNum, 7> ee_pose{};
        double prev_gripper_ctrl = gripper_ctrl;
        std::array<double, 6> delta{};
        saver.new_episode();

        while (!reset_episode && !glfwWindowShouldClose(window)) {
            // Reset episode
            if (reset_episode)
                break;
            // Translation
            double dx = 0, dy = 0, dz = 0;
            if (key_tx_pos)
                dx += EE_STEP;
            if (key_tx_neg)
                dx -= EE_STEP;
            if (key_ty_pos)
                dy += EE_STEP;
            if (key_ty_neg)
                dy -= EE_STEP;
            if (key_tz_pos)
                dz += EE_STEP;
            if (key_tz_neg)
                dz -= EE_STEP;

            // Rotation
            double drx = 0, dry = 0, drz = 0;
            if (key_rx_pos)
                drx += ROT_STEP;
            if (key_rx_neg)
                drx -= ROT_STEP;
            if (key_ry_pos)
                dry += ROT_STEP;
            if (key_ry_neg)
                dry -= ROT_STEP;
            if (key_rz_pos)
                drz += ROT_STEP;
            if (key_rz_neg)
                drz -= ROT_STEP;

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
            if ((mj_data->time - prev_video_time) > (1.0 / video_frame_rate)) {
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
                         tasks[task_idx % 3].c_str(),
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
                if ((mj_data->time - prev_save_time) >= (1.0 / save_frame_rate)) {
                    std::copy(delta.begin(), delta.end(), ee_pose.begin());
                    ee_pose[6] = gripper_ctrl - prev_gripper_ctrl;
                    prev_gripper_ctrl = gripper_ctrl;

                    std::array<mjtNum, 8> state{mj_data->ctrl[0],
                                                mj_data->ctrl[1],
                                                mj_data->ctrl[2],
                                                mj_data->ctrl[3],
                                                mj_data->ctrl[4],
                                                mj_data->ctrl[5],
                                                mj_data->ctrl[6],
                                                mj_data->ctrl[7]};
                    saver.write_data(main_rgb,
                                     wrist_rgb,
                                     main_depth,
                                     wrist_depth,
                                     half_W,
                                     half_H,
                                     ee_pose,
                                     state,
                                     tasks[task_idx % 3]);
                    prev_save_time = mj_data->time;
                }

                prev_video_time = mj_data->time;
                glfwSwapBuffers(window);
            }
            glfwPollEvents();
        }
        task_idx += 1;
        reset_episode = false;
        resetEpisode();
    }

    // Cleanup
    saver.close();
    // mjv_freeScene(&scn);
    // mjr_freeContext(&con);
    // mj_deleteData(mj_data);
    // mj_deleteModel(mj_model);
    // mj_deleteVFS(&vfs);
    // glfwTerminate();

    return 0;
}
