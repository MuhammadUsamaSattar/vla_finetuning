#pragma once

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <array>
#include <memory>
#include <thread>

#include "mujoco/mjmodel.h"
#include "mujoco/mjvisualize.h"
#include "sim/controller.hpp"
#include "sim/hdf5_saver.hpp"

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

class Sim {
   public:
    Sim(Controller& controller,
        int video_frame_rate,
        int save_frame_rate,
        std::array<std::string, 3> tasks,
        bool save);
    void setup_env(int video_frame_rate,
                   int save_frame_rate,
                   std::array<std::string, 3> tasks,
                   bool save);
    void run_sim();
    ~Sim();

   private:
    // Global sim objects
    mjModel* mj_model = nullptr;
    mjData* mj_data = nullptr;
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;
    mjVFS vfs;
    GLFWwindow* window;

    int task_idx{0};
    double video_frame_rate{};
    double save_frame_rate{};
    std::array<std::string, 3> tasks{};

    // Mouse state
    bool button_left = false;
    bool button_middle = false;
    bool button_right = false;
    double lastx{0}, lasty{0};

    // Translation keys (XYZ)
    static bool key_tx_pos, key_tx_neg;  // I/K → X
    static bool key_ty_pos, key_ty_neg;  // J/L → Y
    static bool key_tz_pos, key_tz_neg;  // U/P → Z

    // Rotation keys (XYZ)
    static bool key_rx_pos, key_rx_neg;  // W/S → X
    static bool key_ry_pos, key_ry_neg;  // A/D → Y
    static bool key_rz_pos, key_rz_neg;  // Q/E → Z

    // Gripper keys
    static bool key_gripper_open;   // F
    static bool key_gripper_close;  // H

    // EE control
    double gripper_ctrl = 255.0;
    static double EE_STEP;
    static double ROT_STEP;
    const static double EE_STEP_STEP;
    const static double ROT_STEP_STEP;
    const double GRIPPER_STEP = 0.5;
    double q_target[7] = {0, 0, 0, -1.57079, 0, 1.57079, -0.7853};

    // Reset episdoe
    static bool reset_episode;

    std::unique_ptr<HDF5Saver> saver{};
    Controller& controller;
    std::thread saver_thread;

    void loadMeshFilesToVFS(mjVFS& vfs, const std::string& assets_dir);

    std::array<double, 6> applyEEDelta(
        double dx, double dy, double dz, double drx, double dry, double drz);

    std::pair<std::vector<unsigned char>, std::vector<float>> renderCamera(
        const std::string_view cam_name, const mjrRect& rect);

    static void keyboard(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
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

    void resetEpisode();
    void get_model_and_data();
};
