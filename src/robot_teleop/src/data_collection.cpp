#include <array>
#include <string>

#include "robot_teleop/controller.hpp"
#include "robot_teleop/mujoco_sim.hpp"

int main() {
    int video_frame_rate = 60;
    int save_frame_rate = 60;
    std::array<std::string, 3> tasks{"Pick the green cube and place it on the platform.",
                                     "Pick the blue cylinder and place it on the platform.",
                                     "Pick the red sphere and place it on the platform."};
    bool save = true;

    TeleoperationController controller;

    Sim sim{controller};
    sim.setup_env(video_frame_rate, save_frame_rate, tasks, save);
    sim.run_sim();

    return 0;
}
