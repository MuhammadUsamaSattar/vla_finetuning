#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "sim/controller.hpp"
#include "sim/mujoco_sim.hpp"

std::map<std::string, bool> SAVE_OPTIONS{{"true", true}, {"false", false}};
std::map<std::string, std::function<std::unique_ptr<Controller>()>> CONTROLLER_OPTIONS{
    {"teleoperation", []() { return std::make_unique<TeleoperationController>(); }}};

struct Args {
    bool save = SAVE_OPTIONS.at("false");
    std::string controller = "teleoperation";
};

int main(int argc, char* argv[]) {
    Args args{};
    for (int i = 1; i < argc; i++) {
        std::string arg{std::string(argv[i])};

        if (arg == "--help") {
            std::cout << "sim.cpp\n\n";
            std::cout << "Simulation file to run MuJoCo simulation.\n\n";
            std::cout << "Args:\n";
            std::cout << "\t--save\t\tWhether to save data from the simulation. Default: "
                         "'false'. "
                         "Options: ";
            for (auto opt : SAVE_OPTIONS) {
                std::cout << "'" << opt.first << "' ";
            }
            std::cout << "\n";
            std::cout << "\t--controller\tController to use. Default: 'teleoperation'. Options: ";
            for (auto opt : CONTROLLER_OPTIONS) {
                std::cout << "'" << opt.first << "' ";
            }
            std::cout << "\n";
            return 0;
        } else if (arg == "--save") {
            if (++i < argc) {
                try {
                    args.save = SAVE_OPTIONS.at(std::string(argv[i]));
                } catch (const std::out_of_range&) {
                    throw std::runtime_error("--save requires either 'true' or 'false' value.");
                }
            }
        } else if (arg == "--controller") {
            if (++i < argc) {
                if (std::string(argv[i]) == "teleoperation") {
                    args.controller = "teleoperation";
                } else {
                    throw std::runtime_error("--controller requires 'teleoperation' value.");
                }
            }
        } else {
            throw std::runtime_error("Unrecognized option '" + arg + "'.\n");
        }
    }

    int video_frame_rate = 60;
    int save_frame_rate = 15;
    std::array<std::string, 3> tasks{"Pick the green cube and place it on the platform.",
                                     "Pick the blue cylinder and place it on the platform.",
                                     "Pick the red sphere and place it on the platform."};

    std::unique_ptr<Controller> controller;
    if (args.controller == "teleoperation") {
        controller = std::make_unique<TeleoperationController>();
    }

    Sim sim{*controller, video_frame_rate, save_frame_rate, tasks, args.save};
    sim.run_sim();

    return 0;
}
