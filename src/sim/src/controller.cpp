#include "sim/controller.hpp"

#include <array>

std::array<double, 6> TeleoperationController::get_deltas(bool key_tx_pos,
                                                          bool key_tx_neg,
                                                          bool key_ty_pos,
                                                          bool key_ty_neg,
                                                          bool key_tz_pos,
                                                          bool key_tz_neg,
                                                          bool key_rx_pos,
                                                          bool key_rx_neg,
                                                          bool key_ry_pos,
                                                          bool key_ry_neg,
                                                          bool key_rz_pos,
                                                          bool key_rz_neg,
                                                          double EE_STEP,
                                                          double ROT_STEP) {
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

    return std::array<double, 6>{dx, dy, dz, drx, dry, drz};
}
