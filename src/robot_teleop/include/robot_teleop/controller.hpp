#pragma once

#include <array>

class Controller {
   public:
    virtual std::array<double, 6> get_deltas(bool key_tx_pos,
                                             bool key_tx_neg,
                                             bool key_ty_pos,
                                             bool key_ty_neg,
                                             bool key_tz_pos,
                                             bool key_tz_neg,
                                             double EE_STEP,
                                             double ROT_STEP) = 0;

    virtual ~Controller() = default;
};

class TeleoperationController : public Controller {
   public:
    std::array<double, 6> get_deltas(bool key_tx_pos,
                                     bool key_tx_neg,
                                     bool key_ty_pos,
                                     bool key_ty_neg,
                                     bool key_tz_pos,
                                     bool key_tz_neg,
                                     double EE_STEP,
                                     double ROT_STEP) override;

    ~TeleoperationController() override = default;
};
