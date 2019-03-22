#ifndef AI_TOOLBOX_POMDP_CHENGD35
#define AI_TOOLBOX_POMDP_CHENGD35

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This function returns a POMDP model of the chengD35 problem.
     */
    Model<MDP::Model> makeChengD35() {
        using PModel = Model<MDP::Model>;
        constexpr size_t S = 3, A = 3, O = 3;

        MDP::Model::TransitionMatrix t(A);
        MDP::Model::RewardMatrix r(S, A);
        PModel::ObservationMatrix o(A);

        for (size_t a = 0; a < A; ++a) {
            t[a] = AIToolbox::Matrix2D(S, S);
            o[a] = AIToolbox::Matrix2D(S, O);
        }

        t[0] <<
        0.445, 0.222, 0.333,
        0.500, 0.173, 0.327,
        0.204, 0.553, 0.243;

        t[1] <<
        0.234, 0.064, 0.702,
        0.549, 0.218, 0.233,
        0.061, 0.466, 0.473;

        t[2] <<
        0.535, 0.313, 0.152,
        0.114, 0.870, 0.016,
        0.325, 0.360, 0.315;

        o[0] <<
        0.686, 0.182, 0.132,
        0.138, 0.786, 0.076,
        0.279, 0.083, 0.638;

        o[1] <<
        0.698, 0.131, 0.171,
        0.283, 0.624, 0.093,
        0.005, 0.202, 0.793;

        o[2] <<
        0.567, 0.234, 0.199,
        0.243, 0.641, 0.116,
        0.186, 0.044, 0.770;

        r <<
        5.2, 0.8, 9.0,
        4.6, 6.8, 9.3,
        4.1, 6.9, 0.8;

        return PModel(AIToolbox::NO_CHECK, O, std::move(o), AIToolbox::NO_CHECK, S, A, std::move(t), std::move(r), 0.999);
    }
}

#endif
