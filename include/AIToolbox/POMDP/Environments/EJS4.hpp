#ifndef AI_TOOLBOX_POMDP_EJS4
#define AI_TOOLBOX_POMDP_EJS4

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This function returns a POMDP model of the ejs4 problem.
     */
    Model<MDP::Model> makeEJS4() {
        using PModel = Model<MDP::Model>;
        constexpr size_t S = 3, A = 2, O = 2;

        MDP::Model::TransitionMatrix t(A);
        MDP::Model::RewardMatrix r(S, A);
        PModel::ObservationMatrix o(A);

        for (size_t a = 0; a < A; ++a) {
            t[a] = AIToolbox::Matrix2D(S, S);
            o[a] = AIToolbox::Matrix2D(S, O);
        }
        t[0] <<
        0.1, 0.1, 0.8,
        0.2, 0.5, 0.3,
        0.7, 0.1, 0.2;

        t[1] <<
        0.1, 0.8, 0.1,
        0.7, 0.1, 0.2,
        0.1, 0.9, 0.0;

        o[0] <<
        0.7, 0.3,
        0.1, 0.9,
        0.4, 0.6;

        o[1] <<
        0.2, 0.8,
        0.4, 0.6,
        0.3, 0.7;

        r <<
        -1.0, 0.0,
        0.0,-1.0,
        0.0, 0.0;

        return PModel(AIToolbox::NO_CHECK, O, std::move(o), AIToolbox::NO_CHECK, S, A, std::move(t), std::move(r), 0.999);
    }
}

#endif
