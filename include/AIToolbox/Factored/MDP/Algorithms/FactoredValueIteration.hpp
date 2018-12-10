#ifndef AI_TOOLBOX_FACTORED_MDP_FACTORED_VALUE_ITERATION_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_FACTORED_VALUE_ITERATION_HEADER_FILE

#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace AIToolbox::Factored::MDP {
    // Performs the bellman equation on a single action
    inline FactoredVector bellmanEquation(const Factors & S, double discount, const BayesianNetwork & T, const FactoredVector & A, const Vector & w, const FactoredVector & R) {
        // Q = R + gamma * T * (A * w)
        FactoredVector Q = backProject(S, T, A * w);
        Q *= discount;
        return plusEqual(S, Q, R);
    }
}

#endif
