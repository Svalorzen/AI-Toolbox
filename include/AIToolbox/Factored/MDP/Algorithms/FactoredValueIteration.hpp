#ifndef AI_TOOLBOX_FACTORED_MDP_FACTORED_VALUE_ITERATION_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_FACTORED_VALUE_ITERATION_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>
#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>
#include <AIToolbox/LP.hpp>

namespace AIToolbox::Factored::MDP {
    // TODO:
    //
    // So in order to make the thing work, let's say again what we need to do:
    //
    // - We need to define a factored matrix which can contain both R and
    //   Q, and that is possibly easily convertible to the Q form we have
    //   already been using (FactoredContainer<QFunctionRule>).
    //   ! It might make sense to really use that directly, as I don't think
    //   there;s any access pattern there..
    //   MMh. Note that we want the end result to be factored: Q = sum_j Q_j
    //   where each Q_j only has a small domain (which also allows us to easily
    //   sum R, since we just append it).
    //   So when we do VE we probably want to build the QFunctionRules on the
    //   fly based on what matches.. where we do the matches on the tags.
    // - We need to define a backpropagation for it. This needs to work also
    //   with the new FactoredDDN.
    // - We need to make the LP work with these things.
    //
    // Everything should work then.
    //
    // BONUS - Think how to make QGreedyPolicy work with both
    // FactoredContainer<QFunctionRule> and a Factored2DMatrix!

    // Performs the bellman equation on a single action
    template <typename BN>
    inline FactoredVector bellmanEquation(const State & S, double discount, const BN & T, const FactoredVector & A, const Vector & w, const FactoredVector & R) {
        // Q = R + gamma * T * (A * w)
        FactoredVector Q = backProject(S, T, A * w);
        Q *= discount;
        return plusEqual(S, Q, R);
    }

    // Performs the bellman equation on a single action
    inline Factored2DMatrix bellmanEquation(const CooperativeModel & m, const FactoredVector & V, const Vector & w) {
        // Q = R + gamma * T * (A * w)
        Factored2DMatrix Q = backProject(m.getS(), m.getA(), m.getTransitionFunction(), V * w);
        Q *= m.getDiscount();
        return plusEqual(m.getS(), m.getA(), Q, m.getRewardFunction());
    }


    // inline void makePolicy(const State & S, const CompactDDN & p, const std::vector<FactoredVector> & q, const FactoredVector & A, const Vector & w, const FactoredVector & R) {
    //     std::vector<std::tuple<PartialState, size_t, double>> retval;

    //     auto defaultQ = backProject(S, p.getDefaultTransition(), A * w);
    //     // defaultQ *= discount;

    //     const auto & x = p.getDiffNodes();

    //     for (size_t a = 0; a < q.size(); ++a) {
    //         std::vector<size_t> Ia;
    //         // Compute the bases for which we actually need to compare in order
    //         // to determine the delta of this action w.r.t. the default BN.
    //         for (size_t i = 0; i < A.bases.size(); ++i) {
    //             for (size_t j = 0; j < x[a].size(); ++j) {
    //                 if (sequential_sorted_contains(A.bases[i].tag, x[a][j].id)) {
    //                     Ia.push_back(i);
    //                     break;
    //                 }
    //             }
    //         }
    //     }
    // }
}

#endif
