#include <AIToolbox/Factored/MDP/Utils.hpp>

namespace AIToolbox::Factored::MDP {
    QFunction bellmanBackup(const CooperativeModel & m, const ValueFunction & v) {
        // Q = R + gamma * T * (A * w)
        QFunction Q = backProject(m.getS(), m.getA(), m.getTransitionFunction(), v.values * v.weights);
        Q *= m.getDiscount();
        return plusEqual(m.getS(), m.getA(), Q, m.getRewardFunction());
    }
}
