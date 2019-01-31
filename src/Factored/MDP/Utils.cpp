#include <AIToolbox/Factored/MDP/Utils.hpp>

namespace AIToolbox::Factored::MDP {
    QFunction bellmanBackup(const CooperativeModel & m, const ValueFunction & v) {
        // Q = R + gamma * T * (A * w)
        // We rewrite it as T * A * (w * gamma) to save some multiplications.
        QFunction Q = backProject(m.getS(), m.getA(), m.getTransitionFunction(), v.values * (v.weights * m.getDiscount()));
        return plusEqual(m.getS(), m.getA(), Q, m.getRewardFunction());
    }
}
