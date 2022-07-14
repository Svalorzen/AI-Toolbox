#include <AIToolbox/Factored/MDP/Utils.hpp>

namespace AIToolbox::Factored::MDP {
    QFunction bellmanBackup(const CooperativeModel & m, const ValueFunction & v) {
        // Q = R + gamma * T * (A * w)
        // We rewrite it as T * A * (w * gamma) to save some multiplications.
        QFunction Q = backProject(m.getTransitionFunction(), v.values * (v.weights * m.getDiscount()));
        return plusEqual(m.getS(), m.getA(), Q, m.getRewardFunction());
    }

    QFunction makeQFunction(const DDNGraph & graph, const std::vector<std::vector<size_t>> & basisDomains) {
        QFunction qfun;
        qfun.bases.reserve(basisDomains.size());

        const auto & ps = graph.getParentSets();

        for (const auto & domain : basisDomains) {
            qfun.bases.emplace_back();
            auto & q = qfun.bases.back();

            for (const auto d : domain) {
                // Compute state-action domain for this Q factor.
                q.actionTag = merge(q.actionTag, ps[d].agents);
                for (const auto & n : ps[d].features)
                    q.tag = merge(q.tag, n);
            }

            // Initialize this factor's matrix.
            const size_t sizeA = factorSpacePartial(q.actionTag, graph.getA());
            const size_t sizeS = factorSpacePartial(q.tag, graph.getS());

            q.values.resize(sizeS, sizeA);
            q.values.setZero();
        }
        return qfun;
    }
}
