#include <AIToolbox/Bandit/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/Bandit/Policies/Utils/QGreedyPolicyWrapper.hpp>

namespace AIToolbox::Bandit {
    QGreedyPolicy::QGreedyPolicy(const QFunction & q) : Base(q.size()), q_(q), bestActions_(A) {}

    size_t QGreedyPolicy::sampleAction() const {
        auto wrap = QGreedyPolicyWrapper(q_, bestActions_, rand_);
        return wrap.sampleAction();
    }

    double QGreedyPolicy::getActionProbability(const size_t & a) const {
        auto wrap = QGreedyPolicyWrapper(q_, bestActions_, rand_);
        return wrap.getActionProbability(a);
    }

    Vector QGreedyPolicy::getPolicy() const {
        auto wrap = QGreedyPolicyWrapper(q_, bestActions_, rand_);

        Vector retval{A};
        wrap.getPolicy(retval);

        return retval;
    }
}
