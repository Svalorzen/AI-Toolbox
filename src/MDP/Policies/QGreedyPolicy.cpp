#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/Bandit/Policies/Utils/QGreedyPolicyWrapper.hpp>

namespace AIToolbox::MDP {
    QGreedyPolicy::QGreedyPolicy(const QFunction & q) :
            PolicyInterface::Base(q.rows(), q.cols()), QPolicyInterface(q), bestActions_(getA()) {}

    size_t QGreedyPolicy::sampleAction(const size_t & s) const {
        auto wrap = Bandit::QGreedyPolicyWrapper(q_.row(s), bestActions_, rand_);
        return wrap.sampleAction();
    }

    double QGreedyPolicy::getActionProbability(const size_t & s, const size_t & a) const {
        auto wrap = Bandit::QGreedyPolicyWrapper(q_.row(s), bestActions_, rand_);
        return wrap.getActionProbability(a);
    }

    Matrix2D QGreedyPolicy::getPolicy() const {
        Matrix2D retval(S, A);

        for (size_t s = 0; s < S; ++s) {
            auto wrap = Bandit::QGreedyPolicyWrapper(q_.row(s), bestActions_, rand_);
            wrap.getPolicy(retval.row(s));
        }

        return retval;
    }
}
