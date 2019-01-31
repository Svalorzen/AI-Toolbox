#include <AIToolbox/Factored/MDP/Algorithms/JointActionLearner.hpp>

namespace AIToolbox::Factored::MDP {
    JointActionLearner::JointActionLearner(const size_t ss, Action aa, const size_t i, const double d, const double al) :
            A(std::move(aa)), id_(i),
            stateCounters_(ss, 0),
            stateActionCounts_(boost::extents[ss][A.size() - 1]),
            singleQFun_(ss, A[id_]),
            jointActions_(A, id_),
            qLearning_(ss, factorSpace(A), d, al)
    {
        singleQFun_.setZero();
        for (size_t s = 0; s < qLearning_.getS(); ++s) {
            for (size_t a = 0, i = 0; a < A.size() - 1; ++a, ++i) {
                if (a == id_) ++i;
                stateActionCounts_[s][a].resize(A[i]);
            }
        }
    }

    void JointActionLearner::stepUpdateQ(const size_t s, const Action & aa, const size_t s1, const double rew) {
        // Update counts
        stateCounters_[s] += 1;
        for (size_t a = 0, i = 0; a < A.size() - 1; ++a, ++i) {
            if (a == id_) ++i;
            stateActionCounts_[s][a][aa[i]] += 1;
        }

        // QLearning update
        const auto jointA = toIndex(A, aa);
        qLearning_.stepUpdateQ(s, jointA, s1, rew);

        // Single QFunction update
        jointActions_.reset();

        singleQFun_.row(s).setZero();
        while (jointActions_.isValid()) {
            auto & jointAction = *jointActions_;

            // Compute probability of other agents taking their actions
            // (this doesn't change even if our action changes).
            double p = 1.0;
            for (size_t a = 0, i = 0; a < A.size() - 1; ++a, ++i) {
                if (a == id_) ++i;
                p *= stateActionCounts_[s][a][jointAction.second[i]];
                p /= stateCounters_[s];
            }

            // Finally, update the row for the single QFunction.
            for (size_t ai = 0; ai < A[id_]; ++ai) {
                jointAction.second[id_] = ai;
                const auto aa = toIndex(A, jointAction.second);

                singleQFun_(s, ai) += qLearning_.getQFunction()(s, aa) * p;
            }
            jointActions_.advance();
        }
    }

    const AIToolbox::MDP::QFunction & JointActionLearner::getJointQFunction() const { return qLearning_.getQFunction(); }
    const AIToolbox::MDP::QFunction & JointActionLearner::getSingleQFunction() const { return singleQFun_; }
    void JointActionLearner::setLearningRate(double a) { qLearning_.setLearningRate(a); }
    double JointActionLearner::getLearningRate() const { return qLearning_.getLearningRate(); }
    void JointActionLearner::setDiscount(double d) { qLearning_.setDiscount(d); }
    double JointActionLearner::getDiscount() const { return qLearning_.getDiscount(); }
    size_t JointActionLearner::getS() const { return qLearning_.getS(); }
    const Action & JointActionLearner::getA() const { return A; }
    size_t JointActionLearner::getId() const { return id_; }
}
