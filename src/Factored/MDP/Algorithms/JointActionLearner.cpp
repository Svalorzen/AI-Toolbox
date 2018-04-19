#include <AIToolbox/Factored/MDP/Algorithms/JointActionLearner.hpp>

namespace AIToolbox::Factored::MDP {
    JointActionLearner::JointActionLearner(const size_t s, Action a, const size_t i, const double d, const double al) :
            S(s), A(std::move(a)), id_(i),
            discount_(d), alpha_(al), timestep_(0),
            jointQFun_(s, factorSpace(A)), singleQFun_(s, A[id_]),
            jointActions_(A, id_)
    {
        counts_.resize(A.size());
        for (size_t j = 0; j < A.size(); ++j)
            if (j != id_) counts_.resize(A[j]);
    }

    void JointActionLearner::stepUpdateJointQ(const size_t s, const Action & a, const size_t s1, const double rew) {
        // Update counts
        ++timestep_;
        for (size_t j = 0; j < A.size(); ++j)
            if (j != id_) counts_[j][a[j]] += 1;

        // QLearning update
        const auto jointA = toIndex(A, a);
        jointQFun_(s, jointA) += alpha_ * ( rew + discount_ * jointQFun_.row(s1).maxCoeff() - jointQFun_(s, jointA) );
    }

    void JointActionLearner::stepUpdateSingleQ(size_t s) {
        updateSingleQFunction(s, s+1);
    }

    void JointActionLearner::stepUpdateSingleQ() {
        updateSingleQFunction(0, S);
    }

    void JointActionLearner::updateSingleQFunction(size_t begin, size_t end) {
        jointActions_.reset();

        const double t = static_cast<double>(timestep_);
        singleQFun_.middleRows(begin, end - begin + 1).fill(0.0);
        while (jointActions_.isValid()) {
            auto & jointAction = *jointActions_;

            double p = 1.0;
            for (size_t j = 0; j < A.size(); ++j)
                if (j != id_) p *= counts_[j][jointAction.second[j]] / t;

            for (size_t ai = 0; ai < A[id_]; ++ai) {
                jointAction.second[id_] = ai;
                const auto aa = toIndex(A, jointAction.second);

                for (size_t s = begin; s < end; ++s)
                    singleQFun_(s, ai) += jointQFun_(s, aa) * p;
            }
            jointActions_.advance();
        }
    }
}
