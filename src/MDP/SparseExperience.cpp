#include <AIToolbox/MDP/SparseExperience.hpp>

namespace AIToolbox::MDP {
    SparseExperience::SparseExperience(const size_t s, const size_t a) :
            S(s), A(a), visits_(A, SparseTable2D(S, S)),
            visitsSum_(SparseTable2D(S, A)), rewards_(A, SparseMatrix2D(S, S)),
            rewardsSum_(SparseMatrix2D(S, A)) {}

    void SparseExperience::record(const size_t s, const size_t a, const size_t s1, const double rew) {
        visits_[a].coeffRef(s, s1)  += 1;
        visitsSum_.coeffRef(s, a)   += 1;

        rewards_[a].coeffRef(s, s1) += rew;
        rewardsSum_.coeffRef(s, a)  += rew;
    }

    void SparseExperience::reset() {
        for ( size_t a = 0; a < A; ++a ) {
            visits_[a].setZero();
            rewards_[a].setZero();
        }
        visitsSum_.setZero();
        rewardsSum_.setZero();
    }

    unsigned long SparseExperience::getVisits(const size_t s, const size_t a, const size_t s1) const {
        return visits_[a].coeff(s, s1);
    }

    unsigned long SparseExperience::getVisitsSum(const size_t s, const size_t a) const {
        return visitsSum_.coeff(s, a);
    }

    double SparseExperience::getReward(const size_t s, const size_t a, const size_t s1) const {
        return rewards_[a].coeff(s, s1);
    }

    double SparseExperience::getRewardSum(const size_t s, const size_t a) const {
        return rewardsSum_.coeff(s, a);
    }

    const SparseExperience::VisitTable & SparseExperience::getVisitTable() const {
        return visits_;
    }

    const SparseExperience::RewardMatrix & SparseExperience::getRewardMatrix() const {
        return rewards_;
    }

    size_t SparseExperience::getS() const {
        return S;
    }

    size_t SparseExperience::getA() const {
        return A;
    }
}
