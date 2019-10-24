#include <AIToolbox/MDP/SparseExperience.hpp>

namespace AIToolbox::MDP {
    SparseExperience::SparseExperience(const size_t s, const size_t a) :
            S(s), A(a), visits_(A, SparseTable2D(S, S)), visitsSum_(S, A), rewards_(S, A), M2s_(S, A), timesteps_(0) {}

    void SparseExperience::record(const size_t s, const size_t a, const size_t s1, const double rew) {
        ++timesteps_;

        // Count updates
        visits_[a].coeffRef(s, s1) += 1;
        visitsSum_.coeffRef(s, a)  += 1;

        const auto delta = rew - rewards_.coeffRef(s, a);
        // Rolling average for this s,a,s1 tuple
        rewards_.coeffRef(s, a) += delta / visitsSum_.coeffRef(s, a);
        // Rolling sum of square diffs.
        M2s_.coeffRef(s, a) += delta * (rew - rewards_.coeffRef(s, a));
    }

    void SparseExperience::reset() {
        for ( size_t a = 0; a < A; ++a ) {
            visits_[a].setZero();
            visits_[a].makeCompressed();
        }
        visitsSum_.setZero();
        visitsSum_.makeCompressed();

        rewards_.setZero();
        rewards_.makeCompressed();

        M2s_.setZero();
        M2s_.makeCompressed();

        timesteps_ = 0;
    }

    unsigned long SparseExperience::getTimesteps() const {
        return timesteps_;
    }

    unsigned long SparseExperience::getVisits(const size_t s, const size_t a, const size_t s1) const {
        return visits_[a].coeff(s, s1);
    }

    unsigned long SparseExperience::getVisitsSum(const size_t s, const size_t a) const {
        return visitsSum_.coeff(s, a);
    }

    double SparseExperience::getReward(const size_t s, const size_t a) const {
        return rewards_.coeff(s, a);
    }

    double SparseExperience::getM2(const size_t s, const size_t a) const {
        return M2s_.coeff(s, a);
    }

    const SparseTable3D & SparseExperience::getVisitsTable() const { return visits_; }
    const SparseTable2D & SparseExperience::getVisitsTable(const size_t a) const { return visits_[a]; }
    const SparseTable2D & SparseExperience::getVisitsSumTable() const { return visitsSum_; }
    const SparseMatrix2D & SparseExperience::getRewardMatrix() const { return rewards_; }
    const SparseMatrix2D & SparseExperience::getM2Matrix() const { return M2s_; }
    size_t SparseExperience::getS() const { return S; }
    size_t SparseExperience::getA() const { return A; }
}
