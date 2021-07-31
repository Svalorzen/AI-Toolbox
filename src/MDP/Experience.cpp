#include <AIToolbox/MDP/Experience.hpp>

#include <algorithm>

namespace AIToolbox::MDP {
    Experience::Experience(const size_t s, const size_t a) :
            S(s), A(a), visits_(A, Table2D(S, S)), visitsSum_(S, A), rewards_(S, A), M2s_(S, A), timesteps_(0)
    {
        reset();
    }

    void Experience::record(const size_t s, const size_t a, const size_t s1, const double rew) {
        ++timesteps_;

        // Count updates
        visits_[a](s, s1) += 1;
        visitsSum_(s, a) += 1;

        const auto delta = rew - rewards_(s, a);
        // Rolling average for this s,a,s1 tuple
        rewards_(s, a) += delta / visitsSum_(s, a);
        // Rolling sum of square diffs.
        M2s_(s, a) += delta * (rew - rewards_(s, a));
    }

    void Experience::reset() {
        for (size_t a = 0; a < A; ++a)
            visits_[a].setZero();
        visitsSum_.setZero();
        rewards_.setZero();
        M2s_.setZero();
        timesteps_ = 0;
    }

    void Experience::setVisitsTable(const Table3D & v) {
        visits_ = v;
        visitsSum_.setZero();
        for ( size_t s = 0; s < S; ++s )
            for ( size_t a = 0; a < A; ++a )
                visitsSum_(s, a) = visits_[a].row(s).sum();
    }

    void Experience::setRewardMatrix(const Matrix2D & r) {
        rewards_ = r;
    }

    void Experience::setM2Matrix(const Matrix2D & mm) {
        M2s_ = mm;
    }

    unsigned long Experience::getTimesteps() const {
        return timesteps_;
    }

    unsigned long Experience::getVisits(const size_t s, const size_t a, const size_t s1) const {
        return visits_[a](s, s1);
    }

    unsigned long Experience::getVisitsSum(const size_t s, const size_t a) const {
        return visitsSum_(s, a);
    }

    double Experience::getReward(const size_t s, const size_t a) const {
        return rewards_(s, a);
    }

    double Experience::getM2(const size_t s, const size_t a) const {
        return M2s_(s, a);
    }

    const Table3D & Experience::getVisitsTable() const { return visits_; }
    const Table2D & Experience::getVisitsTable(const size_t a) const { return visits_[a]; }
    const Table2D & Experience::getVisitsSumTable() const { return visitsSum_; }
    const QFunction & Experience::getRewardMatrix() const { return rewards_; }
    const QFunction & Experience::getM2Matrix() const { return M2s_; }
    size_t Experience::getS() const { return S; }
    size_t Experience::getA() const { return A; }
}
