#include <AIToolbox/Bandit/Experience.hpp>

#include <algorithm>

namespace AIToolbox::Bandit {
    Experience::Experience(const size_t A) : q_(A), M2s_(A), counts_(A), timesteps_(0) {
        q_.setZero();
        M2s_.setZero();
    }

    void Experience::record(size_t a, double rew) {
        ++timesteps_;

        // Count update
        ++counts_[a];

        const auto delta = rew - q_[a];
        // Rolling average for this bandit arm
        q_[a] += delta / counts_[a];
        // Rolling sum of square diffs.
        M2s_[a] += delta * (rew - q_[a]);
    }

    void Experience::reset() {
        q_.setZero();
        M2s_.setZero();
        std::fill(std::begin(counts_), std::end(counts_), 0);
        timesteps_ = 0;
    }

    unsigned long Experience::getTimesteps() const { return timesteps_; }
    size_t Experience::getA() const { return counts_.size(); }
    const QFunction & Experience::getRewardMatrix() const { return q_; }
    const Experience::VisitsTable & Experience::getVisitsTable() const { return counts_; }
    const Vector & Experience::getM2Matrix() const { return M2s_; }
}
