#include <AIToolbox/Bandit/Algorithms/RollingAverage.hpp>

#include <algorithm>

namespace AIToolbox::Bandit {
    RollingAverage::RollingAverage(const size_t A) : q_(A), M2s_(A), counts_(A) {
        q_.setZero();
        M2s_.setZero();
    }

    void RollingAverage::stepUpdateQ(size_t a, double rew) {
        // Count update
        ++counts_[a];

        const auto delta = rew - q_[a];
        // Rolling average for this bandit arm
        q_[a] += delta / counts_[a];
        // Rolling sum of square diffs.
        M2s_[a] += delta * (rew - q_[a]);
    }

    void RollingAverage::reset() {
        q_.setZero();
        M2s_.setZero();
        std::fill(std::begin(counts_), std::end(counts_), 0);
    }

    size_t RollingAverage::getA() const { return counts_.size(); }
    const QFunction & RollingAverage::getQFunction() const { return q_; }
    const std::vector<unsigned> & RollingAverage::getCounts() const { return counts_; }
    const Vector & RollingAverage::getM2s() const { return M2s_; }
}
