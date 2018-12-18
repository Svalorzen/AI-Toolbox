#include <AIToolbox/Bandit/Algorithms/RollingAverage.hpp>

#include <algorithm>

namespace AIToolbox::Bandit {
    RollingAverage::RollingAverage(const size_t A) : q_(A), counts_(A) {
        q_.setZero();
    }

    void RollingAverage::stepUpdateQ(size_t a, double rew) {
        // Rolling average for this bandit arm
        q_[a] = (counts_[a] * q_[a] + rew) / (counts_[a] + 1);
        ++counts_[a];
    }

    void RollingAverage::reset() {
        q_.setZero();
        std::fill(std::begin(counts_), std::end(counts_), 0);
    }

    size_t RollingAverage::getA() const { return counts_.size(); }
    const QFunction & RollingAverage::getQFunction() const { return q_; }
    const std::vector<unsigned> & RollingAverage::getCounts() const { return counts_; }
}
