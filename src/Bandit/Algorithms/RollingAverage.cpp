#include <AIToolbox/Bandit/Algorithms/RollingAverage.hpp>

namespace AIToolbox::Bandit {
    RollingAverage::RollingAverage(const size_t A) : q_(A), counts_(A) {
        q_.setZero();
    }

    void RollingAverage::stepUpdateQ(size_t a, double rew) {
        // Rolling average for this bandit arm
        q_[a] = (counts_[a] * q_[a] + rew) / (counts_[a] + 1);
        ++counts_[a];
    }

    size_t RollingAverage::getA() const { return counts_.size(); }
    const QFunction & RollingAverage::getQFunction() const { return q_; }
    const std::vector<unsigned> & RollingAverage::getCounts() const { return counts_; }
}
