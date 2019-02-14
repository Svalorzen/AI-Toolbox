#include <AIToolbox/MDP/Experience.hpp>

#include <algorithm>

namespace AIToolbox::MDP {
    Experience::Experience(const size_t s, const size_t a) :
            S(s), A(a), visits_(boost::extents[S][A][S]), visitsSum_(boost::extents[S][A]),
            rewards_(boost::extents[S][A][S]), rewardsSum_(boost::extents[S][A]) {}

    void Experience::record(const size_t s, const size_t a, const size_t s1, const double rew) {
        visits_[s][a][s1]   += 1;
        visitsSum_[s][a]    += 1;

        rewards_[s][a][s1]  += rew;
        rewardsSum_[s][a]   += rew;
    }

    void Experience::reset() {
        std::fill(visits_.data(), visits_.data() + visits_.num_elements(), 0ul);
        std::fill(visitsSum_.data(), visitsSum_.data() + visitsSum_.num_elements(), 0ul);

        std::fill(rewards_.data(), rewards_.data() + rewards_.num_elements(), 0.0);
        std::fill(rewardsSum_.data(), rewardsSum_.data() + rewardsSum_.num_elements(), 0.0);
    }

    unsigned long Experience::getVisits(const size_t s, const size_t a, const size_t s1) const {
        return visits_[s][a][s1];
    }

    unsigned long Experience::getVisitsSum(const size_t s, const size_t a) const {
        return visitsSum_[s][a];
    }

    double Experience::getReward(const size_t s, const size_t a, const size_t s1) const {
        return rewards_[s][a][s1];
    }

    double Experience::getRewardSum(const size_t s, const size_t a) const {
        return rewardsSum_[s][a];
    }

    const Experience::VisitTable & Experience::getVisitTable() const {
        return visits_;
    }

    const Experience::RewardMatrix & Experience::getRewardMatrix() const {
        return rewards_;
    }

    size_t Experience::getS() const {
        return S;
    }

    size_t Experience::getA() const {
        return A;
    }
}
