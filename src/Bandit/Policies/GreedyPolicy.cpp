#include <AIToolbox/Bandit/Policies/GreedyPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::Bandit {
    GreedyPolicy::GreedyPolicy(const size_t A) : Base(A), experience_(A), bestActions_(A) {}

    void GreedyPolicy::stepUpdateP(const size_t a, const double reward) {
        auto & [avg, tries] = experience_[a];
        // Rolling average for this bandit arm
        avg = (tries * avg + reward) / (tries + 1);
        ++tries;
    }

    size_t GreedyPolicy::sampleAction() const {
        // Automatically sets initial best action as bestAction[0] = 0
        bestActions_[0] = 0;

        // This work is due to multiple max-valued actions
        double bestValue = experience_[0].first; unsigned bestActionCount = 1;
        for ( size_t a = 1; a < A; ++a ) {
            auto & val = experience_[a].first;
            // The checkEqualGeneral is before the greater since we want to
            // trap here things that may be equal (even if one is a tiny bit
            // higher than the other).
            if ( checkEqualGeneral(val, bestValue) ) {
                bestActions_[bestActionCount] = a;
                ++bestActionCount;
            }
            else if ( val > bestValue ) {
                bestActions_[0] = a;
                bestActionCount = 1;
                bestValue = val;
            }
        }

        auto pickDistribution = std::uniform_int_distribution<unsigned>(0, bestActionCount-1);
        const unsigned selection = pickDistribution(rand_);

        return bestActions_[selection];
    }

    double GreedyPolicy::getActionProbability(const size_t & a) const {
        double max = experience_[0].first; unsigned count = 1;
        for ( size_t aa = 1; aa < A; ++aa ) {
            auto & val = experience_[aa].first;
            // The checkEqualGeneral is before the greater since we want to
            // trap here things that may be equal (even if one is a tiny bit
            // higher than the other).
            if ( checkEqualGeneral(val, max) ) ++count;
            else if ( val > max ) {
                max = val;
                count = 1;
            }
        }
        if ( checkDifferentGeneral(experience_[a].first, max) ) return 0.0;

        return 1.0 / count;
    }
}
