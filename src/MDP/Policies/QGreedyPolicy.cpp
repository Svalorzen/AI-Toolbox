#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/Utils.hpp>

namespace AIToolbox {
    namespace MDP {
        QGreedyPolicy::QGreedyPolicy(const QFunction & q) : QPolicyInterface(q) {}

        size_t QGreedyPolicy::sampleAction(const size_t & s) const {
            // Automatically sets initial best action as bestAction[0] = 0
            std::vector<unsigned> bestActions(A, 0);

            // This work is due to multiple max-valued actions
            double bestQValue = q_(s, 0); unsigned bestActionCount = 1;
            for ( size_t a = 1; a < A; ++a ) {
                if ( q_(s, a) > bestQValue ) {
                    bestActions[0] = a;
                    bestActionCount = 1;
                    bestQValue = q_(s, a);
                }
                else if ( checkEqualGeneral(q_(s, a), bestQValue) ) {
                    bestActions[bestActionCount] = a;
                    ++bestActionCount;
                }
            }

            auto pickDistribution = std::uniform_int_distribution<unsigned>(0, bestActionCount-1);
            const unsigned selection = pickDistribution(rand_);

            return bestActions[selection];
        }

        double QGreedyPolicy::getActionProbability(const size_t & s, const size_t & a) const {
            double max = q_(s, 0); unsigned count = 1;
            for ( size_t aa = 1; aa < A; ++aa ) {
                if ( checkEqualGeneral(q_(s, aa), max) ) ++count;
                else if ( q_(s, aa) > max ) {
                    max = q_(s, aa);
                    count = 1;
                }
            }
            if ( checkDifferentGeneral(q_(s, a), max) ) return 0.0;

            return 1.0 / count;
        }
    }
}
