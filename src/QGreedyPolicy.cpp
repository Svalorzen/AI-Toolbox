#include <AIToolbox/MDP/QGreedyPolicy.hpp>
#include <iostream>

namespace AIToolbox {
    namespace MDP {
        QGreedyPolicy::QGreedyPolicy(const QFunction & q) : QPolicyInterface(q) {}

        size_t QGreedyPolicy::sampleAction(size_t s) const {
            std::vector<unsigned> probs(A, 0);

            // This work is due to multiple max-valued actions
            double max = q_[s][0]; unsigned count = 1, sign = 0;
            for ( size_t a = 1; a < A; ++a ) {
                if ( q_[s][a] == max ) {
                    ++count;
                    probs[a] = sign;
                }
                else if ( q_[s][a] > max ) {
                    max = q_[s][a];
                    count = 1;
                    probs[a] = ++sign;
                }
            }

            // The multiplication avoids the need to normalize the whole probs vector
            auto pickDistribution = std::uniform_int_distribution<unsigned>(0, count-1);
            unsigned p = pickDistribution(rand_);
            for ( size_t a = 0; a < A; ++a ) {
                if ( probs[a] == sign && !p ) return a;
                --p;
            }
            std::cerr << "QGreedyPolicy missed a sampling.\n";
            // Make program fail in case this does not work.
            return 100000000000;
        }

        double QGreedyPolicy::getActionProbability(size_t s, size_t a) const {
            double max = q_[s][0]; unsigned count = 1;
            for ( size_t aa = 1; aa < A; ++aa ) {
                if ( q_[s][aa] == max ) ++count;
                else if ( q_[s][aa] > max ) {
                    max = q_[s][aa];
                    count = 1;
                }
            }
            // This can be weird with double math unfortunately..
            if ( q_[s][a] != max ) return 0.0;

            return 1.0 / count;
        }
    }
}
