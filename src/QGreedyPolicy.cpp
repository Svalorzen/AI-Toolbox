#include <AIToolbox/MDP/QGreedyPolicy.hpp>

namespace AIToolbox {
    namespace MDP {
        QGreedyPolicy::QGreedyPolicy(const QFunction & q) : QPolicyInterface(q) {}

        size_t QGreedyPolicy::sampleAction(size_t s) const {
            std::vector<unsigned> probs(A, 0);

            // This work is due to multiple max-valued actions
            double max = q_[s][0]; unsigned count = 0, sign = 0;
            for ( size_t a = 0; a < A; ++a ) {
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
            unsigned p = sampleDistribution_(rand_) * count;

            for ( size_t a = 0; a < A; ++a ) {
                if ( probs[a] == sign && !p ) return a;
                --p;
            }

            // Return last action just in case
            return A-1;
        }

        double QGreedyPolicy::getActionProbability(size_t s, size_t a) const {
            double max = q_[s][0]; unsigned count = 0;
            for ( size_t aa = 0; aa < A; ++aa ) {
                if ( q_[s][aa] == max ) ++count;
                else if ( q_[s][aa] > max ) {
                    max = q_[s][aa];
                    count = 1;
                }
            }

            if ( q_[s][a] != max ) return 0.0;

            return 1.0 / count;
        }
    }
}
