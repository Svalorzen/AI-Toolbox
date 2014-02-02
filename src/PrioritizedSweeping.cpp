#include <AIToolbox/MDP/PrioritizedSweeping.hpp>
#include <cassert>
#include <string>

namespace AIToolbox {
    namespace MDP {
        bool PrioritizedSweeping::PriorityTupleLess::operator() (const PriorityQueueElement& arg1, const PriorityQueueElement& arg2) const
        {
            return std::get<0>(arg1) < std::get<0>(arg2);
        }

        PrioritizedSweeping::PrioritizedSweeping(size_t s, size_t a, double alpha, double discount, double theta, unsigned n) : DynaQInterface(s, a, alpha, discount, n), theta_(theta) {}

        void PrioritizedSweeping::stepUpdateQ(size_t s, size_t s1, size_t a, double rew, const QFunction & q) {
            double p = std::fabs( rew
                                  + discount_ * (*std::max_element(std::begin(q[s1]), std::end(q[s1])))
                                  - q[s][a] );

            auto stateActionPair = std::make_tuple(s, a);
            if ( p > theta_ ) {
                auto it = queueHandles_.find(stateActionPair);
                if ( it != std::end(queueHandles_) ) {
                    if ( std::get<0>(*(it->second)) < p ) queue_.increase(it->second, std::make_tuple(p, s, a));
                }
                else {
                    queueHandles_[stateActionPair] = queue_.emplace(p, s, a);
                }
            }
        }

        void PrioritizedSweeping::batchUpdateQ(const RLModel & m, QFunction * q) {
            assert(q != nullptr);

            auto & ttable = m.getTransitionFunction();
            auto & rtable = m.getRewardFunction();

            for ( unsigned i = 0; i < N; ++i ) {
                if ( queue_.empty() ) return;

                size_t s, s1, a;
                double rew;
                std::tie(std::ignore, s, a) = queue_.top();

                queue_.pop();
                queueHandles_.erase(std::make_pair(s, a));

                std::tie(s1, rew) = m.sample(s, a);
                QLearning::stepUpdateQ(s, s1, a, rew, q);

                for ( size_t ss = 0; ss < S; ++ss ) {
                    for ( size_t aa = 0; aa < A; ++aa ) {
                        if ( ttable[ss][s][aa] > 0.0 ) {
                            stepUpdateQ(ss, s, aa, rtable[ss][s][aa], *q);
                        }
                    }
                }
            }
        }

        size_t PrioritizedSweeping::getQueueLength() const {
            return queue_.size();
        }
    }
}
