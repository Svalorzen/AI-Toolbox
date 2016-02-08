#include <AIToolbox/MDP/Policies/WoLFPolicy.hpp>

#include <AIToolbox/Utils.hpp>

namespace AIToolbox {
    namespace MDP {

        WoLFPolicy::WoLFPolicy(const QFunction & q, double deltaw, double deltal, double scaling) : QPolicyInterface(q),
                                                                                    deltaW_(deltaw), deltaL_(deltal), scaling_(scaling),
                                                                                    c_(S, 0),
                                                                                    avgPolicy_(S,A), actualPolicy_(S,A) {}

        void WoLFPolicy::updatePolicy(size_t s) {
            ++c_[s];

            // Update estimate of average policy
            auto avgstate = avgPolicy_.getStatePolicy(s);
            for ( size_t a = 0; a < A; ++a )
                avgstate[a] += (1.0/c_[s]) * ( actualPolicy_.getActionProbability(s,a) - avgstate[a] );

            avgPolicy_.setStatePolicy(s, avgstate);

            // Obtain argmax of Q[s], check whether we are losing or winning.
            auto actualstate  = actualPolicy_.getStatePolicy(s);

            size_t bestAction; double finalDelta;
            {
                unsigned bestActionCount = 1; double bestQValue = q_(s, 0);
                double avgValue    = avgstate[0]    * bestQValue;
                double actualValue = actualstate[0] * bestQValue;
                // Automatically sets initial best action as bestAction[0] = 0
                std::vector<size_t> bestActions(A,0);

                for ( size_t a = 1; a < A; ++a ) {
                    double qsa = q_(s, a);
                    avgValue        += avgstate[a]    * qsa;
                    actualValue     += actualstate[a] * qsa;

                    if ( checkEqualGeneral(qsa, bestQValue) ) {
                        bestActions[bestActionCount] = a;
                        ++bestActionCount;
                    }
                    else if ( qsa > bestQValue ) {
                        bestActions[0] = a;
                        bestActionCount = 1;
                        bestQValue = qsa;
                    }
                }

                auto pickDistribution = std::uniform_int_distribution<unsigned>(0, bestActionCount-1);
                unsigned selection = pickDistribution(rand_);

                bestAction = bestActions[selection];
                finalDelta = actualValue > avgValue ? deltaW_ : deltaL_;
            }

            finalDelta /= ( c_[s] / scaling_ + 1.0 );

            actualstate[bestAction] = std::min(1.0, actualstate[bestAction] + finalDelta);
            for ( size_t a = 0; a < A; ++a ) {
                if ( a == bestAction ) continue;
                actualstate[a] = std::max(0.0, actualstate[a] - finalDelta / ( A - 1 ) );
            }

            // Policy automatically normalizes this to 1
            actualPolicy_.setStatePolicy(s, actualstate);
        }

        size_t WoLFPolicy::sampleAction(const size_t & s) const {
            return actualPolicy_.sampleAction(s);
        }

        double WoLFPolicy::getActionProbability(const size_t & s, size_t a) const {
            return actualPolicy_.getActionProbability(s,a);
        }

        void WoLFPolicy::setDeltaW(double deltaW) {
            deltaW_ = deltaW;
        }

        double WoLFPolicy::getDeltaW() const {
            return deltaW_;
        }

        void WoLFPolicy::setDeltaL(double deltaL) {
            deltaL_ = deltaL;
        }

        double WoLFPolicy::getDeltaL() const {
            return deltaL_;
        }

        void WoLFPolicy::setScaling(double scaling) {
            scaling_ = scaling;
        }

        double WoLFPolicy::getScaling() const {
            return scaling_;
        }
    }
}
