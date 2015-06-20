#ifndef AI_TOOLBOX_POMDP_AMDP_HEADER_FILE
#define AI_TOOLBOX_POMDP_AMDP_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/BeliefGenerator.hpp>
#include <cmath>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This class implements the Augmented MDP algorithm.
         *
         * This algorithm transforms a POMDP into an approximately equivalent
         * MDP. This is done by extending the original POMDP statespace with
         * a discretized entropy component, which approximates a sufficient
         * statistic for the belief. In essence, AMDP builds states which
         * contain intrinsically information about the uncertainty of the agent.
         *
         * In order to compute a new transition and reward function, AMDP needs
         * to sample possible transitions at random, since each belief can
         * potentially update to any other belief. We sample beliefs using
         * the BeliefGenerator class which creates both random beliefs and
         * beliefs generated using the original POMDP model, in order to try
         * to obtain beliefs distributed in a way that better resembles the
         * original problem.
         *
         * Once this is done, it is simply a matter of taking each belief,
         * computing every possible new belief given an action and observation,
         * and sum up all possibilities.
         *
         * This class also bundles together with the resulting MDP a function
         * to convert an original POMDP belief into an equivalent AMDP state;
         * this is done so that a policy can be applied, observation gathered
         * and beliefs updated while continuing to use the approximated model.
         */
        class AMDP {
            public:
                using Discretizer = std::function<size_t(const Belief&)>;

                /**
                 * @brief Basic constructor.
                 *
                 * @param nBeliefs The number of beliefs to sample from when building the MDP model.
                 * @param entropyBuckets The number of buckets into which discretize entropy.
                 */
                AMDP(size_t nBeliefs, size_t entropyBuckets);


                /**
                 * @brief This function sets a new number of sampled beliefs.
                 *
                 * @param nBeliefs The new number of sampled beliefs.
                 */
                void setBeliefSize(size_t nBeliefs);

                /**
                 * @brief This function sets the new number of buckets in which to discretize the entropy.
                 *
                 * @param buckets The new number of buckets.
                 */
                void setEntropyBuckets(size_t buckets);

                /**
                 * @brief This function returns the currently set number of sampled beliefs.
                 *
                 * @return The number of sampled beliefs.
                 */
                size_t getBeliefSize() const;

                /**
                 * @brief This function returns the currently set number of entropy buckets.
                 *
                 * @return The number of entropy buckets.
                 */
                size_t getEntropyBuckets() const;

                /**
                 * @brief This function constructs and approximation of the provided POMDP model.
                 *
                 * @tparam M The type of the POMDP model.
                 * @param model The POMDP model to be approximated.
                 *
                 * @return A tuple containing an MDP model which approximate the POMDP argument, and a function that converts a POMDP belief into a state of the MDP model.
                 */
                template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
                std::tuple<MDP::Model, Discretizer> operator()(const M& model);

            private:
                size_t beliefSize_, buckets_;
        };

        template <typename M, typename>
        std::tuple<MDP::Model, AMDP::Discretizer> AMDP::operator()(const M& model) {
            size_t S = model.getS(), A = model.getA(), O = model.getO();
            size_t S1 = S * buckets_;

            BeliefGenerator<M> bGen(model);
            auto beliefs = bGen(beliefSize_);

            auto T = MDP::Model::TransitionTable   (A, Matrix2D::Zero(S1, S1));
            auto R = MDP::Model::RewardTable       (A, Matrix2D::Zero(S1, S1));

            // This is because lambdas are stupid and can't
            // capture member variables..
            auto buckets = buckets_ - 1;
            Discretizer discretizer = [S, buckets](const Belief & b) {
                // This stepsize is bounded by the minimum value entropy can take for a belief:
                // when the belief is uniform it would be: S * 1/S * log(1/S) = log(1/S)
                static double stepSize = std::log(1.0/S) / static_cast<double>(buckets + 1);
                size_t maxS = 0;
                double entropy = 0.0;
                for ( size_t s = 0; s < S; ++s ) {
                    if ( b[s] ) {
                        entropy += b[s] * std::log(b[s]);
                        if ( b[s] > b[maxS] ) maxS = s;
                    }
                }
                maxS += S * std::min(static_cast<size_t>(entropy / stepSize), buckets);
                return maxS;
            };

            for ( auto & b : beliefs ) {
                size_t s = discretizer(b);

                for ( size_t a = 0; a < A; ++a ) {
                    double r = beliefExpectedReward(model, b, a);

                    for ( size_t o = 0; o < O; ++o ) {
                        double p = beliefObservationProbability(model, b, a, o);
                        auto b1 = updateBelief(model, b, a, o);
                        size_t s1 = discretizer(b1);

                        T[a](s, s1) += p;
                        R[a](s, s1) += p * r;
                    }
                }
            }

            for ( size_t a = 0; a < A; ++a )
                for ( size_t s = 0; s < S1; ++s ) {
                    for ( size_t s1 = 0; s1 < S1; ++s1 )
                        if ( T[a](s, s1) ) R[a](s, s1) /= T[a](s, s1);
                    double sum = T[a].row(s).sum();
                    if ( checkEqualSmall(sum, 0.0) ) T[a](s, s) = 1.0;
                    else T[a].row(s) /= sum;
                }

            return std::make_tuple(MDP::Model(S1, A, T, R, model.getDiscount()), discretizer);
        }
    }
}

#endif
