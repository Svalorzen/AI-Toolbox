#ifndef AI_TOOLBOX_POMDP_AMDP_HEADER_FILE
#define AI_TOOLBOX_POMDP_AMDP_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
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
                 * @brief This function constructs an approximate *dense* MDP of the provided POMDP model.
                 *
                 * @tparam M The type of the POMDP model.
                 * @param model The POMDP model to be approximated.
                 *
                 * @return A tuple containing a dense MDP model which approximates the POMDP argument, and a function that converts a POMDP belief into a state of the MDP model.
                 */
                template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
                std::tuple<MDP::Model, Discretizer> discretizeDense(const M& model);

                /**
                 * @brief This function constructs an approximate *sparse* MDP of the provided POMDP model.
                 *
                 * @tparam M The type of the POMDP model.
                 * @param model The POMDP model to be approximated.
                 *
                 * @return A tuple containing a sparse MDP model which approximates the POMDP argument, and a function that converts a POMDP belief into a state of the MDP model.
                 */
                template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
                std::tuple<MDP::SparseModel, Discretizer> discretizeSparse(const M& model);

            private:
                Discretizer makeDiscretizer(size_t S);

                size_t beliefSize_, buckets_;
        };

        template <typename M, typename>
        std::tuple<MDP::Model, AMDP::Discretizer> AMDP::discretizeDense(const M& model) {
            size_t S = model.getS(), A = model.getA(), O = model.getO();
            size_t S1 = S * buckets_;

            BeliefGenerator<M> bGen(model);
            auto beliefs = bGen(beliefSize_);

            auto T = MDP::Model::TransitionTable   (A, Matrix2D::Zero(S1, S1));
            auto R = MDP::Model::RewardTable       (A, Matrix2D::Zero(S1, S1));

            auto discretizer = makeDiscretizer(S);

            Belief b1(S);
            for ( auto & b : beliefs ) {
                size_t s = discretizer(b);

                for ( size_t a = 0; a < A; ++a ) {
                    double r = beliefExpectedReward(model, b, a);

                    for ( size_t o = 0; o < O; ++o ) {
                        updateBeliefUnnormalized(model, b, a, o, &b1);
                        auto p = b1.sum();
                        if (checkDifferentSmall(0.0, p)) {
                            b1 /= p;
                            size_t s1 = discretizer(b1);

                            T[a](s, s1) += p;
                            R[a](s, s1) += p * r;
                        }
                    }
                }
            }

            for ( size_t a = 0; a < A; ++a )
                for ( size_t s = 0; s < S1; ++s ) {
                    for ( size_t s1 = 0; s1 < S1; ++s1 ) {
                        if ( checkDifferentSmall(0.0, T[a](s, s1)) )
                            R[a](s, s1) /= T[a](s, s1);
                    }
                    double sum = T[a].row(s).sum();
                    if ( checkEqualSmall(sum, 0.0) ) T[a](s, s) = 1.0;
                    else T[a].row(s) /= sum;
                }

            return std::make_tuple(MDP::Model::makeFromTrustedData(S1, A, std::move(T), std::move(R), model.getDiscount()), discretizer);
        }

        template <typename M, typename>
        std::tuple<MDP::SparseModel, AMDP::Discretizer> AMDP::discretizeSparse(const M& model) {
            size_t S = model.getS(), A = model.getA(), O = model.getO();
            size_t S1 = S * buckets_;

            BeliefGenerator<M> bGen(model);
            auto beliefs = bGen(beliefSize_);

            auto T = MDP::SparseModel::TransitionTable   (A, SparseMatrix2D(S1, S1));
            auto R = MDP::SparseModel::RewardTable       (A, SparseMatrix2D(S1, S1));

            auto discretizer = makeDiscretizer(S);

            Belief b1(S);
            for ( auto & b : beliefs ) {
                size_t s = discretizer(b);

                for ( size_t a = 0; a < A; ++a ) {
                    double r = beliefExpectedReward(model, b, a);

                    for ( size_t o = 0; o < O; ++o ) {
                        updateBeliefUnnormalized(model, b, a, o, &b1);
                        auto p = b1.sum();
                        if (checkDifferentSmall(0.0, p)) {
                            b1 /= p;
                            size_t s1 = discretizer(b1);

                            T[a].coeffRef(s, s1) += p;
                            R[a].coeffRef(s, s1) += p * r;
                        }
                    }
                }
            }

            for ( size_t a = 0; a < A; ++a )
                for ( size_t s = 0; s < S1; ++s ) {
                    for ( size_t s1 = 0; s1 < S1; ++s1 ) {
                        if ( checkDifferentSmall(0.0, T[a].coeff(s, s1)) )
                            R[a].coeffRef(s, s1) /= T[a].coeff(s, s1);
                    }
                    double sum = T[a].row(s).sum();
                    if ( checkEqualSmall(sum, 0.0) ) T[a].coeffRef(s, s) = 1.0;
                    else T[a].row(s) /= sum;
                }

            return std::make_tuple(MDP::SparseModel::makeFromTrustedData(S1, A, std::move(T), std::move(R), model.getDiscount()), discretizer);
        }
    }
}

#endif
