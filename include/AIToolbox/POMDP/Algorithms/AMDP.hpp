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
        class AMDP {
            public:
                using Discretizer = std::function<size_t(const Belief&)>;

                AMDP(size_t nBeliefs, size_t entropyBuckets);

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

            auto ext = boost::extents[S1][A][S1];
            auto T = MDP::Model::TransitionTable   (ext);
            auto R = MDP::Model::RewardTable       (ext);

            // This is because lambdas are stupid and can't
            // capture member variables..
            auto buckets = buckets_ - 1;
            Discretizer discretizer = [S, buckets](const Belief & b) {
                static double stepSize = std::log(1.0/S) / static_cast<double>(buckets + 1);
                size_t maxS = 0;
                double entropy = 0.0;
                for ( size_t s = 0; s < S; ++s ) {
                    if ( b[s] > b[maxS] ) maxS = s;
                    if ( b[s] ) entropy += b[s] * std::log(b[s]);
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

                        T[s][a][s1] += p;
                        R[s][a][s1] += p * r;
                    }
                }
            }

            for ( size_t s = 0; s < S1; ++s )
                for ( size_t a = 0; a < A; ++a ) {
                    for ( size_t s1 = 0; s1 < S1; ++s1 )
                        if ( T[s][a][s1] ) R[s][a][s1] /= T[s][a][s1];
                    auto ref = T[s][a];
                    normalizeProbability(std::begin(ref), std::end(ref), std::begin(ref));
                }

            return std::make_tuple(MDP::Model(S1, A, T, R, model.getDiscount()), discretizer);
        }
    }
}

#endif
