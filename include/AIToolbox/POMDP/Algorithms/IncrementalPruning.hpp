#ifndef AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE
#define AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE

#include <list>
#include <utility>

#include <iostream>

#include <lpsolve/lp_lib.h>

#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {

        using VList = std::vector<MDP::ValueFunction>;

        class IncrementalPruning {
            public:
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                void operator()(const M & model, unsigned horizon);

            private:
        };


        template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
        boost::multi_array<VList, 2> makeAllProjs(const M & model, const VList & w);

        void prune(VList * w);
        VList crossSum(const VList & a, const VList & b);

        void dominationPrune(VList * pw);
        VList extractBestAtCorners(VList * pw);

        size_t findBestVector(const Belief & belief, const VList & w, size_t start, size_t end);
        std::pair<bool, Belief> findWitnessPoint(const MDP::ValueFunction & v, const VList & best);

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        void IncrementalPruning::operator()(const M & model, unsigned horizon) {
            VList w;
            w.emplace_back(model.getS(), 0); // Or input

            unsigned timestep = 0;

            while ( timestep < horizon ) {
                // Compute all possible outcomes, from our previous results.
                // This means that for each action-observation pair, we are going
                // to obtain the same number of possible outcomes as the number
                // of entries in our initial vector w.
                auto projs = makeAllProjs(model, w);
                w.clear();
                size_t finalWSize = 0;
                // In this method we split the work by action, which will then
                // be joined again at the end of the loop.
                for ( size_t a = 0; a < model.getA(); ++a ) {
                    // We prune each outcome separately to be sure
                    // we do not replicate work later.
                    for ( size_t o = 0; o < model.getO(); ++o )
                        prune(&projs[a][o]);

                    for ( size_t o = 1; o < model.getO(); ++o ) {
                        projs[a][0] = crossSum( projs[a][0], projs[a][o] );
                        prune( &projs[a][0] );
                    }
                    finalWSize += projs[a][0].size();
                }
                w.reserve(finalWSize);

                for ( size_t a = 0; a < model.getA(); ++a )
                    std::move(std::begin(projs[a][0]), std::end(projs[a][0]), std::back_inserter(w));

                // We have them all, and we prune one final time to be sure we have
                // computed the parsimonious set of value functions.
                prune( &w );

                // We save them into the policy, which will them sample them to
                // obtain actions at runtime.

                // #### copy_into_policy(w, timestep); ####
                for ( auto & v : w ) {
                    for ( auto & s : v )
                        std::cout << "[" << s << "]";
                    std::cout << "\n";
                }

                ++timestep;
            }

            // Return Policy, V, Q?
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        boost::multi_array<VList, 2> makeAllProjs(const M & model, const VList & w) {
            size_t S = model.getS(), A = model.getA(), O = model.getO();
            double discount = model.getDiscount();

            boost::multi_array<VList, 2> projections( boost::extents[A][O] );

            static auto immediateRewards = computeImmediateRewards(model);
            static auto possibleObservations = computePossibleObservations(model);

            for ( size_t a = 0; a < A; ++a ) {
                for ( size_t o = 0; o < O; ++o ) {
                    // Here we put in just the immediate rewards so that the cross-summing step in the main
                    // function works correctly. However we communicate via the boolean that pruning should
                    // not be done at this step (since adding constants shouldn't do anything anyway).
                    if ( !possibleObservations[a][o] ) { projections[a][o].push_back(immediateRewards); continue; }

                    for ( auto & v : w ) {
                        // For each value function in the previous timestep, we compute the new value
                        // if we performed action a and obtained observation o.
                        MDP::ValueFunction vproj(S, 0.0);
                        for ( size_t s = 0; s < S; ++s ) {
                            // vproj_{a,o}[s] = R(s,a) / |O| + discount * sum_{s'} ( T(s,a,s') * O(s',a,o) * v_{t-1}(s') )
                            for ( size_t s1 = 0; s1 < S; ++s1 )
                                vproj[s] += model.getTransitionProbability(s,a,s1) * model.getObservationProbability(s1,a,o) * v[s1];

                            vproj[s] *= discount;
                            vproj[s] += immediateRewards[s] / O;
                        }
                        projections[a][o].push_back(v);
                    }
                }
            }
            return projections;
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        Table2D computeImmediateRewards(const M & model) {
            size_t S = model.getS(), A = model.getA(), O = model.getO();
            Table2D immediateRewards( boost::extents[A][S] );

            for ( size_t a = 0; a < A; ++a ) {
                for ( size_t s = 0; s < S; ++s ) {
                    for ( size_t s1 = 0; s1 < S; ++s1 )
                        immediateRewards[a][s] += model.getTransitionProbability(s,a,s1) * model.getExpectedReward(s,a,s1);

                    // You can find out why this is divided in the incremental pruning paper =)
                    // The idea is that at the end of all the cross sums it's going to add up to the correct value.
                    immediateRewards[a][s] /= static_cast<double>(O);
                }
            }
            return immediateRewards;
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        boost::multi_array<bool,2> computePossibleObservations(const M& model) {
            size_t S = model.getS(), A = model.getA(), O = model.getO();
            // Everything is false here.
            boost::multi_array<bool,2> isObsPossible( boost::extents[A][S] );

            for ( size_t a = 0; a < A; ++a )
                for ( size_t o = 0; o < O; ++o )
                    for ( size_t s = 0; s < S; ++s )
                        if ( checkDifferent(model.getObservationProbability(s,a,o), 0.0) ) { isObsPossible[a][o] = true; break; } // We only break the S loop!

            return isObsPossible;
        }
    }
}

#endif
