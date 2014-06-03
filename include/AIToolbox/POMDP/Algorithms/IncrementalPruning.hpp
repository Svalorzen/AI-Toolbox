#ifndef AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE
#define AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE

#include <iostream>
#include <fstream>
#include <iomanip>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Pruner.hpp>

#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This class implements the Incremental Pruning algorithm.
         */
        class IncrementalPruning {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * This constructor sets the default horizon used to solve a POMDP::Model.
                 *
                 * @param h The horizon chosen.
                 */
                IncrementalPruning(unsigned h);

                /**
                 * @brief This function returns the currently set horizon parameter.
                 *
                 * @return The current horizon.
                 */
                unsigned getHorizon() const;

                /**
                 * @brief This function allows setting the horizon parameter.
                 *
                 * @param h The new horizon parameter.
                 */
                void setHorizon(unsigned h);

                /**
                 * @brief This function solves a POMDP::Model completely.
                 *
                 * This function is pretty expensive (as are possibly all POMDP solvers).
                 * It generates for each new solved timestep the whole set of possible ValueFunctions,
                 * and prunes it incrementally, trying to reduce as much as possible the linear
                 * programming solves required.
                 *
                 * This function returns a tuple to be consistent with MDP solving methods, but
                 * it should always succeed.
                 *
                 * @tparam M The type of POMDP model that needs to be solved.
                 *
                 * @param model The POMDP model that needs to be solved.
                 *
                 * @return True, and the computed ValueFunction up to the requested horizon.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                std::tuple<bool, ValueFunction> operator()(const M & model);

            private:
                using ProjectionsTable          = boost::multi_array<VList, 2>;
                using PossibleObservationsTable = boost::multi_array<bool,  2>;

                /**
                 * @brief This function computes all possible next-step ValueFunctions.
                 *
                 * This function in addition records into the VEntries produced which action
                 * and VEntry index were used, to allow for the final policy reconstruction.
                 *
                 * @tparam M The type of POMDP model that needs to be solved.
                 *
                 * @param model The POMDP model that needs to be solved.
                 * @param w The previous timestep VList.
                 * @param po A table containing which are the possible observations from specific actions.
                 * @param ir The immediate rewards from each state-action pair.
                 *
                 * @return A table containing, for each action-observation pair, the projected VEntries.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                ProjectionsTable makeAllProjections(const M & model, const VList & w, const PossibleObservationsTable & po, const Table2D & ir);

                /**
                 * @brief This function precomputes which observations are possible from specific actions.
                 *
                 * @tparam M The type of POMDP model that needs to be solved.
                 *
                 * @param model The POMDP model that needs to be solved.
                 *
                 * @return A boolean table.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                PossibleObservationsTable computePossibleObservations(const M& model);

                /**
                 * @brief This function precomputes immediate rewards for the POMDP state-action pairs.
                 *
                 * @tparam M The type of POMDP model that needs to be solved.
                 *
                 * @param model The POMDP model that needs to be solved.
                 *
                 * @return A reward table.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                Table2D computeImmediateRewards(const M & model);

                /**
                 * @brief This function computes a VList composed of all possible combinations of sums of the VLists provided.
                 *
                 * This function is in addition peculiar as it performs the job of accumulating
                 * the information required to obtain the final policy. It assumes that the
                 * rhs List is being cross-summed to the lhs one, and not vice-versa. This is
                 * because the final result List will need to know which where the original VEntries
                 * that made up its particular sum. To do so, each cross-sum adds a single new
                 * parent. This function assumes that the new parent arrives from the rhs.
                 *
                 * @param l1 The "main" parent list.
                 * @param l2 The list being cross-summed to l1.
                 * @param a The action that this cross-sum is about.
                 * @param o The observation that generated the l2 list.
                 *
                 * @return The cross-sum between l1 and l2.
                 */
                VList crossSum(const VList & l1, const VList & l2, size_t a, size_t o);

                size_t S, A, O;
                unsigned horizon_;
        };

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        std::tuple<bool, ValueFunction> IncrementalPruning::operator()(const M & model) {
            // Initialize "global" variables
            S = model.getS();
            A = model.getA();
            O = model.getO();

            auto possibleObservations = computePossibleObservations(model);
            auto immediateRewards = computeImmediateRewards(model);

            ValueFunction v(1, VList(1, makeVEntry(S))); // TODO: May take user input

            unsigned timestep = 1;

            Pruner prune(S);

            // And off we go
            while ( timestep <= horizon_ ) {
                // Compute all possible outcomes, from our previous results.
                // This means that for each action-observation pair, we are going
                // to obtain the same number of possible outcomes as the number
                // of entries in our initial vector w.
                auto projs = makeAllProjections(model, v[timestep-1], possibleObservations, immediateRewards);

                size_t finalWSize = 0;
                // In this method we split the work by action, which will then
                // be joined again at the end of the loop.
                for ( size_t a = 0; a < model.getA(); ++a ) {
                    // We prune each outcome separately to be sure
                    // we do not replicate work later.
                    for ( size_t o = 0; o < model.getO(); ++o ) {
                        prune( &projs[a][o] );
                    }

                    for ( size_t o = 1; o < model.getO(); ++o ) {
                        projs[a][0] = crossSum( projs[a][0], projs[a][o], a, o );
                        prune( &projs[a][0] );
                    }
                    finalWSize += projs[a][0].size();
                }
                VList w;
                w.reserve(finalWSize);

                for ( size_t a = 0; a < model.getA(); ++a )
                    std::move(std::begin(projs[a][0]), std::end(projs[a][0]), std::back_inserter(w));

                // We have them all, and we prune one final time to be sure we have
                // computed the parsimonious set of value functions.
                prune( &w );

                v.emplace_back(std::move(w));

                ++timestep;
            }

            return std::make_tuple(true, v);
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        IncrementalPruning::ProjectionsTable IncrementalPruning::makeAllProjections(const M & model, const VList & w,
                                                                                    const PossibleObservationsTable & possibleObservations,
                                                                                    const Table2D & immediateRewards)
        {
            size_t S = model.getS(), A = model.getA(), O = model.getO();
            double discount = model.getDiscount();

            ProjectionsTable projections( boost::extents[A][O] );

            for ( size_t a = 0; a < A; ++a ) {
                for ( size_t o = 0; o < O; ++o ) {
                    // Here we put in just the immediate rewards so that the cross-summing step in the main
                    // function works correctly. However we communicate via the boolean that pruning should
                    // not be done at this step (since adding constants shouldn't do anything anyway).
                    if ( !possibleObservations[a][o] ) {
                        MDP::Values vproj(S, 0.0);
                        for ( size_t s = 0; s < S; ++s )
                            vproj[s] += immediateRewards[a][s];
                        // We add a parent id anyway in order to keep the code that cross-sums simple. However
                        // note that this fake ID of 0 should never be used, so it should be safe to avoid
                        // setting it to a special value like -1. If one really wants to check, he/she can
                        // just look at the observation table and the belief and see if it makes sense.
                        projections[a][o].emplace_back(std::move(vproj), a, VObs(1,0));
                        continue;
                    }

                    // Otherwise we compute a projection for each ValueFunction supplied to us.
                    for ( size_t i = 0; i < w.size(); ++i ) {
                        auto & v = std::get<VALUES>(w[i]);
                        MDP::Values vproj(S, 0.0);
                        // For each value function in the previous timestep, we compute the new value
                        // if we performed action a and obtained observation o.
                        for ( size_t s = 0; s < S; ++s ) {
                            // vproj_{a,o}[s] = R(s,a) / |O| + discount * sum_{s'} ( T(s,a,s') * O(s',a,o) * v_{t-1}(s') )
                            for ( size_t s1 = 0; s1 < S; ++s1 )
                                vproj[s] += model.getTransitionProbability(s,a,s1) * model.getObservationProbability(s1,a,o) * v[s1];

                            vproj[s] *= discount;
                            vproj[s] += immediateRewards[a][s];
                        }
                        // Set new projection with found value and previous V id.
                        projections[a][o].emplace_back(std::move(vproj), a, VObs(1,i));
                    }
                }
            }
            return projections;
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        Table2D IncrementalPruning::computeImmediateRewards(const M & model) {
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
        boost::multi_array<bool,2> IncrementalPruning::computePossibleObservations(const M& model) {
            // Everything is false here.
            boost::multi_array<bool,2> isObsPossible( boost::extents[A][O] );

            for ( size_t a = 0; a < A; ++a )
                for ( size_t o = 0; o < O; ++o )
                    for ( size_t s = 0; s < S; ++s ) // This NEEDS to be last!
                        if ( checkDifferent(model.getObservationProbability(s,a,o), 0.0) ) { isObsPossible[a][o] = true; break; } // We only break the S loop!

            return isObsPossible;
        }
    }
}

#endif
