#ifndef AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE
#define AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Pruner.hpp>
#include <AIToolbox/POMDP/Algorithms/Projecter.hpp>

#include <AIToolbox/ProbabilityUtils.hpp>

#include <iostream>

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

            ValueFunction v(1, VList(1, makeVEntry(S))); // TODO: May take user input

            unsigned timestep = 1;

            Pruner prune(S);
            Projecter<M> projecter(model);

            // And off we go
            while ( timestep <= horizon_ ) {
                // Compute all possible outcomes, from our previous results.
                // This means that for each action-observation pair, we are going
                // to obtain the same number of possible outcomes as the number
                // of entries in our initial vector w.
                auto projs = projecter(v[timestep-1]);

                size_t finalWSize = 0;
                // In this method we split the work by action, which will then
                // be joined again at the end of the loop.
                for ( size_t a = 0; a < A; ++a ) {
                    // We prune each outcome separately to be sure
                    // we do not replicate work later.
                    for ( size_t o = 0; o < O; ++o )
                        prune( &projs[a][o] );

                    for ( size_t o = 1; o < O; ++o ) {
                        projs[a][0] = crossSum( projs[a][0], projs[a][o], a, o );
                        prune( &projs[a][0] );
                    }
                    finalWSize += projs[a][0].size();
                }
                VList w;
                w.reserve(finalWSize);

                for ( size_t a = 0; a < A; ++a )
                    std::move(std::begin(projs[a][0]), std::end(projs[a][0]), std::back_inserter(w));

                // We have them all, and we prune one final time to be sure we have
                // computed the parsimonious set of value functions.
                prune( &w );

                v.emplace_back(std::move(w));

                ++timestep;
            }

            return std::make_tuple(true, v);
        }

    }
}

#endif
