#ifndef AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE
#define AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Pruner.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/WitnessLP_lpsolve.hpp>
// #include <AIToolbox/POMDP/Algorithms/Utils/WitnessLP_clp.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>

#include <AIToolbox/ProbabilityUtils.hpp>

#include <limits>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This class implements the Incremental Pruning algorithm.
         *
         * This algorithm solves a POMDP Model perfectly. It computes solutions
         * for each horizon incrementally, every new solution building upon the
         * previous one.
         *
         * From each solution, it computes the full set of possible
         * projections. It then computes all possible cross-sums of such
         * projections, in order to compute all possible vectors that can be
         * included in the final solution.
         *
         * What makes this method unique is its pruning strategy. Instead of
         * generating every possible vector, combining them and pruning, it
         * tries to prune at every possible occasion in order to minimize the
         * number of possible vectors at any given time. Thus it will prune
         * after creating the projections, after every single cross-sum, and
         * in the end when combining all projections for each action.
         *
         * The performances of this method are *heavily* dependent on the linear
         * programming methods used. In particular, this code currently
         * utilizes the lp_solve55 library. However, this library is not the
         * most efficient implementation, as it defaults to a somewhat slow
         * solver, and its problem-building API also tends to be slow due to
         * lots of bounds checking (which are cool, but sometimes people know
         * what they are doing). Still, to avoid replicating infinite amounts
         * of code and managing memory by ourselves, we use its API. It would
         * be nice if one day we could port directly into the code a fast lp
         * implementation; for now we do what we can.
         */
        class IncrementalPruning {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * This constructor sets the default horizon used to solve a POMDP::Model.
                 *
                 * The epsilon parameter must be >= 0.0, otherwise the
                 * constructor will throw an std::runtime_error. The epsilon
                 * parameter sets the convergence criterion. An epsilon of 0.0
                 * forces IncrementalPruning to perform a number of iterations
                 * equal to the horizon specified. Otherwise, IncrementalPruning
                 * will stop as soon as the difference between two iterations
                 * is less than the epsilon specified.
                 *
                 * @param h The horizon chosen.
                 * @param epsilon The epsilon factor to stop the value iteration loop.
                 */
                IncrementalPruning(unsigned h, double epsilon);

                /**
                 * @brief This function sets the epsilon parameter.
                 *
                 * The epsilon parameter must be >= 0.0, otherwise the
                 * constructor will throw an std::runtime_error. The epsilon
                 * parameter sets the convergence criterion. An epsilon of 0.0
                 * forces IncrementalPruning to perform a number of iterations
                 * equal to the horizon specified. Otherwise, IncrementalPruning
                 * will stop as soon as the difference between two iterations
                 * is less than the epsilon specified.
                 *
                 * @param e The new epsilon parameter.
                 */
                void setEpsilon(double e);

                /**
                 * @brief This function allows setting the horizon parameter.
                 *
                 * @param h The new horizon parameter.
                 */
                void setHorizon(unsigned h);

                /**
                 * @brief This function will return the currently set epsilon parameter.
                 *
                 * @return The currently set epsilon parameter.
                 */
                double getEpsilon() const;

                /**
                 * @brief This function returns the currently set horizon parameter.
                 *
                 * @return The current horizon.
                 */
                unsigned getHorizon() const;

                /**
                 * @brief This function solves a POMDP::Model completely.
                 *
                 * This function is pretty expensive (as are possibly all POMDP
                 * solvers).  It generates for each new solved timestep the
                 * whole set of possible ValueFunctions, and prunes it
                 * incrementally, trying to reduce as much as possible the
                 * linear programming solves required.
                 *
                 * This function returns a tuple to be consistent with MDP
                 * solving methods, but it should always succeed.
                 *
                 * @tparam M The type of POMDP model that needs to be solved.
                 *
                 * @param model The POMDP model that needs to be solved.
                 *
                 * @return A tuple containing a boolean value specifying whether
                 *         the specified epsilon bound was reached and the computed
                 *         ValueFunction.
                 */
                template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
                std::tuple<bool, ValueFunction> operator()(const M & model);

            private:
                /**
                 * @brief This function computes a VList composed of all possible combinations of sums of the VLists provided.
                 *
                 * This function performs the job of accumulating the
                 * information required to obtain the final policy. It assumes
                 * that the rhs List is being cross-summed to the lhs one, and
                 * not vice-versa. This is because the final result List will
                 * need to know which where the original VEntries that made up
                 * its particular sum. To do so, each cross-sum adds a single
                 * new parent. This function assumes that the new parent
                 * arrives from the rhs.
                 *
                 * @param l1 The "main" parent list.
                 * @param l2 The list being cross-summed to l1.
                 * @param a The action that this cross-sum is about.
                 * @param o The observation that generated the l2 list.
                 *
                 * @return The cross-sum between l1 and l2.
                 */
                VList crossSum(const VList & l1, const VList & l2, size_t a, bool order);

                size_t S, A, O;
                unsigned horizon_;
                double epsilon_;
        };

        template <typename M, typename>
        std::tuple<bool, ValueFunction> IncrementalPruning::operator()(const M & model) {
            // Initialize "global" variables
            S = model.getS();
            A = model.getA();
            O = model.getO();

            ValueFunction v(1, VList(1, makeVEntry(S))); // TODO: May take user input

            unsigned timestep = 0;

            Pruner<WitnessLP_lpsolve> prune(S);
            Projecter<M> projecter(model);

            bool useEpsilon = checkDifferentSmall(epsilon_, 0.0);
            double variation = epsilon_ * 2; // Make it bigger
            while ( timestep < horizon_ && ( !useEpsilon || variation > epsilon_ ) ) {
                ++timestep;

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

                    // Here we reduce at the minimum the cross-summing, by alternating
                    // merges. We pick matches like a reverse binary tree, so that
                    // we always pick lists that have been merged the least.
                    //
                    // Example for O==6:
                    //
                    //  0 <- 1    2 <- 3    4 <- 5    6
                    //  0 ------> 2         4 ------> 6
                    //            2 <---------------- 6

                    bool oddOld = O % 2;
                    int i, front = 0, back = O - oddOld, stepsize = 2, diff = 1, elements = O;
                    while ( elements > 1 ) {
                        for ( i = front; i != back; i += stepsize ) {
                            projs[a][i] = crossSum(projs[a][i], projs[a][i + diff], a, stepsize > 0);
                            prune(&projs[a][i]);
                            --elements;
                        }

                        bool oddNew = elements % 2;

                        int tmp   = back;
                        back      = front - ( oddNew ? 0 : stepsize );
                        front     = tmp   - ( oddOld ? 0 : stepsize );
                        stepsize *= -2;
                        diff     *= -2;

                        oddOld = oddNew;
                    }
                    // Put the result where we can find it
                    std::swap(projs[a][0], projs[a][front]);
                    finalWSize += projs[a][0].size();
                }
                VList w;
                w.reserve(finalWSize);

                // Here we don't have to do fancy merging since no cross-summing is involved
                for ( size_t a = 0; a < A; ++a )
                    std::move(std::begin(projs[a][0]), std::end(projs[a][0]), std::back_inserter(w));

                // We have them all, and we prune one final time to be sure we have
                // computed the parsimonious set of value functions.
                prune( &w );

                v.emplace_back(std::move(w));

                // Check convergence
                if ( useEpsilon ) {
                    variation = weakBoundDistance(v[timestep-1], v[timestep]);
                }
            }

            return std::make_tuple(variation <= epsilon_, v);
        }
    }
}

#endif
