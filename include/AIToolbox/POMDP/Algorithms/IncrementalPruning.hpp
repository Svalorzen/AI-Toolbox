#ifndef AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE
#define AI_TOOLBOX_POMDP_INCREMENTAL_PRUNING_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Pruner.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/WitnessLP.hpp>
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
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
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
                VList crossSum(const VList & l1, const VList & l2, size_t a, size_t o);

                size_t S, A, O;
                unsigned horizon_;
                double epsilon_;
        };

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        std::tuple<bool, ValueFunction> IncrementalPruning::operator()(const M & model) {
            // Initialize "global" variables
            S = model.getS();
            A = model.getA();
            O = model.getO();

            ValueFunction v(1, VList(1, makeVEntry(S))); // TODO: May take user input

            unsigned timestep = 0;

            Pruner<WitnessLP_lpsolve> prune(S);
            Projecter<M> projecter(model);

            double variation = epsilon_ * 2; // Make it bigger

            bool useEpsilon = checkDifferentSmall(epsilon_, 0.0);
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

                // Check convergence
                if ( useEpsilon ) {
                    // Here we implement a weak bound (can also be seen in Cassandra's code)
                    // This is mostly because a strong bound is more costly (it requires performing
                    // multiple LPs) and also the code at the moment does not support it cleanly, so
                    // I prefer waiting until I have a good implementation of an LP class that hides
                    // complexity from here.
                    //
                    // The logic of the weak bound is the following: the variation between the old
                    // VList and the new one is equal to the maximum distance between a ValueFunction
                    // in the old VList with its closest match in the new VList. So the farthest from
                    // closest.
                    //
                    // We define distance between two ValueFunctions as the maximum between their
                    // element-wise difference.

                    MDP::Values helper(S); // We use this to compute differences.
                    auto hBegin = std::begin(helper), hEnd = std::end(helper);

                    variation = 0.0;
                    for ( auto & newVE : v[timestep] ) {
                        auto nBegin = std::begin(std::get<0>(newVE)), nEnd = std::end(std::get<0>(newVE));

                        double closestDistance = std::numeric_limits<double>::infinity();
                        for ( auto & oldVE : v[timestep-1] ) {
                            auto computeVariation = [](double lhs, double rhs) { return std::fabs(lhs - rhs); };
                            std::transform(nBegin, nEnd, std::begin(std::get<0>(oldVE)), hBegin, computeVariation );

                            // Compute the distance, we pick the max
                            double distance = *std::max_element(hBegin, hEnd);

                            // Keep the closest, we pick the min
                            closestDistance = std::min(closestDistance, distance);
                        }
                        variation = std::max(variation, closestDistance);
                    }
                }
            }

            return std::make_tuple(variation <= epsilon_, v);
        }

    }
}

#endif
