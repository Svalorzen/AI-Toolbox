#ifndef AI_TOOLBOX_POMDP_PBVI_HEADER_FILE
#define AI_TOOLBOX_POMDP_PBVI_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Pruner.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>
#include <list>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This class implements the Point Based Value Iteration algorithm.
         *
         * The idea behind this algorithm is to solve a POMDP Model
         * approximately. When computing a perfect solution, the main problem
         * is pruning the resulting ValueFunction in order to contain only a
         * parsimonious representation. What this means is that many vectors
         * inside can be dominated by others, and so they do not add any
         * additional information, while at the same time occupying memory and
         * computational time.
         *
         * The way this method tries to fix the problem is by solving the Model
         * in a set of specified Beliefs. Doing so results in no need for
         * pruning at all, since every belief uniquely identifies one of the
         * optimal solution vectors (only uniqueness in the final set is
         * required, but it is way cheaper than linear programming).
         *
         * The set of Beliefs are stochastically computed as to cover as much
         * as possible of the belief space, to ensure minimization of the final
         * error. The final solution will thus be correct 100% in the Beliefs
         * that have been selected, and will (possibly) overshoot in
         * non-covered Beliefs.
         *
         * In addition, the fact that we solve only for a fixed set of Beliefs
         * guarantees that our final solution is limited in size, which is
         * useful since even small POMDP true solutions can explode in size
         * with high horizons, for very little gain.
         *
         * There is no convergence guarantee of this method, but the error is
         * bounded.
         */
        class PBVI {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * This constructor sets the default horizon used to solve a POMDP::Model
                 * and the number of beliefs used to approximate the ValueFunction.
                 *
                 * @param nBeliefs The number of support beliefs to use.
                 * @param h The horizon chosen.
                 */
                PBVI(size_t nBeliefs, unsigned h);

                /**
                 * @brief This function sets a new horizon parameter.
                 *
                 * @param h The new horizon parameter.
                 */
                void setHorizon(unsigned h);

                /**
                 * @brief This function sets a new number of support beliefs.
                 *
                 * @param nBeliefs The new number of support beliefs.
                 */
                void setBeliefSize(size_t nBeliefs);

                /**
                 * @brief This function returns the currently set horizon parameter.
                 *
                 * @return The current horizon.
                 */
                unsigned getHorizon() const;

                /**
                 * @brief This function returns the currently set number of support beliefs to use during a solve pass.
                 *
                 * @return The number of support beliefs.
                 */
                size_t getBeliefSize() const;

                /**
                 * @brief This function solves a POMDP::Model approximately.
                 *
                 * This function computes a set of beliefs for which to solve
                 * the input model. The beliefs are chosen stochastically,
                 * trying to cover as much as possible of the belief space in
                 * order to offer as precise a solution as possible. The final
                 * solution will only contain ValueFunctions for those Beliefs
                 * (so that in those points the solution will be 100% correct),
                 * and will interpolate them for points it did not solve for.
                 * Even though the resulting solution is approximate very often
                 * it is good enough, and this comes with an incredible
                 * increase in speed.
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
                using BeliefList = std::list<Belief>;

                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                void expandBeliefs(const M& model, BeliefList * bl) const;

                /**
                 * @brief This function computes a VList composed the maximized cross-sums with respect to the provided beliefs.
                 *
                 * This function performs the job of accumulating the
                 * information required to obtain the final policy. It
                 * processes an action at a time.
                 *
                 * For each belief contained in the argument BeliefList, it
                 * will create the optimal VEntry by cherry picking the best
                 * projections for each observation. Finally it prunes the
                 * resulting VList by removing duplicates.
                 *
                 * @param ProjectionsRow The type containing the projections to process.
                 * @param projs A 1d container containing O elements: each a VList of projections for the respective observation.
                 * @param a The action that this cross-sum is about.
                 * @param bl The beliefs for which we are trying to find VEntries.
                 *
                 * @return The optimal cross-sum list for the given projections and BeliefList.
                 */
                template <typename ProjectionsRow>
                VList crossSum(const ProjectionsRow & projs, size_t a, const BeliefList & bl);

                size_t S, A, O, beliefSize_;
                unsigned horizon_;

                mutable std::default_random_engine rand_;
        };

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        std::tuple<bool, ValueFunction> PBVI::operator()(const M & model) {
            // Initialize "global" variables
            S = model.getS();
            A = model.getA();
            O = model.getO();

            // In this implementation we compute all beliefs in advance. This
            // is mostly due to the fact that I prefer counter parameters (how
            // many beliefs do you want?) versus timers (loop until time is
            // up). However, this is easily changeable, since the function that
            // computes beliefs itself respects the interface defined in the
            // original PBVI paper (it tries to double the belief list given to
            // it).
            //
            // We add all simplex corners and the middle belief.
            BeliefList beliefs{Belief(S, 1.0/S)};
            for ( size_t s = 0; s < S; ++s ) {
                Belief b(S, 0.0); b[s] = 1.0;
                beliefs.emplace_back(std::move(b));
            }

            // Since the original method of obtaining beliefs is stochastic,
            // we keep trying for a while in case we don't find any new beliefs.
            // However, for some problems (for example the Tiger problem) still
            // this does not perform too well since the probability of finding
            // a new belief (via action LISTEN) is pretty low.
            size_t currentSize = beliefs.size(); unsigned counter = 0;
            while ( currentSize < beliefSize_ && counter < 30 ) {
                expandBeliefs(model, &beliefs);
                if ( currentSize == beliefs.size() ) ++counter;
                currentSize = beliefs.size();
            }

            ValueFunction v(1, VList(1, makeVEntry(S)));

            unsigned timestep = 1;

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
                for ( size_t a = 0; a < model.getA(); ++a ) {
                    projs[a][0] = crossSum( projs[a], a, beliefs );
                    finalWSize += projs[a][0].size();
                }
                VList w;
                w.reserve(finalWSize);

                for ( size_t a = 0; a < model.getA(); ++a )
                    std::move(std::begin(projs[a][0]), std::end(projs[a][0]), std::back_inserter(w));

                w.erase(extractDominated(S, std::begin(w), std::end(w)), std::end(w));
                v.emplace_back(std::move(w));

                ++timestep;
            }

            return std::make_tuple(true, v);
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        void PBVI::expandBeliefs(const M& model, BeliefList * blp) const {
            assert(blp);
            auto & bl = *blp;

            std::vector<Belief> newBeliefs(A);
            std::vector<double> distances(A);
            auto dBegin = std::begin(distances), dEnd = std::end(distances);

            // L1 distance
            auto computeDistance = [this](const Belief & lhs, const Belief & rhs) {
                double distance = 0.0;
                for ( size_t i = 0; i < S; ++i )
                    distance += std::abs(lhs[i] - rhs[i]);
                return distance;
            };

            Belief helper; double distance;
            // We apply the discovery process also to all beliefs we discover
            // along the way.
            for ( auto it = std::begin(bl); it != std::end(bl); ++it ) {
                // Compute all new beliefs
                for ( size_t a = 0; a < A; ++a ) {
                    distances[a] = 0.0;
                    for ( int j = 0; j < 20; ++j ) {
                        size_t s = sampleProbability(S, *it, rand_);

                        size_t o;
                        std::tie(std::ignore, o, std::ignore) = model.sampleSOR(s, a);
                        helper = updateBelief(model, *it, a, o);

                        // Compute distance (here we compare also against elements we just added!)
                        distance = computeDistance(helper, bl.front());
                        for ( auto jt = ++std::begin(bl); jt != std::end(bl); ++jt ) {
                            if ( checkEqualSmall(distance, 0.0) ) break; // We already have it!
                            distance = std::min(distance, computeDistance(helper, *jt));
                        }
                        // Select the best found over 20 times
                        if ( distance > distances[a] ) {
                            distances[a] = distance;
                            newBeliefs[a] = helper;
                        }
                    }
                }
                // Find furthest away, add only if it is new.
                size_t id = std::distance( dBegin, std::max_element(dBegin, dEnd) );
                if ( checkDifferentSmall(distances[id], 0.0) )
                    bl.emplace_back(std::move(newBeliefs[id]));
            }
        }

        template <typename ProjectionsRow>
        VList PBVI::crossSum(const ProjectionsRow & projs, size_t a, const BeliefList & bl) {
            VList result;
            result.reserve(bl.size());

            for ( auto & b : bl ) {
                MDP::Values v(S, 0.0);
                VObs obs(O, 0);

                // We compute the crossSum between each best vector for the belief.
                for ( size_t o = 0; o < O; ++o ) {
                    const VList & projsO = projs[o];
                    auto bestMatch = findBestAtBelief(std::begin(b), std::end(b), std::begin(projsO), std::end(projsO));

                    for ( size_t s = 0; s < S; ++s )
                        v[s] += std::get<VALUES>(*bestMatch)[s];

                    obs[o] = std::get<OBS>(*bestMatch)[0];
                }
                result.emplace_back(std::move(v), a, std::move(obs));
            }
            result.erase(extractDominated(S, std::begin(result), std::end(result)), std::end(result));

            return result;
        }
    }
}

#endif
