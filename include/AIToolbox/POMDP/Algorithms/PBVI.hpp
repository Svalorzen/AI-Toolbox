#ifndef AI_TOOLBOX_POMDP_PBVI_HEADER_FILE
#define AI_TOOLBOX_POMDP_PBVI_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Pruner.hpp>
#include <AIToolbox/POMDP/Algorithms/Projecter.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

#include <iostream>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This class implements the Point Based Value Iteration algorithm.
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
                using BeliefList                = std::vector<Belief>;

                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                void expandBeliefs(const M& model, BeliefList & bl) const;

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
            BeliefList beliefs(1, Belief(S, 0.0)); beliefs[0][0] = 1.0; // TODO: May take user input

            // Since the original method of obtaining beliefs is stochastic,
            // we keep trying for a while in case we don't find any new beliefs.
            // However, for some problems (for example the Tiger problem) still
            // this does not perform too well since the probability of finding
            // a new belief (via action LISTEN) is pretty low.
            size_t currentSize = beliefs.size(); unsigned counter = 0;
            while ( currentSize < beliefSize_ && counter < 30 ) {
                expandBeliefs(model, beliefs);
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

                v.emplace_back(std::move(w));

                ++timestep;
            }

            return std::make_tuple(true, v);
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        void PBVI::expandBeliefs(const M& model, BeliefList & bl) const {
            BeliefList newBeliefs(A);
            std::vector<double> distances(A);
            auto dBegin = std::begin(distances), dEnd = std::end(distances);

            size_t size = bl.size();
            bl.reserve(size * 2);

            // L1 distance
            auto computeDistance = [](const Belief & lhs, const Belief & rhs, size_t S) {
                double distance = 0.0;
                for ( size_t i = 0; i < S; ++i )
                    distance += std::abs(lhs[i] - rhs[i]);
                return distance;
            };

            for ( size_t i = 0; i < size; ++i ) {
                // Compute all new beliefs
                for ( size_t a = 0; a < A; ++a ) {
                    size_t s = sampleProbability(S, bl[i], rand_);

                    size_t o;
                    std::tie(std::ignore, o, std::ignore) = model.sampleSOR(s, a);
                    newBeliefs[a] = updateBelief(model, bl[i], a, o);

                    // Compute distance (here we compare also against elements we just added!)
                    distances[a] = computeDistance(newBeliefs[a], bl[0], S);
                    for ( size_t j = 1; j < bl.size(); ++j ) {
                        if ( checkEqual(distances[a], 0.0) ) break; // We already have it!
                        distances[a] = std::min(distances[a], computeDistance(newBeliefs[a], bl[j], S));
                    }
                }
                // Find furthest away, add only if it is new.
                size_t id = std::distance( dBegin, std::max_element(dBegin, dEnd) );
                if ( checkDifferent(distances[id], 0.0) )
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
                    auto bestMatch = findBestAtBelief(S, b, std::begin(projsO), std::end(projsO));

                    for ( size_t s = 0; s < S; ++s )
                        v[s] += std::get<VALUES>(*bestMatch)[s];

                    obs[o] = std::get<OBS>(*bestMatch)[0];
                }
                result.emplace_back(v, a, obs);
            }
            // We could probably do better than this but what the hell.
            Pruner::dominationPrune( S, &result );

            return result;
        }
    }
}

#endif
