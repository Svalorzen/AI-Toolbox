#ifndef AI_TOOLBOX_POMDP_PERSEUS_HEADER_FILE
#define AI_TOOLBOX_POMDP_PERSEUS_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/BeliefGenerator.hpp>

namespace AIToolbox {
    namespace POMDP {

        class PERSEUS {
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
                PERSEUS(size_t nBeliefs, unsigned h);

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
                template <typename ProjectionsTable>
                VList crossSum(const ProjectionsTable & projs, const std::vector<Belief> & bl, const VList & oldV);

                size_t S, A, O, beliefSize_;
                unsigned horizon_;

                mutable std::default_random_engine rand_;
        };

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        std::tuple<bool, ValueFunction> PERSEUS::operator()(const M & model) {
            // Initialize "global" variables
            S = model.getS();
            A = model.getA();
            O = model.getO();

            // In this implementation we compute all beliefs in advance. This
            // is mostly due to the fact that I prefer counter parameters (how
            // many beliefs do you want?) versus timers (loop until time is
            // up). However, this is easily changeable, since the belief generator
            // can be called multiple times to increase the size of the belief
            // vector.
            BeliefGenerator<M> bGen(model);
            auto beliefs = bGen(beliefSize_);

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

                v.emplace_back( crossSum( projs, beliefs, v[timestep-1] ) );

                ++timestep;
            }

            return std::make_tuple(true, v);
        }

        template <typename ProjectionsTable>
        VList PERSEUS::crossSum(const ProjectionsTable & projs, const std::vector<Belief> & bl, const VList & oldV) {
            VList result, helper;
            result.reserve(bl.size());
            helper.reserve(A);
            bool start = true;
            double currentValue, oldValue;

            for ( auto & b : bl ) {
                auto bbegin = std::begin(b), bend = std::end(b);

                if ( !start ) {
                    // If we have already improved this belief, skip it
                    findBestAtBelief( bbegin, bend, std::begin(result), std::end(result), &currentValue );
                    findBestAtBelief( bbegin, bend, std::begin(oldV),   std::end(oldV),   &oldValue     );
                    if ( currentValue >= oldValue ) continue;
                }
                helper.clear();
                for ( size_t a = 0; a < A; ++a ) {
                    MDP::Values v(S, 0.0);
                    VObs obs(O);

                    // We compute the crossSum between each best vector for the belief.
                    for ( size_t o = 0; o < O; ++o ) {
                        const VList & projsO = projs[a][o];
                        auto bestMatch = findBestAtBelief(std::begin(b), std::end(b), std::begin(projsO), std::end(projsO));

                        for ( size_t s = 0; s < S; ++s )
                            v[s] += std::get<VALUES>(*bestMatch)[s];

                        obs[o] = std::get<OBS>(*bestMatch)[0];
                    }
                    helper.emplace_back(std::move(v), a, std::move(obs));
                }
                extractWorstAtBelief(bbegin, bend, std::begin(helper), std::begin(helper), std::end(helper));
                result.emplace_back(std::move(helper[0]));
                start = false;
            }
            result.erase(extractDominated(S, std::begin(result), std::end(result)), std::end(result));

            return result;
        }
    }
}

#endif
