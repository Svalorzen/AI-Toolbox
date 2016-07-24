#ifndef AI_TOOLBOX_POMDP_PERSEUS_HEADER_FILE
#define AI_TOOLBOX_POMDP_PERSEUS_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/BeliefGenerator.hpp>

namespace AIToolbox {
    namespace POMDP {

        /**
         * @brief This class implements the PERSEUS algorithm.
         *
         * The idea behind this algorithm is very similar to PBVI. The thing
         * that changes is how beliefs are considered; in PERSEUS we only try
         * to find as little VEntries as possible as to ensure that all beliefs
         * considered are improved. This allows to skip generating VEntry for
         * most beliefs considered, since usually few VEntry are responsible
         * for supporting most of the beliefs.
         *
         * At the same time, this means that solutions found by PERSEUS may be
         * *extremely* approximate with respect to the true Value Functions. This
         * is because as long as the values for all the particle beliefs are
         * increased, no matter how slightly, the algorithm stops looking - in
         * effect simply guaranteeing that the worst action is never taken.
         * However for many problems the solution found is actually very good,
         * also given that due to the increased performance PERSEUS can do
         * many more iterations than, for example, PBVI.
         *
         * This method is works best when it is allowed to iterate until convergence,
         * and thus shouldn't be used on problems with finite horizons.
         */
        class PERSEUS {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * This constructor sets the default horizon/epsilon used to
                 * solve a POMDP::Model and the number of beliefs used to
                 * approximate the ValueFunction.
                 *
                 * @param nBeliefs The number of support beliefs to use.
                 * @param h The horizon chosen.
                 */
                PERSEUS(size_t nBeliefs, unsigned h, double epsilon);

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
                 * order to offer as precise a solution as possible.
                 *
                 * The final solution will try to be as small as possible, in
                 * order to drastically improve performances, while at the same
                 * time provide a reasonably good result.
                 *
                 * Note that the model input cannot have a discount of 1, due to
                 * how PERSEUS initializes the value function internally; if
                 * the model provided has a discount of 1 we throw.
                 *
                 * @tparam M The type of POMDP model that needs to be solved.
                 *
                 * @param model The POMDP model that needs to be solved.
                 * @param minReward The minimum reward obtainable from this model.
                 *
                 * @return True, and the computed ValueFunction up to the requested horizon.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                std::tuple<bool, ValueFunction> operator()(const M & model, double minReward);

            private:

                /**
                 * @brief This function computes a VList composed the maximized cross-sums with respect to the provided beliefs.
                 *
                 * This function performs the job of accumulating the
                 * information required to obtain the final policy. It
                 * processes all actions at once.
                 *
                 * For each belief it will check whether a VEntry which
                 * improves it from the previous timestep has already been
                 * found. If not, will create the optimal VEntry by cherry
                 * picking the best projections for each observation. Finally
                 * it prunes the resulting VList by removing duplicates.
                 *
                 * @param ProjectionsRow The type containing the projections to process.
                 * @param projs A 2d container containing AxO elements: each a VList of projections for the respective action-observation pair.
                 * @param bl The beliefs for which we are trying to find VEntries.
                 * @param oldV The previous timestep VList.
                 *
                 * @return The optimal cross-sum list for the given projections and BeliefList.
                 */
                template <typename ProjectionsTable>
                VList crossSum(const ProjectionsTable & projs, const std::vector<Belief> & bl, const VList & oldV);

                size_t S, A, O, beliefSize_;
                unsigned horizon_;
                double epsilon_;

                mutable std::default_random_engine rand_;
        };

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        std::tuple<bool, ValueFunction> PERSEUS::operator()(const M & model, double minReward) {
            if ( model.getDiscount() == 1 ) throw std::invalid_argument("The model cannot have a discount of 1 in PERSEUS!");
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

            // We initialize the ValueFunction to the "worst" case scenario.
            ValueFunction v(1, VList(1, std::make_tuple(MDP::Values(S, minReward / (1.0 - model.getDiscount())), 0, VObs(0))));

            unsigned timestep = 0;

            Projecter<M> projecter(model);

            // And off we go
            bool useEpsilon = checkDifferentSmall(epsilon_, 0.0);
            double variation = epsilon_ * 2; // Make it bigger
            while ( timestep < horizon_ && ( !useEpsilon || variation > epsilon_ ) ) {
                ++timestep;
                // Compute all possible outcomes, from our previous results.
                // This means that for each action-observation pair, we are going
                // to obtain the same number of possible outcomes as the number
                // of entries in our initial vector w.
                auto projs = projecter(v[timestep-1]);
                // Here we find the minimum number of VEntries that we need to improve
                // v on all beliefs from v[timestep-1].
                v.emplace_back( crossSum( projs, beliefs, v[timestep-1] ) );

                // Check convergence
                if ( useEpsilon ) {
                    variation = weakBoundDistance(v[timestep-1], v[timestep]);
                }
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
                if ( !start ) {
                    // If we have already improved this belief, skip it
                    findBestAtBelief( b, std::begin(result), std::end(result), &currentValue );
                    findBestAtBelief( b, std::begin(oldV),   std::end(oldV),   &oldValue     );
                    if ( currentValue >= oldValue ) continue;
                }
                helper.clear();
                for ( size_t a = 0; a < A; ++a ) {
                    MDP::Values v(S, 0.0);
                    VObs obs(O);

                    // We compute the crossSum between each best vector for the belief.
                    for ( size_t o = 0; o < O; ++o ) {
                        const VList & projsO = projs[a][o];
                        auto bestMatch = findBestAtBelief(b, std::begin(projsO), std::end(projsO));

                        v += std::get<VALUES>(*bestMatch);

                        obs[o] = std::get<OBS>(*bestMatch)[0];
                    }
                    helper.emplace_back(std::move(v), a, std::move(obs));
                }
                extractWorstAtBelief(b, std::begin(helper), std::begin(helper), std::end(helper));
                result.emplace_back(std::move(helper[0]));
                start = false;
            }
            result.erase(extractDominated(S, std::begin(result), std::end(result)), std::end(result));

            return result;
        }
    }
}

#endif
