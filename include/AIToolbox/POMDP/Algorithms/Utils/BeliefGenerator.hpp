#ifndef AI_TOOLBOX_POMDP_BELIEF_GENERATOR_HEADER_FILE
#define AI_TOOLBOX_POMDP_BELIEF_GENERATOR_HEADER_FILE

#include <AIToolbox/ProbabilityUtils.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox {
    namespace POMDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_generative_model<M>::value>::type>
        class BeliefGenerator;
#endif
        /**
         * @brief This class generates reachable beliefs from a given Model.
         */
        template <typename M>
        class BeliefGenerator<M> {
            public:
                using BeliefList = std::vector<Belief>;

                /**
                 * @brief Basic constructor.
                 *
                 * @param model The Model used to generate beliefs.
                 */
                BeliefGenerator(const M& model);

                /**
                 * @brief This function tries to generate at least the specified input number of Beliefs.
                 *
                 * This function generates beliefs starting from the simplex
                 * corners of the Belief space, and goes from there.
                 *
                 * \sa operator()(size_t, BeliefList*) const;
                 *
                 * @param beliefNumber The number of beliefs to generate.
                 *
                 * @return A list containing the generated beliefs.
                 */
                BeliefList operator()(size_t beliefNumber) const;

                /**
                 * @brief This function tries to generate beliefs so that the input list has at least the specified number of elements.
                 *
                 * This function generates beliefs by simulating actions onto
                 * already generated beliefs, and sampling from the Model to
                 * try to obtain new Beliefs.
                 *
                 * Since the generation process is inherently stochastic, this
                 * function is not guaranteed to generate the specified number
                 * of Beliefs, depending on the probability to actually obtain
                 * a new Belief from the ones that have already been generated.
                 *
                 * @param beliefNumber The numbers of beliefs the input list should at least have.
                 * @param bl The list to add beliefs to.
                 */
                void operator()(size_t beliefNumber, BeliefList * bl) const;

            private:

                /**
                 * @brief This function uses the model to generate new beliefs, and adds them to the provided list.
                 *
                 * @param max The maximum number of elements that the list should have.
                 * @param bl The list to expand.
                 */
                void expandBeliefList(size_t max, BeliefList * bl) const;

                const M& model_;
                size_t S, A;

                mutable std::default_random_engine rand_;
        };

        template <typename M>
        BeliefGenerator<M>::BeliefGenerator(const M& model) :
                model_(model), S(model_.getS()), A(model_.getA()), rand_(Impl::Seeder::getSeed()) {}

        template <typename M>
        typename BeliefGenerator<M>::BeliefList BeliefGenerator<M>::operator()(const size_t beliefNumber) const {
            // We add all simplex corners and the middle belief.
            BeliefList beliefs; beliefs.reserve(std::max(beliefNumber, S));

            beliefs.emplace_back(S);
            beliefs.back().fill(1.0/S);

            for ( size_t s = 0; s < S && s < beliefNumber; ++s ) {
                beliefs.emplace_back(S);
                beliefs.back().fill(0.0); beliefs.back()(s) = 1.0;
            }

            this->operator()(beliefNumber, &beliefs);

            return beliefs;
        }

        template <typename M>
        void BeliefGenerator<M>::operator()(const size_t beliefNumber, BeliefList * bl) const {
            if ( !bl ) return;
            auto & beliefs = *bl;

            // Since the original method of obtaining beliefs is stochastic,
            // we keep trying for a while in case we don't find any new beliefs.
            // However, for some problems (for example the Tiger problem) still
            // this does not perform too well since the probability of finding
            // a new belief (via action LISTEN) is pretty low.
            size_t currentSize = beliefs.size();
            while ( currentSize < beliefNumber ) {
                unsigned counter = 0;
                while ( counter < 5 ) {
                    expandBeliefList(beliefNumber, &beliefs);
                    if ( currentSize == beliefs.size() ) ++counter;
                    else {
                        currentSize = beliefs.size();
                        if ( currentSize >= beliefNumber ) break;
                    }
                }
                for ( size_t i = 0; currentSize < beliefNumber && i < (beliefNumber/20); ++i, ++currentSize )
                    beliefs.emplace_back(makeRandomBelief(S, rand_));
            }
        }

        template <typename M>
        void BeliefGenerator<M>::expandBeliefList(const size_t max, BeliefList * blp) const {
            assert(blp);
            auto & bl = *blp;
            size_t size = bl.size();

            std::vector<Belief> newBeliefs(A);
            std::vector<double> distances(A);
            auto dBegin = std::begin(distances), dEnd = std::end(distances);

            // L1 distance
            const auto computeDistance = [this](const Belief & lhs, const Belief & rhs) {
                return (lhs - rhs).cwiseAbs().sum();
            };

            Belief helper(S); double distance;
            // We apply the discovery process also to all beliefs we discover
            // along the way.
            for ( auto it = std::begin(bl); it != std::end(bl); ++it ) {
                // Compute all new beliefs
                for ( size_t a = 0; a < A; ++a ) {
                    distances[a] = 0.0;
                    for ( int j = 0; j < 20; ++j ) {
                        const size_t s = sampleProbability(S, *it, rand_);

                        size_t o;
                        std::tie(std::ignore, o, std::ignore) = model_.sampleSOR(s, a);
                        updateBelief(model_, *it, a, o, &helper);

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
                if ( checkDifferentSmall(distances[id], 0.0) ) {
                    bl.emplace_back(std::move(newBeliefs[id]));
                    ++size;
                    if ( size >= max ) break;
                }
            }
        }
    }
}

#endif
