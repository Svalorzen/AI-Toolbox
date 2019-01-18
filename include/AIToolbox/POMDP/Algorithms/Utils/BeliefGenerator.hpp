#ifndef AI_TOOLBOX_POMDP_BELIEF_GENERATOR_HEADER_FILE
#define AI_TOOLBOX_POMDP_BELIEF_GENERATOR_HEADER_FILE

#include <array>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class generates reachable beliefs from a given Model.
     */
    template <typename M>
    class BeliefGenerator {
        static_assert(is_generative_model_v<M>, "This class only works for generative POMDP models!");

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
             * @brief This function uses the model to generate new Beliefs, and adds them to the provided list.
             *
             * WARNING: This function assumes that the input list has at least one element!
             *
             * @param max The maximum number of elements that the list should have.
             * @param firstProductiveBelief The id of the first element assumed to be able to produce new Beliefs.
             * @param bl The list to expand.
             */
            void expandBeliefList(size_t max, size_t firstProductiveBelief, BeliefList * bl) const;

            const M& model_;
            size_t S, A;
            mutable Belief helper1_, helper2_; // These are used to avoid reallocating memory all the time.

            mutable RandomEngine rand_;
    };

    template <typename M>
    BeliefGenerator<M>::BeliefGenerator(const M& model) :
            model_(model), S(model_.getS()), A(model_.getA()),
            helper1_(S), helper2_(S), rand_(Impl::Seeder::getSeed()) {}

    template <typename M>
    typename BeliefGenerator<M>::BeliefList BeliefGenerator<M>::operator()(const size_t beliefNumber) const {
        // We add all simplex corners and the middle belief.
        BeliefList beliefs; beliefs.reserve(std::max(beliefNumber, S));

        beliefs.emplace_back(S);
        beliefs.back().fill(1.0/S);

        for ( size_t s = 0; s < S && s < beliefNumber; ++s ) {
            beliefs.emplace_back(S);
            beliefs.back().setZero(); beliefs.back()(s) = 1.0;
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
        size_t firstProductiveBelief = 0;
        size_t bonusBeliefsToAdd = std::max((size_t)1, beliefNumber / 20);
        while ( currentSize < beliefNumber ) {
            unsigned counter = 0;
            while ( counter < 5 ) {
                expandBeliefList(beliefNumber, firstProductiveBelief, &beliefs);
                if ( currentSize == beliefs.size() ) ++counter;
                else {
                    counter = 0;
                    currentSize = beliefs.size();
                    if ( currentSize >= beliefNumber ) return;
                }
            }
            firstProductiveBelief = beliefs.size();
            for ( size_t i = 0; currentSize < beliefNumber && i < bonusBeliefsToAdd; ++i, ++currentSize )
                beliefs.emplace_back(makeRandomProbability(S, rand_));
        }
    }

    template <typename M>
    void BeliefGenerator<M>::expandBeliefList(const size_t max, const size_t firstProductiveBelief, BeliefList * blp) const {
        assert(blp);
        auto & bl = *blp;

        std::vector<Belief> newBeliefs(A);
        std::vector<double> distances(A);
        auto dBegin = std::begin(distances), dEnd = std::end(distances);

        // L1 distance
        const auto computeDistance = [this](const Belief & lhs, const Belief & rhs) {
            return (lhs - rhs).cwiseAbs().sum();
        };

        constexpr unsigned jMax = 20;
        std::array<size_t, jMax> observationBuffer;
        // We apply the discovery process also to all beliefs we discover
        // along the way. We start from the first good one, since the others
        // have already produced as much as they can.
        for (size_t i = firstProductiveBelief; i < bl.size(); ++i) {
            // Compute all new beliefs
            for ( size_t a = 0; a < A; ++a ) {
                double distance = std::numeric_limits<double>::max();
                distances[a] = 0.0;
                size_t bufferFill = 0;
                updateBeliefPartial(model_, bl[i], a, &helper1_);
                for (unsigned j = 0; j < jMax; ++j) {
                    const size_t s = sampleProbability(S, bl[i], rand_);

                    size_t o;
                    std::tie(std::ignore, o, std::ignore) = model_.sampleSOR(s, a);

                    // Check the new observation against the ones we have already
                    // produced this round. If it passes, add it to them.
                    bool pass = true;
                    for ( unsigned k = 0; k < bufferFill; ++k ) {
                        if (o == observationBuffer[k]) {
                            pass = false;
                            break;
                        }
                    }
                    if (!pass) continue;

                    // If we haven't had this observation before, we can update the belief.
                    observationBuffer[bufferFill++] = o;
                    updateBeliefPartialNormalized(model_, helper1_, a, o, &helper2_);

                    // Now check the new belief's distance against all others.
                    for (size_t k = 0; k < bl.size(); ++k) {
                        distance = std::min(distance, computeDistance(helper2_, bl[k]));
                        if (distance <= distances[a]) break;
                    }
                    // Select the best found over 20 times, or just set this one if
                    // we hadn't set it before (this may speed up generation as the
                    // one we have created gets instantly checked next round, possibly
                    // saving us a lot of work).
                    if ( distance > distances[a] || distances[a] == 0.0) {
                        distances[a] = distance;
                        newBeliefs[a] = helper2_;
                    }
                }
            }
            // Find furthest away, add only if it is new.
            size_t id = std::distance( dBegin, std::max_element(dBegin, dEnd) );
            if ( checkDifferentSmall(distances[id], 0.0) ) {
                bl.emplace_back(std::move(newBeliefs[id]));
                if ( bl.size() >= max ) break;
            }
        }
    }
}

#endif
