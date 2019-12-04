#ifndef AI_TOOLBOX_POMDP_BELIEF_GENERATOR_HEADER_FILE
#define AI_TOOLBOX_POMDP_BELIEF_GENERATOR_HEADER_FILE

#include <array>

#include <boost/container/flat_set.hpp>

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
            using SeenObservations = std::vector<boost::container::flat_set<std::pair<size_t, size_t>>>;

            /**
             * @brief This function uses the model to generate new Beliefs, and adds them to the provided list.
             *
             * WARNING: This function assumes that the input list has at least one element!
             *
             * @param max The maximum number of elements that the list should have.
             * @param randomBeliefsToAdd How many random beliefs to add to the looking set.
             * @param firstProductiveBelief The id of the first element assumed to be able to produce new Beliefs.
             */
            void expandBeliefList(size_t max, size_t randomBeliefsToAdd, size_t firstProductiveBelief) const;

            const M& model_;
            size_t S, A;

            mutable RandomEngine rand_;

            // Helpers and internal pointers.
            static constexpr unsigned triesPerRun_ = 20;
            static constexpr unsigned retryLimit_ = 5;
            static constexpr unsigned minProductiveBeliefs_ = 10;
            mutable BeliefList * blp_;
            mutable SeenObservations * sop_;
            mutable std::vector<unsigned> * up_; // unproductives
            mutable std::vector<double> * dp_; // distances
            mutable size_t goodBeliefsSize_, allBeliefsSize_, productiveBeliefs_;

            mutable Belief helper_;
    };

    template <typename M>
    BeliefGenerator<M>::BeliefGenerator(const M& model) :
            model_(model), S(model_.getS()), A(model_.getA()),
            rand_(Impl::Seeder::getSeed()),
            blp_(nullptr), sop_(nullptr), up_(nullptr), dp_(nullptr), helper_(S) {}

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
    void BeliefGenerator<M>::operator()(const size_t maxBeliefs, BeliefList * bl) const {
        if ( !bl ) return;

        // Initialize all helper storage.
        //
        // We have:
        //
        // - bl: The belief list, which will contain beliefs we ever find,
        //   divided into two groups: the good ones (which we will return), and
        //   the bad ones. The good ones are further subdivided into the
        //   unproductive ones (which we do not want to sample from anymore as
        //   they are unlikely to produce anything new), and the productive
        //   ones.
        // - seenObservations: This list contains, for each good belief, a list
        //   of action/observation pairs seen from it. This is used to avoid
        //   actually creating an updated belief if we have observed the a/o
        //   pair before.
        // - unproductiveBeliefs: This list tracks the number of times we have
        //   tried to expand a particular belief. After a certain number amount
        //   of times we give up and signal that it is unproductive.
        // - distances: This list contains, for each bad belief, its distance
        //   from the current good space. This is used to only pick the
        //   farthest beliefs when adding to the good set.
        blp_ = bl;
        auto & beliefs = *blp_;

        SeenObservations seenObservations;
        seenObservations.resize(beliefs.size());

        std::vector<unsigned> unproductiveBeliefs;
        unproductiveBeliefs.resize(beliefs.size());

        sop_ = &seenObservations;
        up_ = &unproductiveBeliefs;

        beliefs.reserve(maxBeliefs);
        seenObservations.reserve(maxBeliefs);
        unproductiveBeliefs.reserve(maxBeliefs);

        std::vector<double> distances;
        dp_ = &distances;

        // Since the original method of obtaining beliefs is stochastic,
        // we keep trying for a while in case we don't find any new beliefs.
        // However, for some problems (for example the Tiger problem) still
        // this does not perform too well since the probability of finding
        // a new belief (via action LISTEN) is pretty low.
        size_t firstProductiveBelief = 0;
        productiveBeliefs_ = goodBeliefsSize_ = allBeliefsSize_ = beliefs.size();

        unsigned randomBeliefsToAdd = 0;

        while ( goodBeliefsSize_ < maxBeliefs ) {
            expandBeliefList(maxBeliefs, randomBeliefsToAdd, firstProductiveBelief);
            if (goodBeliefsSize_ >= maxBeliefs) break;

            // Shift firstProductiveBelief to avoid checking the initial
            // non-productive beliefs every single time.
            for (size_t i = firstProductiveBelief; i < goodBeliefsSize_; ++i) {
                if (unproductiveBeliefs[i] < retryLimit_) break;
                else ++firstProductiveBelief;
            }

            // Fill the missing if needed with random beliefs so we always have new stuff.
            randomBeliefsToAdd = productiveBeliefs_ >= minProductiveBeliefs_ ? 0 : minProductiveBeliefs_ - productiveBeliefs_;
        }
        // Remove unused bad beliefs.
        beliefs.resize(maxBeliefs);
    }

    template <typename M>
    void BeliefGenerator<M>::expandBeliefList(const size_t max, const size_t randomBeliefsToAdd, const size_t firstProductiveBelief) const {
        auto & bl = *blp_;
        auto & seenObservations = *sop_;
        auto & unproductiveBeliefs = *up_;
        auto & distances = *dp_;

        // This is our optimistic estimate of how many beliefs we want to add
        // this run; should be one per productive belief, or at least one per
        // new random belief we are going to add.
        //
        // We refine this estimate later.
        auto beliefsToAdd = std::max(randomBeliefsToAdd, productiveBeliefs_);

        // L1 distance
        auto computeDistance = [](const Belief & lhs, const Belief & rhs) {
            return (lhs - rhs).cwiseAbs().sum();
        };

        // If we have some tentative test in the list, pop it off since we need
        // to emplace the random beliefs.
        if (allBeliefsSize_ < bl.size())
            bl.pop_back();

        // Add the required random beliefs, computing distances for each.
        for ( size_t i = 0; i < randomBeliefsToAdd; ++i) {
            bl.emplace_back(makeRandomProbability(S, rand_));
            ++allBeliefsSize_;

            // Compute distance for this belief.
            distances.push_back(std::numeric_limits<double>::max());
            for (size_t k = 0; k < goodBeliefsSize_; ++k) {
                distances.back() = std::min(distances.back(), computeDistance(bl.back(), bl[k]));
            }
        }

        // We apply the discovery process to all beliefs we have approved as
        // good. We start from the first productive one, since the others have
        // already produced as much as they can.
        for (size_t i = firstProductiveBelief; i < goodBeliefsSize_; ++i) {
            // Skip this belief if it is unproductive.
            auto & notFoundCounter = unproductiveBeliefs[i];
            if (notFoundCounter >= retryLimit_) continue;

            auto & beliefObservations = seenObservations[i];
            bool foundAnything = false;

            // Compute all new beliefs
            for ( size_t a = 0; a < A; ++a ) {
                updateBeliefPartial(model_, bl[i], a, &helper_);

                for (unsigned j = 0; j < triesPerRun_; ++j) {
                    // Sample state from belief, and generate an observation
                    // for it (given the current action)
                    const size_t s = sampleProbability(S, bl[i], rand_);

                    size_t o;
                    std::tie(std::ignore, o, std::ignore) = model_.sampleSOR(s, a);

                    // Check the new observation against the ones we have
                    // already produced for this belief. If we did, try again.
                    // Otherwise, mark it as seen.
                    if (beliefObservations.find({a,o}) != beliefObservations.end())
                        continue;

                    beliefObservations.insert({a,o});
                    foundAnything = true;

                    // Now we can update the belief. We use the last element of
                    // bl as temporary storage to avoid re-allocating when we
                    // don't need to.
                    if (allBeliefsSize_ == bl.size())
                        bl.emplace_back(S);

                    updateBeliefPartialNormalized(model_, helper_, a, o, &bl.back());

                    // Now check that the belief did not already exist in our
                    // list. If it did, we don't have to do anything else;
                    bool found = false;
                    for (size_t k = 0; k < allBeliefsSize_; ++k) {
                        if (checkEqualProbability(bl[k], bl.back())) {
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;

                    // Otherwise, the new belief is truly new. We keep it in
                    // the list and compute its distance. Note that we give an
                    // observation list only to the beliefs in the good set
                    // though (since we only sample those), so not yet to this
                    // one.
                    ++allBeliefsSize_;

                    distances.push_back(std::numeric_limits<double>::max());
                    for (size_t k = 0; k < goodBeliefsSize_; ++k) {
                        distances.back() = std::min(distances.back(), computeDistance(bl.back(), bl[k]));
                    }
                }
            }
            // We update the production counter for this belief, so we can skip
            // the ones which are not needed.
            if (!foundAnything) {
                ++notFoundCounter;
                // Mark it as unproductive if that's the case.
                if (notFoundCounter == retryLimit_)
                    --productiveBeliefs_;
            } else {
                notFoundCounter = 0;
            }
        }
        // Our optimistic estimane gets now scaled back by how many bad beliefs
        // we actually have to make into good.
        beliefsToAdd = std::min(beliefsToAdd, allBeliefsSize_ - goodBeliefsSize_);

        for (size_t i = 0; i < beliefsToAdd; ++i) {
            assert((allBeliefsSize_ - goodBeliefsSize_) == distances.size());

            // Find furthest away. It's guaranteed to be new, so we add it to
            // the good guys.
            //
            // We do a double swap here to avoid having to remove elements in
            // the middle of distances; just saving a slight amount of work.
            auto dBegin = std::begin(distances), dEnd = std::end(distances);
            size_t id = std::distance( dBegin, std::max_element(dBegin, dEnd) );

            // Move the good one at the end
            std::swap(distances[id], distances.back());
            std::swap(bl[goodBeliefsSize_ + id], bl[allBeliefsSize_ - 1]);

            // We finally move the new selected belief in the right spot.
            // Let's wait to do all the rest of the work until we know we have
            // to.
            std::swap(bl[goodBeliefsSize_], bl[allBeliefsSize_ - 1]);

            // Check if we are done.
            ++goodBeliefsSize_;
            if (goodBeliefsSize_ >= max) break;

            // If we are not done we:
            //
            // 1 - Remove the last entry from distances, which belonged to the
            //     now good belief.
            // 2 - Add a seenObservations entry for the belief, since we can
            //     now sample from it.
            // 3 - Recompute all remaining distances against the new good
            //     belief, as the space has changed.
            distances.pop_back();
            seenObservations.emplace_back();
            unproductiveBeliefs.emplace_back();
            ++productiveBeliefs_;

            for (size_t k = 0; k < distances.size(); ++k) {
                distances[k] = std::min(distances[k], computeDistance(bl[goodBeliefsSize_ - 1], bl[goodBeliefsSize_ + k]));
            }
        }
    }
}

#endif
