#ifndef AI_TOOLBOX_POMDP_PRUNER_HEADER_FILE
#define AI_TOOLBOX_POMDP_PRUNER_HEADER_FILE

#include <utility>

#include <boost/iterator/transform_iterator.hpp>

#include <AIToolbox/Utils/Prune.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Types.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class offers pruning facilities for non-parsimonious ValueFunction sets.
     */
    template <typename WitnessLP>
    class Pruner {
        static_assert(is_witness_lp_v<WitnessLP>, "This class only works for linear programming witness classes!");

        public:
            Pruner(size_t S);

            /**
             * @brief This function prunes all non useful ValueFunctions from the provided VList.
             *
             * @param w The list that needs to be pruned.
             */
            void operator()(VList * w);

        private:
            size_t S;

            WitnessLP lp;
    };

    template <typename WitnessLP>
    Pruner<WitnessLP>::Pruner(const size_t s) : S(s), lp(s) {}

    // The idea is that the input thing already has all the best vectors,
    // thus we only need to find them and discard the others.
    template <typename WitnessLP>
    void Pruner<WitnessLP>::operator()(VList * pw) {
        if ( !pw ) return;
        auto & w = *pw;

        const auto unwrap = +[](VEntry & ve) -> MDP::Values & {return ve.values;};
        const auto wbegin = boost::make_transform_iterator(std::begin(w), unwrap);
        const auto wend   = boost::make_transform_iterator(std::end  (w), unwrap);

        // Remove easy ValueFunctions to avoid doing more work later.
        w.erase(extractDominated(S, wbegin, wend).base(), std::end(w));

        const size_t size = w.size();
        if ( size < 2 ) return;

        // Initialize the new best list with some easy finds, and remove them from
        // the old list.
        VList::iterator begin = std::begin(w), end = std::end(w), bound = begin;

        bound = extractBestAtSimplexCorners(S, begin, bound, end);

        // Here we could do some random belief lookups..

        // If we actually have still work to do..
        if ( bound < end ) {
            // We setup the lp preparing for a max of size rows.
            lp.reset();
            lp.allocate(size);

            // Setup initial LP rows. Note that best can't be empty, since we have
            // at least one best for the simplex corners.
            for ( auto it = begin; it != bound; ++it )
                lp.addOptimalRow(it->values);
        }

        // For each of the remaining points now we try to find a witness
        // point with respect to the best ones. If there is, there is
        // something we need to extract to best.
        //
        // What we are going to do is to push each 'best' constraint into
        // the lp, and then push/pop the 'v' constraint every time we need
        // to try out a new one.
        //
        // That we do in the findWitnessPoint function.
        while ( bound < end ) {
            const auto witness = lp.findWitness((end-1)->values);
            // If we get a belief point, we search for the actual vector that provides
            // the best value on the belief point, we move it into the best vector.
            if ( witness ) {
                bound = extractBestAtBelief(*witness, bound, bound, end);  // Advance bound with the next best
                lp.addOptimalRow((bound-1)->values);             // Add the newly found vector to our lp.
            }
            // We only advance if we did not find anything. Otherwise, we may have found a
            // witness point for the current value, but since we are not guaranteed to have
            // put into best that value, it may still keep witness to other belief points!
            else
                --end;
        }

        // Finally, we discard all bad vectors and we return just the best list.
        w.erase(bound, std::end(w));
    }
}

#endif
