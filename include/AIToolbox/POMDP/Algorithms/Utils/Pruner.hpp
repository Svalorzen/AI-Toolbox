#ifndef AI_TOOLBOX_POMDP_PRUNER_HEADER_FILE
#define AI_TOOLBOX_POMDP_PRUNER_HEADER_FILE

#include <utility>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>

namespace AIToolbox {
    namespace POMDP {

        /**
         * @brief This check the interface for a WitnessLP.
         *
         * @tparam LP The type of the LP to be checked.
         */
        template <typename LP>
        struct is_witness_lp {
            private:
                template <typename Z> static auto test(int) -> decltype(

                        Z(0), // Check we can build it from a size_t
                        static_cast<void (Z::*)(size_t size)>                                       (&Z::resetAndAllocate),
                        static_cast<void (Z::*)(const std::vector<double>&)>                        (&Z::addOptimalRow),
                        static_cast<std::tuple<bool, Belief> (Z::*)(const std::vector<double>&)>    (&Z::findWitness),

                        std::true_type()
                );

                template <typename Z> static auto test(...) -> std::false_type;

            public:
                enum { value = std::is_same<decltype(test<LP>(0)),std::true_type>::value };
        };

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename WitnessLP, typename = typename std::enable_if<is_witness_lp<WitnessLP>::value>::type>
        class Pruner;
#endif
        /**
         * @brief This class offers pruning facilities for non-parsimonious ValueFunction sets.
         */
        template <typename WitnessLP>
        class Pruner<WitnessLP> {
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
        Pruner<WitnessLP>::Pruner(size_t s) : S(s), lp(s) {}

        // The idea is that the input thing already has all the best vectors,
        // thus we only need to find them and discard the others.
        template <typename WitnessLP>
        void Pruner<WitnessLP>::operator()(VList * pw) {
            auto & w = *pw;

            // Remove easy ValueFunctions to avoid doing more work later.
            w.erase(extractDominated(S, std::begin(w), std::end(w)), std::end(w));

            size_t size = w.size();
            if ( size < 2 ) return;

            // We setup the lp preparing for a max of size rows.
            lp.resetAndAllocate(size);

            // Initialize the new best list with some easy finds, and remove them from
            // the old list.
            VList::iterator begin = std::begin(w), end = std::end(w), bound = end;

            bound = extractBestAtSimplexCorners(S, begin, bound, end);

            // Here we could do some random belief lookups..

            // Initialize best list with what we have found so far.
            VList best(std::make_move_iterator(bound), std::make_move_iterator(end));

            // Setup initial LP rows. Note that best can't be empty, since we have
            // at least one best for the simplex corners.
            for ( auto & bv : best )
                lp.addOptimalRow(std::get<VALUES>(bv));

            // For each of the remaining points now we try to find a witness
            // point with respect to the best ones. If there is, there is
            // something we need to extract to best.
            //
            // What we are going to do is to push each 'best' constraint into
            // the lp, and then push/pop the 'v' constraint every time we need
            // to try out a new one.
            //
            // That we do in the findWitnessPoint function.
            while ( begin < bound ) {
                auto result = lp.findWitness( std::get<VALUES>(*begin) );
                // If we get a belief point, we search for the actual vector that provides
                // the best value on the belief point, we move it into the best vector.
                if ( std::get<0>(result) ) {
                    auto & witness = std::get<1>(result);
                    bound = extractBestAtBelief(std::begin(witness),
                                                std::end(witness), begin, bound, bound);       // Moves the best at the "end"
                    best.emplace_back(std::move(*bound));                                      // We don't care about what we leave here..
                    lp.addOptimalRow(std::get<VALUES>(best.back()));                           // Add the newly found vector to our lp.
                }
                // We only advance if we did not find anything. Otherwise, we may have found a
                // witness point for the current value, but since we are not guaranteed to have
                // put into best that value, it may still keep witness to other belief points!
                else
                    ++begin;
            }

            // Finally, we discard all bad vectors (and remains of moved ones) and
            // we return just the best list.
            std::swap(w, best);
        }
    }
}

#endif
