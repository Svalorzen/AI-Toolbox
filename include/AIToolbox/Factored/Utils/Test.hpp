#ifndef AI_TOOLBOX_FACTORED_TEST_HEADER_FILE
#define AI_TOOLBOX_FACTORED_TEST_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <array>
#include <vector>

// This file is temporary to test the operations.
namespace AIToolbox::Factored {
            // General TODO:
            //
            // 0) The matrices below, what has already gone in the main model
            // and the general QFunctionRules etc should be reworked into a
            // single entity.
            //
            // The idea is to have something as flexible as the rules, but a
            // bit more efficient, which stores same indexes in the same
            // substructure, but without the overhead of the FactorGraph.
            // Look at SparseCooperativeQLearning, maybe a FactoredContainer
            // could be used?
            //
            // Note: Hopefully then factoredLP can be rewritten to take those,
            // as using FactorGraph to store matrices does not seem to be a
            // good idea (and also does not help with VE, since VE has to
            // rebuild the graph anyway since it stores tags inside).
            // --> FactoredLP does not really seem to need graphs as inputs, so
            // that's even better.
            //
            // QFunctionRules must contain the target state/action because
            // FactoredContainer is not able to reconstruct it after filtering.
            // This could probably be worked around, but since it's used by
            // many entities and the workaround would be expensive, we probably
            // want to leave them be.
            //
            // However, we can devise another, alternative format that is done
            // as below, but instead of Matrices it enumerates the non-zero
            // cells explicitly, in a sparse representation.
            //
            // Need to think well how to use it tho. It's good to have since we
            // do need to separate rules by target variables, and a simple
            // vector of rules is by definition unsorted.
            //
            // Find something that would work well in the FactoredLPTests.
            //
            // 1) Implement Policy using PolicyIteration.pdf. Note that
            // implementing the policy requires the operations down below. At
            // the same time, it turns out that when you do the multiplications
            // a lot of things are the same (that's the result they use to
            // argue that policy lists are compact and efficient), so we may
            // want to implement the down below operations in a sort of batch
            // way. It depends on how it actually works tho.
            //
            //

            template <size_t N>
            struct FactoredFunctionRule {
                std::array<PartialFactors, N> tags;
                double value;
            };

            template <size_t N>
            struct FComponent {
                using TagType = std::conditional_t<N == 1, Factors, std::array<Factors, N>>;

                struct FPartial {
                    TagType tagValue;
                    double value;
                };

                TagType tag;
                std::vector<FPartial> elements;
            };

            template <size_t N>
            using FactoredFunction = std::vector<FComponent<N>>;

            // Note that we don't really store the vector orientation.
            using FactoredVector   = FactoredFunction<1>;
            using Factored2DMatrix = FactoredFunction<2>;
            using Factored3DMatrix = FactoredFunction<3>;

            // TODO:
            //
            // - assert: inputs must have the same dimensions. This is because
            //           they have to reference to the same Factors.
            // -
            FactoredVector operator+(const FactoredVector & lhs, const FactoredVector & rhs);
            double operator*(const FactoredVector & lhs, const FactoredVector & rhs);

            Factored2DMatrix operator+(const Factored2DMatrix & lhs, const Factored2DMatrix & rhs);

            FactoredVector operator*(const Factored2DMatrix & lhs, const FactoredVector & rhs);
            FactoredVector operator*(const FactoredVector & lhs, const Factored2DMatrix & rhs);
}

#endif
