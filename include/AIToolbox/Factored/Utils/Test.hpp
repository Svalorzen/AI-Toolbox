#ifndef AI_TOOLBOX_FACTORED_TEST_HEADER_FILE
#define AI_TOOLBOX_FACTORED_TEST_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
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
            struct BasisFunction {
                using TagType = std::conditional_t<N == 1, PartialKeys, std::array<PartialKeys, N>>;

                struct FPartial {
                    TagType tagValue;
                    double value;
                };

                TagType tag;
                std::vector<FPartial> elements;
            };

            template <size_t N>
            using FactoredFunction = std::vector<BasisFunction<N>>;
            // TODO
            // TODO: Do we ever need N != 1?
            // TODO


            struct FactoredMatrix {
                Factors tag;
                Matrix2D matrix;
            };
            using Factored2DMatrix = std::vector<FactoredMatrix>;
            using Factored3DMatrix = boost::multi_array<FactoredMatrix, 2>;


            // TODO:
            //
            // - assert: inputs must have the same dimensions. This is because
            //           they have to reference to the same Factors.
            // -
            inline BasisFunction<1> operator*(const BasisFunction<1> & lhs, const BasisFunction<1> & rhs) {
                BasisFunction<1> retval;

                // The output function will have the domain of both inputs.
                std::vector<std::pair<size_t, size_t>> overlap;
                retval.tag = merge(lhs.tag, rhs.tag, &overlap);

                // Note that we don't need to iterate on the domain here, as
                // the input functions are zero where they are not defined.
                // Since we are implementing the product, all resulting
                // combinations that do not originate from both inputs are
                // going to be zero, and, thus, we skip them.
                retval.elements.reserve(lhs.elements.size() * (rhs.elements.size() - overlap.size()));
                for (const auto & el : lhs.elements) {
                    for (const auto & er : rhs.elements) {
                        if (!overlap.size()) {
                            // If there is no overlap, the new element is simply the product of each two elements.
                            // FIXME: Replace with emplace with C++20 (p0960)
                            retval.elements.push_back({merge(lhs.tag, el.tagValue, rhs.tag, er.tagValue), el.value * er.value});
                        } else {
                            // Otherwise, we need to check that the two
                            // overlapping parts of the domain actually match,
                            // or we would get an impossible combination.
                            bool match = true;
                            for (const auto & op : overlap) {
                                if (el.tagValue[op.first] != er.tagValue[op.second]) {
                                    match = false;
                                    break;
                                }
                            }
                            if (!match) continue;
                            // And if they do, again the result is a simple product.
                            retval.elements.push_back({merge(lhs.tag, el.tagValue, rhs.tag, er.tagValue), el.value * er.value});
                        }
                    }
                }

                return retval;
            }

            inline BasisFunction<1> backProject(const Factors & space, const Factored2DMatrix & lhs, const BasisFunction<1> & rhs) {
                // Here we have the two function inputs, in this form:
                //
                //     lhs: [parents, child] -> value
                //     rhs: [children] -> value
                BasisFunction<1> retval;

                // The domain here depends on the parents of all elements of
                // the domain of the input basis.
                for (auto d : rhs.tag)
                    merge(retval.tag, lhs[d].tag);

                // Iterate over the domain, since the output basis is going to
                // be dense pretty much.
                PartialFactorsEnumerator domain(space, retval.tag);
                while (domain.isValid()) {
                    // The new basis will have a value for every possible
                    // assignment of the domain.
                    BasisFunction<1>::FPartial partial{(*domain).second, 0.0};
                    // For each domain assignment, we need to go over every
                    // possible children assignment. As we are computing
                    // products, it is sufficient to go over the elements
                    // stored in the RHS (as all other children combinations
                    // are zero by definition).
                    //
                    // For each such assignment, we compute the product of the
                    // rhs there with the value of the lhs at the current
                    // domain & children.
                    for (const auto & r : rhs.elements) {
                        // The rhs has a single value for this children
                        // assignment, so we just pick that.
                        double x = r.value;
                        // The lhs however is made up of one component per
                        // child, and we need to multiply all of them together.
                        // At each iteration we look at a different "child".
                        for (size_t i = 0; i < rhs.tag.size(); ++i) {
                            // Find the matrix relative to this child
                            const auto & fun = lhs[rhs.tag[i]];
                            // Select the parents for this child
                            const auto & dom = fun.tag;
                            // Compute the "dense" id for the needed parents
                            // from the current domain.
                            auto id = toIndexPartial(dom, space, *domain);
                            // Multiply the current value by the lhs value.
                            x *= fun.matrix(id, r.tagValue[i]);
                        }

                        partial.value += x;
                    }
                    // If it's not zero, we can add it to the result basis.
                    if (checkDifferentGeneral(partial.value, 0.0))
                        retval.elements.emplace_back(std::move(partial));

                    domain.advance();
                }
                return retval;
            }
}

#endif
