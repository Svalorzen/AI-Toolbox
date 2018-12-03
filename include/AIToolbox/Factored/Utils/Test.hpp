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

            // Basis function (vector of size |S|)
            struct BasisFunction {
                PartialKeys tag;
                Vector values;
            };

            // Qa, Ra, A: vector of size |S|, represented as linear sum of K basis functions.
            using FactoredVector = std::vector<BasisFunction>;
            // Q, R: Matrix of |S| x |A|. The vector is of size |A|
            // std::vector<FactoredVector>;

            struct FactoredMatrix {
                PartialKeys tag;
                Matrix2D matrix;
            };
            using Factored2DMatrix = std::vector<FactoredMatrix>;
            // T: Matrix of |S| x |A| x |S|
            using Factored3DMatrix = boost::multi_array<FactoredMatrix, 2>;

            inline double getValue(const Factors & space, const Factors & value, const FactoredVector & v) {
                double retval = 0.0;
                for (const auto & e : v) {
                   auto id = toIndexPartial(e.tag, space, value);
                   retval += e.values[id];
                }
                return retval;
            }

            inline BasisFunction dot(const Factors & space, const BasisFunction & lhs, const BasisFunction & rhs) {
                BasisFunction retval;

                // The output function will have the domain of both inputs.
                retval.tag = merge(lhs.tag, rhs.tag);

                retval.values.resize(toIndexPartial(retval.tag, space, space));
                // No need to zero fill

                size_t i = 0;
                PartialFactorsEnumerator e(space, retval.tag);
                while (e.isValid()) {
                    // We don't need to compute the index for retval since it
                    // increases sequentially anyway.
                    auto lhsId = toIndexPartial(lhs.tag, space, *e);
                    auto rhsId = toIndexPartial(rhs.tag, space, *e);

                    retval.values[i] = lhs.values[lhsId] * rhs.values[rhsId];

                    ++i;
                    e.advance();
                }

                return retval;
            }

            inline BasisFunction plus(const Factors & space, const BasisFunction & lhs, const BasisFunction & rhs) {
                BasisFunction retval;

                // The output function will have the domain of both inputs.
                retval.tag = merge(lhs.tag, rhs.tag);

                retval.values.resize(toIndexPartial(retval.tag, space, space));
                // No need to zero fill

                size_t i = 0;
                PartialFactorsEnumerator e(space, retval.tag);
                while (e.isValid()) {
                    // We don't need to compute the index for retval since it
                    // increases sequentially anyway.
                    auto lhsId = toIndexPartial(lhs.tag, space, *e);
                    auto rhsId = toIndexPartial(rhs.tag, space, *e);

                    retval.values[i] = lhs.values[lhsId] + rhs.values[rhsId];

                    ++i;
                    e.advance();
                }

                return retval;
            }

            inline BasisFunction backProject(const Factors & space, const Factored2DMatrix & lhs, const BasisFunction & rhs) {
                // Here we have the two function inputs, in this form:
                //
                //     lhs: [parents, child] -> value
                //     rhs: [children] -> value
                BasisFunction retval;

                // The domain here depends on the parents of all elements of
                // the domain of the input basis.
                for (auto d : rhs.tag)
                    merge(retval.tag, lhs[d].tag);

                retval.values.resize(toIndexPartial(retval.tag, space, space));
                // Don't need to zero fill

                // Iterate over the domain, since the output basis is going to
                // be dense pretty much.
                size_t id = 0;
                PartialFactorsEnumerator domain(space, retval.tag);
                PartialFactorsEnumerator rhsdomain(space, rhs.tag);
                while (domain.isValid()) {
                    // For each domain assignment, we need to go over every
                    // possible children assignment. As we are computing
                    // products, it is sufficient to go over the elements
                    // stored in the RHS (as all other children combinations
                    // are zero by definition).
                    //
                    // For each such assignment, we compute the product of the
                    // rhs there with the value of the lhs at the current
                    // domain & children.
                    double currentVal = 0.0;
                    size_t i = 0;
                    while (rhsdomain.isValid()) {
                        // The rhs has a single value for this children
                        // assignment, so we just pick that.
                        double x = rhs.values[i];

                        // The lhs however is made up of one component per
                        // child, and we need to multiply all of them together.
                        // At each iteration we look at a different "child".
                        for (size_t j = 0; j < rhs.tag.size(); ++j) {
                            // Find the matrix relative to this child
                            const auto & fun = lhs[rhs.tag[j]];
                            // Select the parents for this child
                            const auto & dom = fun.tag;
                            // Compute the "dense" id for the needed parents
                            // from the current domain.
                            auto id = toIndexPartial(dom, space, *domain);
                            // Multiply the current value by the lhs value.
                            x *= fun.matrix(id, (*rhsdomain).second[j]);
                        }
                        currentVal += x;

                        ++i;
                        rhsdomain.advance();
                    }
                    retval.values[id] = currentVal;

                    ++id;
                    domain.advance();
                    rhsdomain.reset();
                }
                return retval;
            }

            inline FactoredVector backProject(const Factors & space, const Factored2DMatrix & lhs, const FactoredVector & rhs) {
                FactoredVector retval;
                retval.reserve(rhs.size());

                for (const auto & basis : rhs)
                    retval.emplace_back(backProject(space, lhs, basis));

                return retval;
            }

            inline FactoredVector & operator*=(FactoredVector & lhs, const Vector & w) {
                for (size_t i = 0; i < lhs.size(); ++i)
                    lhs[i].values *= w[i];

                return lhs;
            }

            inline FactoredVector & operator*=(FactoredVector & lhs, const double v) {
                for (auto & l : lhs)
                    l.values *= v;

                return lhs;
            }

            // Performs the bellman equation on a single action
            FactoredVector bellmanEquation(const Factors & S, double gamma, const Factored2DMatrix & P, const FactoredVector & A, const Vector & w, const FactoredVector & R) {
                FactoredVector Q = backProject(S, P, A);
                Q *= w;
                Q *= gamma;

                return Q + R;
            }
}

#endif
