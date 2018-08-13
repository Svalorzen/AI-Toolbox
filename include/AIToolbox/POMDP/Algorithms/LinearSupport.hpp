#ifndef AI_TOOLBOX_POMDP_LINEAR_SUPPORT_HEADER_FILE
#define AI_TOOLBOX_POMDP_LINEAR_SUPPORT_HEADER_FILE

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>

#include <set>
#include <boost/container/flat_set.hpp>
#include <boost/iterator/indirect_iterator.hpp>

#include <Eigen/Dense>
#include <AIToolbox/Utils/Combinatorics.hpp>

namespace AIToolbox::POMDP {
    class LinearSupport {
        public:
            LinearSupport(unsigned horizon, double epsilon);

            /**
             * @brief This function sets the epsilon parameter.
             *
             * The epsilon parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The epsilon
             * parameter sets the convergence criterion. An epsilon of 0.0
             * forces LinearSupport to perform a number of iterations equal to
             * the horizon specified. Otherwise, LinearSupport will stop as soon
             * as the difference between two iterations is less than the
             * epsilon specified.
             *
             * @param e The new epsilon parameter.
             */
            void setEpsilon(double e);

            /**
             * @brief This function allows setting the horizon parameter.
             *
             * @param h The new horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function will return the currently set epsilon parameter.
             *
             * @return The currently set epsilon parameter.
             */
            double getEpsilon() const;

            /**
             * @brief This function returns the currently set horizon parameter.
             *
             * @return The current horizon.
             */
            unsigned getHorizon() const;

            /**
             * @brief This function solves a POMDP::Model completely.
             *
             * This function is pretty expensive (as are possibly all POMDP
             * solvers). It generates for each new solved timestep the
             * whole set of possible ValueFunctions, and prunes it
             * incrementally, trying to reduce as much as possible the
             * linear programming solves required.
             *
             * This function returns a tuple to be consistent with MDP
             * solving methods, but it should always succeed.
             *
             * @tparam M The type of POMDP model that needs to be solved.
             *
             * @param model The POMDP model that needs to be solved.
             *
             * @return A tuple containing the maximum variation for the
             *         ValueFunction and the computed ValueFunction.
             */
            template <typename M, typename = std::enable_if_t<is_model<M>::value>>
            std::tuple<double, ValueFunction> operator()(const M & model);

        private:
            /**
             * @brief This function uses already computed projections to create the best possible cross-sum for a single belief.
             *
             * @param projs The projections to use.
             * @param a The action for the cross-sum.
             * @param b The belief to use.
             *
             * @return The best possible cross-sum for the provided belief.
             */
            template <typename ProjectionsRow>
            VEntry crossSumBestAtBelief(const ProjectionsRow & projs, size_t a, const Belief & b);

            /**
             * @brief This function adds a default cross-sum to the agenda, to start off the algorithm.
             *
             * @param projs The projections to use.
             * @param a The action for the cross-sum.
             */
            template <typename ProjectionsRow>
            void addDefaultEntry(const ProjectionsRow & projs, size_t a);

            /**
             * @brief This function adds all possible variations of a given VEntry to the agenda.
             *
             * @param projs The projections from which the VEntry was derived.
             * @param a The action for the cross-sums.
             * @param variated The VEntry to use as a base.
             */
            template <typename ProjectionsRow>
            void addVariations(const ProjectionsRow & projs, size_t a, const VEntry & variated);

            size_t S, A, O;
            unsigned horizon_;
            double epsilon_;

            struct Vertex {
                Belief belief;
                boost::container::flat_set<std::set<VEntry>::iterator> supports;
                double currentValue;
                double error;
            };

            using Agenda = boost::heap::fibonacci_heap<Vertex>;

            Agenda agenda_;
    };

    template <typename NewIt, typename OldIt>
    std::vector<std::pair<Belief, double>> findVertices(NewIt beginNew, NewIt endNew, OldIt alphasBegin, OldIt alphasEnd);

    template <typename M, typename>
    std::tuple<double, ValueFunction> LinearSupport::operator()(const M& model) {
        unsigned timestep = 0;
        ++timestep;

        Projecter project(model);
        auto v = makeValueFunction(S); // TODO: May take user input

        auto projections = project(v[timestep-1]);

        // These are the good vectors, the ones that we are going to return for
        // sure.
        VList goodSupports;

        // We use this as a sorted linked list to handle all vectors, so it's
        // easy to check whether we already have one, and also each vertex can
        // keep references to them and we don't duplicate vectors all over the
        // place.
        std::set<VEntry> allSupports;

        std::vector<std::pair<Belief, double>> vertices;
        std::vector<Belief> triedVertices;

        // For each corner belief, find its value and alphavector. Add the
        // alphavectors in a separate list, remove duplicates. Note: In theory
        // we must be able to find all alphas for each corner, not just a
        // single best but all best. Cassandra does not do that though.. maybe
        // we can avoid it? He uses the more powerful corner detection tho.
        Belief corner(S); corner.fill(0.0);
        for (size_t s = 0; s < S; ++s) {
            corner[s] = 1.0;

            const auto [it, inserted] = allSupports.emplace(crossSumBestAtBelief(corner, projections));
            if (inserted) goodSupports.push_back(*it);

            corner[s] = 0.0;
        }

        // Now we find for all the alphavectors we have found, the vertices of
        // the polytope that they created. These vertices will bootstrap the
        // algorithm.
        auto goodBegin = goodSupports.begin();
        for (size_t i = 0; i < goodSupports.size(); ++i, ++goodBegin) {
            // For each alpha, we find its vertices against the others.
            IndexSkipMap map({i}, goodSupports);

            // Note that the range we pass here is made by a single vector.
            auto newVertices = findVertices(goodBegin, goodBegin + 1, map.cbegin(), map.cend());

            for (auto && v : newVertices)
                vertices.emplace_back(std::move(v));
        }
        // Here we remove duplicates, although ideally when we improve the
        // algorithm we won't have to do this.
        std::sort(std::begin(vertices), std::end(vertices),
                [](const std::pair<Belief, double> & lhs, const std::pair<Belief, double> & rhs) {
                    return veccmp(lhs.first, rhs.first) < 0;
                }
        );
        vertices.erase(std::unique(std::begin(vertices), std::end(vertices)), std::end(vertices));

        // BEGIN(loop)

        // For each corner, we find its true alphas and its best possible value.
        // Then we compute the error between a corner's known true value and
        // what we can do with the optimal alphas we already have.
        // If the error is low enough, we don't need them. Otherwise we add
        // them to the priority queue.
        for (auto & vertex : vertices) {
            double trueValue;
            auto support = crossSumBestAtBelief(vertex.first, projections, &trueValue);

            double currentValue = vertex.second;

            auto diff = trueValue - currentValue;
            if (diff < epsilon_) {
                triedVertices.emplace_back(std::move(vertex));
            } else {
                // newVertices.add_if_not_duplicate(support);
                // agenda.emplace_back(diff | std::move(vertex), std::move(support), currentValue);
            }
        }

        if (agenda_.size() == 0) {} // break;

        Vertex best = agenda_.top();
        agenda_.pop();

        auto supportsToCheck = best.supports;
        std::vector<Agenda::handle_type> verticesToRemove;

        // For each element in the agenda, we need to check whether any would
        // be made useless by the new supports that best is bringing in. If so,
        // we can remove them from the queue.
        for (auto it = agenda_.begin(); it != agenda_.end(); ++it) {
            bool remove = false;
            for (const auto & sIt : best.supports) {
                // If so, *their* supports, plus the supports of best form the surface
                // in which we need to find vertices.
                if (it->belief.dot(sIt->values) > it->currentValue) {
                    remove = true;
                    break;
                }
            }
            if (remove) verticesToRemove.push_back(Agenda::s_handle_from_iterator(it));
        }
        for (const auto & h : verticesToRemove) {
            supportsToCheck.merge(std::move((*h).supports));
            agenda_.erase(h);
        }

        // Find vertices between the best support of this belief and the list
        // we already have.
        vertices = findVertices(
                        boost::make_indirect_iterator(std::begin(best.supports)),
                        boost::make_indirect_iterator(std::end(best.supports)),
                        boost::make_indirect_iterator(std::begin(supportsToCheck)),
                        boost::make_indirect_iterator(std::end(supportsToCheck))
                   );

        // We now can add the support for this vertex to the main list.  We
        // don't need checks here because we are guaranteed that we are
        // improving the VList.
        for (auto s : best.supports)
            goodSupports.push_back(*s);

        // END(loop)

        // return big vlist, next loop timestep.
    }

    template <typename NewIt, typename OldIt>
    std::vector<std::pair<Belief, double>> findVertices(NewIt beginNew, NewIt endNew, OldIt alphasBegin, OldIt alphasEnd) {
        std::vector<std::pair<Belief, double>> vertices;

        const size_t alphasSize = std::distance(alphasBegin, alphasEnd);
        if (alphasSize == 0) return vertices;
        const size_t S = alphasBegin->values.size();

        // This enumerator allows us to compute all possible subsets of S-1
        // elements. We use it on both the alphas, and the boundaries, thus the
        // number of elements we iterate over is alphasSize + S.
        SubsetEnumerator enumerator(S - 1, 0ul, alphasSize + S);

        Matrix2D m(S + 1, S + 1);
        m.row(0)[S] = -1; // First row is always a vector

        Vector boundary(S+1);
        boundary[S] = 0.0; // The boundary doesn't care about the value

        Vector b(S+1); b.fill(0.0);

        Vector result(S+1);

        // Common matrix/vector setups

        for (auto newVIt = beginNew; newVIt != endNew; ++newVIt) {
            const auto & newV = *newVIt;

            enumerator.reset();

            m.row(0).head(S) = newV.values;

            // Get subset of planes, find corner with LU
            size_t last = 0;
            while (enumerator.isValid()) {
                // Reset boundaries to care about all dimensions
                boundary.head(S).fill(1.0);
                size_t counter = 1;
                // Note that we start from last to avoid re-copying vectors
                // that are already in the matrix in their correct place.
                for (auto i = last; i < enumerator->size(); ++i) {
                    // For each value in the enumerator, if it is less than
                    // alphasSize it is referring to an alphaVector we need to
                    // take into account.
                    const auto index = (*enumerator)[i];
                    if (index < alphasSize) {
                        // Copy the right vector in the matrix.
                        m.row(counter).head(S) = std::next(alphasBegin, index)->values;
                        m.row(counter)[S] = -1;
                        ++counter;
                    } else {
                        // We limit the index-th dimension (minus alphasSize to scale in a 0-S range)
                        boundary[index - alphasSize] = 0.0;
                    }
                }
                m.row(counter) = boundary;
                b[counter] = 1.0;
                ++counter;

                // Note that we only need to consider the first "counter" rows,
                // as the boundaries get merged in a single one.
                result = m.topRows(counter).colPivHouseholderQr().solve(b.head(counter));

                b[counter-1] = 0.0;

                // Add to found only if valid, otherwise skip.
                if (((result.head(S).array() >= 0) && (result.head(S).array() <= 1.0)).all()) {
                    vertices.emplace_back(result.head(S), result[S]);
                }

                // Advance, and take the id of the first index changed in the
                // next combination.
                last = enumerator.advance();

                // If the index went over the alpha list, then we'd only have
                // boundaries, but we don't care about those cases (since we
                // assume we already have the corners of the simplex computed).
                // Thus, terminate.
                if ((*enumerator)[last] >= alphasSize) break;
            }
        }
        return vertices;
    }
}

#endif
