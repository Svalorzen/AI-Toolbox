#ifndef AI_TOOLBOX_POMDP_LINEAR_SUPPORT_HEADER_FILE
#define AI_TOOLBOX_POMDP_LINEAR_SUPPORT_HEADER_FILE

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>

#include <set>
#include <boost/container/flat_set.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <AIToolbox/Utils/VertexEnumeration.hpp>
#include <AIToolbox/LP.hpp>

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

    template <typename M, typename>
    std::tuple<double, ValueFunction> LinearSupport::operator()(const M& model) {
        const auto S = model.getS();
        const auto A = model.getA();
        const auto O = model.getO();

        Projecter project(model);
        auto v = makeValueFunction(S); // TODO: May take user input

        unsigned timestep = 0;
        const bool useEpsilon = checkDifferentSmall(epsilon_, 0.0);
        double variation = epsilon_ * 2; // Make it bigger
        while ( timestep < horizon_ && ( !useEpsilon || variation > epsilon_ ) ) {
            ++timestep;

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

            const auto unwrap = +[](VEntry & ve) -> MDP::Values & {return ve.values;};
            const auto unwrapIt = +[](auto & ve) -> MDP::Values & {return ve->values;};

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
                const auto cbegin = boost::make_transform_iterator(map.cbegin(), unwrap);
                const auto cend   = boost::make_transform_iterator(map.cend(), unwrap);

                // Note that the range we pass here is made by a single vector.
                auto newVertices = findVerticesNaive(goodBegin, goodBegin + 1, cbegin, cend);

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
            const auto supBegin = boost::make_transform_iterator(std::begin(best.supports), unwrapIt);
            const auto supEnd   = boost::make_transform_iterator(std::end(best.supports), unwrapIt);
            const auto chkBegin = boost::make_transform_iterator(std::begin(supportsToCheck), unwrapIt);
            const auto chkEnd   = boost::make_transform_iterator(std::end(supportsToCheck), unwrapIt);
            vertices = findVerticesNaive(supBegin, supEnd, chkBegin, chkEnd);

            // We now can add the support for this vertex to the main list.  We
            // don't need checks here because we are guaranteed that we are
            // improving the VList.
            for (auto s : best.supports)
                goodSupports.push_back(*s);
        }
        // END(loop)

        return std::make_tuple(false, makeValueFunction(S));
    }

    /**
     * @brief This function computes the optimistic value of a point given known vertices and values.
     *
     * This function computes an LP to determine the best possible value of a
     * point given all known best vertices around it.
     *
     * This function is needed in multi-objective settings (rather than
     * POMDPs), since the step where we compute the optimal value for a given
     * point is extremely expensive (it requires solving a full MDP). Thus
     * linear programming is used in order to determine an optimistic bound
     * when deciding the next point to extract from the queue during the linear
     * support process.
     *
     * @param b The point where we want to compute the best possible value.
     * @param bvBegin The start of the range of point-value pairs representing all surrounding vertices.
     * @param bvEnd The end of that same range.
     *
     * @return The best possible value that the input point can have given the known vertices.
     */
    template <typename It>
    double computeOptimisticValue(const Vector & p, It pvBegin, It pvEnd) {
        const size_t vertexNumber = std::distance(pvBegin, pvEnd);
        if (vertexNumber == 0) return 0.0;
        const size_t S = p.size();

        LP lp(S);

        /*
         * With this LP we are looking for an optimistic hyperplane that can
         * tightly fit all corners that we already have, and maximize the value
         * at the input point.
         *
         * Our constraints are of the form
         *
         * vertex[0][0]) * h0 + vertex[0][1]) * h1 + ... <= vertex[0].currentValue
         * vertex[1][0]) * h0 + vertex[1][1]) * h1 + ... <= vertex[1].currentValue
         * ...
         *
         * Since we are looking for an optimistic hyperplane, all variables are
         * unbounded since the hyperplane may need to go negative at some
         * states.
         *
         * Finally, our constraint is a row to maximize:
         *
         * b * v0 + b * v1 + ...
         *
         * Which means we try to maximize the value of the input point with the
         * newly found hyperplane.
         */

        // Set objective to maximize
        lp.row = p;
        lp.setObjective(true);

        // Set unconstrained to all variables
        for (size_t s = 0; s < S; ++s)
            lp.setUnbounded(s);

        // Set constraints for all input belief points and current values.
        for (auto it = pvBegin; it != pvEnd; ++it) {
            lp.row = it->first;
            lp.pushRow(LP::Constraint::LessEqual, it->second);
        }

        double retval;
        // Note that we don't care about the optimistic alphavector, so we
        // discard it. We check that everything went fine though, in theory
        // there shouldn't be any problems here.
        auto solution = lp.solve(0, &retval);
        assert(solution);

        return retval;
    }
}

#endif
