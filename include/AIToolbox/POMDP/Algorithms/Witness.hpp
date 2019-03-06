#ifndef AI_TOOLBOX_POMDP_WITNESS_HEADER_FILE
#define AI_TOOLBOX_POMDP_WITNESS_HEADER_FILE

#include <unordered_set>

#include <boost/functional/hash.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>
#include <AIToolbox/Utils/Prune.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the Witness algorithm.
     *
     * This algorithm solves a POMDP Model perfectly. It computes solutions
     * for each horizon incrementally, every new solution building upon the
     * previous one.
     *
     * The Witness algorithm tries to avoid creating all possible cross-sums
     * of the projected vectors. Instead, it relies on a proof that states
     * that if a VEntry is suboptimal, then we can at least find a better one
     * by modifying a single subtree.
     *
     * Given this, the Witness algorithm starts off by finding a single optimal
     * VEntry for a random belief. Then, using the theorem, it knows that if a
     * better VEntry exists, then there must be at least one VEntry completely
     * equal to the one we just found but for a subtree, and that one will
     * be better. Thus, it adds to an agenda all possible variations of the
     * found optimal VEntry.
     *
     * From there, it examines each one of them, trying to look for a witness
     * point. Once found, again it produces an optimal VEntry for that point
     * and adds to the agenda all of its possible variations. VEntry which do
     * not have any witness points are removed from the agenda.
     *
     * In addition, Witness will not add to the agenda any VEntry which it has
     * already added; it uses a set to keep track of which combinations of
     * subtrees it has already tried.
     */
    class Witness {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor sets the default horizon used to solve a POMDP::Model.
             *
             * The tolerance parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The tolerance
             * parameter sets the convergence criterion. A tolerance of 0.0
             * forces Witness to perform a number of iterations equal to
             * the horizon specified. Otherwise, Witness will stop as soon
             * as the difference between two iterations is less than the
             * tolerance specified.
             *
             * @param h The horizon chosen.
             * @param tolerance The tolerance factor to stop the value iteration loop.
             */
            Witness(unsigned horizon, double tolerance);

            /**
             * @brief This function sets the tolerance parameter.
             *
             * The tolerance parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The tolerance
             * parameter sets the convergence criterion. A tolerance of 0.0
             * forces Witness to perform a number of iterations equal to
             * the horizon specified. Otherwise, Witness will stop as soon
             * as the difference between two iterations is less than the
             * tolerance specified.
             *
             * @param t The new tolerance parameter.
             */
            void setTolerance(double t);

            /**
             * @brief This function allows setting the horizon parameter.
             *
             * @param h The new horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function will return the currently set tolerance parameter.
             *
             * @return The currently set tolerance parameter.
             */
            double getTolerance() const;

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
             * solvers). It solves a series of LPs trying to find all possible
             * beliefs where an alphavector has not yet been found.
             *
             * @tparam M The type of POMDP model that needs to be solved.
             *
             * @param model The POMDP model that needs to be solved.
             *
             * @return A tuple containing the maximum variation for the
             *         ValueFunction and the computed ValueFunction.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<double, ValueFunction> operator()(const M & model);

        private:
            /**
             * @brief This function adds a default cross-sum to the agenda, to start off the algorithm.
             *
             * @param projs The projections to use.
             */
            template <typename ProjectionsRow>
            void addDefaultEntry(const ProjectionsRow & projs);

            /**
             * @brief This function adds all possible variations of a given VEntry to the agenda.
             *
             * @param projs The projections from which the VEntry was derived.
             * @param variated The VEntry to use as a base.
             */
            template <typename ProjectionsRow>
            void addVariations(const ProjectionsRow & projs, const VEntry & variated);

            size_t S, A, O;
            unsigned horizon_;
            double tolerance_;

            std::vector<MDP::Values> agenda_;
            std::unordered_set<VObs, boost::hash<VObs>> triedVectors_;
    };

    template <typename M, typename>
    std::tuple<double, ValueFunction> Witness::operator()(const M& model) {
        S = model.getS();
        A = model.getA();
        O = model.getO();

        std::vector<VList> U(A);

        auto v = makeValueFunction(S); // TODO: May take user input

        unsigned timestep = 0;

        // This variable we use to manually control the allocations
        // for the LP solver. This is because this algorithm cannot
        // know in advance just how many constraints the LP is going
        // to get. Thus we implement a x2 doubling allocation scheme
        // to avoid too many reallocations.
        size_t reserveSize = 1;

        Projecter project(model);
        Pruner prune(S);
        WitnessLP lp(S);

        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);
        double variation = tolerance_ * 2; // Make it bigger
        while ( timestep < horizon_ && ( !useTolerance || variation > tolerance_ ) ) {
            ++timestep;

            // As default, we allocate double the numbers of VEntries for last step.
            reserveSize = std::max(reserveSize, 2 * v[timestep-1].size());
            // Compute all possible outcomes, from our previous results.
            // This means that for each action-observation pair, we are going
            // to obtain the same number of possible outcomes as the number
            // of entries in our initial vector w.
            auto projections = project(v[timestep-1]);

            size_t finalWSize = 0;
            for ( size_t a = 0; a < A; ++a ) {
                U[a].clear();
                lp.reset();
                agenda_.clear();
                triedVectors_.clear();
                size_t counter = 0;

                lp.allocate(reserveSize);

                // We add the VEntry to startoff the whole process. This
                // VEntry does not even need to be optimal, as we are going
                // to compute the optimal one for the witness point anyway.
                addDefaultEntry(projections[a]);

                // We check whether any element in the agenda improves what we have
                while ( !agenda_.empty() ) {
                    const auto witness = lp.findWitness(agenda_.back());
                    if ( witness ) {
                        // If so, we generate the best vector for that particular belief point.
                        U[a].push_back(crossSumBestAtBelief(*witness, projections[a], a));
                        lp.addOptimalRow(U[a].back().values);
                        // We add to the agenda all possible "variations" of the VEntry found.
                        addVariations(projections[a], U[a].back());
                        // We manually check memory for the lp, since this method
                        // cannot know in advance how many rows it'll need to do.
                        if ( ++counter == reserveSize ) {
                            reserveSize *= 2;
                            lp.allocate(reserveSize);
                        }
                    }
                    else
                        agenda_.pop_back();
                }
                finalWSize += U[a].size();
            }
            VList w;
            w.reserve(finalWSize);

            // We put together all VEntries we found.
            for ( size_t a = 0; a < A; ++a )
                w.insert(std::end(w), std::make_move_iterator(std::begin(U[a])), std::make_move_iterator(std::end(U[a])));

            // We have them all, and we prune one final time to be sure we have
            // computed the parsimonious set of value functions.
            const auto begin = std::begin(w);
            const auto end   = std::end  (w);
            w.erase(prune(begin, end, unwrap), end);

            v.emplace_back(std::move(w));

            // Check convergence
            if ( useTolerance ) {
                variation = weakBoundDistance(v[timestep-1], v[timestep]);
            }
        }

        return std::make_tuple(useTolerance ? variation : 0.0, v);
    }

    template <typename ProjectionsRow>
    void Witness::addDefaultEntry(const ProjectionsRow & projs) {
        MDP::Values v(S); v.setZero();

        // We compute the crossSum between each best vector for the belief.
        for ( size_t o = 0; o < O; ++o )
            v.noalias() += projs[o][0].values;

        triedVectors_.emplace(O, 0);
        agenda_.emplace_back(std::move(v));
    }

    template <typename ProjectionsRow>
    void Witness::addVariations(const ProjectionsRow & projs, const VEntry & variated) {
        // We need to copy this one unfortunately
        auto vObs = variated.observations;
        const auto & vValues = variated.values;

        for ( size_t o = 0; o < O; ++o ) {
            const size_t skip = vObs[o];

            for ( size_t i = 0; i < projs[o].size(); ++i ) {
                if ( i == skip ) continue;

                vObs[o] = i;
                if ( triedVectors_.find(vObs) != std::end(triedVectors_) ) continue;

                triedVectors_.insert(vObs);

                auto v = vValues - projs[o][skip].values + projs[o][i].values;
                agenda_.emplace_back(std::move(v));
            }
            vObs[o] = skip;
        }
    }
}

#endif
