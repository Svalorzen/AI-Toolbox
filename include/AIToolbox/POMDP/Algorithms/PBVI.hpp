#ifndef AI_TOOLBOX_POMDP_PBVI_HEADER_FILE
#define AI_TOOLBOX_POMDP_PBVI_HEADER_FILE

#include <AIToolbox/Utils/Prune.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/BeliefGenerator.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the Point Based Value Iteration algorithm.
     *
     * The idea behind this algorithm is to solve a POMDP Model
     * approximately. When computing a perfect solution, the main problem
     * is pruning the resulting ValueFunction in order to contain only a
     * parsimonious representation. What this means is that many vectors
     * inside can be dominated by others, and so they do not add any
     * additional information, while at the same time occupying memory and
     * computational time.
     *
     * The way this method tries to fix the problem is by solving the Model
     * in a set of specified Beliefs. Doing so results in no need for
     * pruning at all, since every belief uniquely identifies one of the
     * optimal solution vectors (only uniqueness in the final set is
     * required, but it is way cheaper than linear programming).
     *
     * The Beliefs can be given as input, or stochastically computed as to
     * cover as much as possible of the belief space, to ensure minimization of
     * the final error. The final solution will be correct 100% in the Beliefs
     * that have been selected, and will (possibly) undershoot in non-covered
     * Beliefs.
     *
     * In addition, the fact that we solve only for a fixed set of Beliefs
     * guarantees that our final solution is limited in size, which is
     * useful since even small POMDP true solutions can explode in size
     * with high horizons, for very little gain.
     *
     * There is no convergence guarantee of this method, but the error is
     * bounded.
     */
    class PBVI {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor sets the default horizon/tolerance used to
             * solve a POMDP::Model and the number of beliefs used to
             * approximate the ValueFunction.
             *
             * @param nBeliefs The number of support beliefs to use.
             * @param h The horizon chosen.
             * @param tolerance The tolerance factor to stop the PBVI loop.
             */
            PBVI(size_t nBeliefs, unsigned h, double tolerance);

            /**
             * @brief This function sets the tolerance parameter.
             *
             * The tolerance parameter must be >= 0.0, otherwise the
             * function will throw an std::runtime_error. The tolerance
             * parameter sets the convergence criterion. A tolerance of 0.0
             * forces PBVI to perform a number of iterations equal to
             * the horizon specified. Otherwise, PBVI will stop as soon
             * as the difference between two iterations is less than the
             * tolerance specified.
             *
             * @param tolerance The new tolerance parameter.
             */
            void setTolerance(double tolerance);

            /**
             * @brief This function sets a new horizon parameter.
             *
             * @param h The new horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function sets a new number of support beliefs.
             *
             * @param nBeliefs The new number of support beliefs.
             */
            void setBeliefSize(size_t nBeliefs);

            /**
             * @brief This function returns the currently set tolerance parameter.
             *
             * @return The current tolerance.
             */
            double getTolerance() const;

            /**
             * @brief This function returns the currently set horizon parameter.
             *
             * @return The current horizon.
             */
            unsigned getHorizon() const;

            /**
             * @brief This function returns the currently set number of support beliefs to use during a solve pass.
             *
             * @return The number of support beliefs.
             */
            size_t getBeliefSize() const;

            /**
             * @brief This function solves a POMDP::Model approximately.
             *
             * This function computes a set of beliefs for which to solve
             * the input model. The beliefs are chosen stochastically,
             * trying to cover as much as possible of the belief space in
             * order to offer as precise a solution as possible. The final
             * solution will only contain ValueFunctions for those Beliefs
             * and will interpolate them for points it did not solve for.
             * Even though the resulting solution is approximate very often
             * it is good enough, and this comes with an incredible
             * increase in speed.
             *
             * Note that even in the beliefs sampled the solution is not
             * guaranteed to be optimal. This is because a solution for
             * horizon h can only be computed with the true solution from
             * horizon h-1. If such a solution is approximate (and it is
             * here), then the solution for h will not be optimal by
             * definition.
             *
             * @tparam M The type of POMDP model that needs to be solved.
             *
             * @param model The POMDP model that needs to be solved.
             * @param v The ValueFunction to startup the process from, if needed.
             *
             * @return A tuple containing the maximum variation for the
             *         ValueFunction and the computed ValueFunction.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<double, ValueFunction> operator()(const M & model, ValueFunction v = {});

            /**
             * @brief This function solves a POMDP::Model approximately.
             *
             * This function uses and evaluates the input beliefs.
             *
             * The final solution will only contain ValueFunctions for those
             * Beliefs and will interpolate them for points it did not solve
             * for. Even though the resulting solution is approximate very
             * often it is good enough, and this comes with an incredible
             * increase in speed.
             *
             * @tparam M The type of POMDP model that needs to be solved.
             *
             * @param model The POMDP model that needs to be solved.
             * @param beliefs The list of beliefs to evaluate.
             * @param v The ValueFunction to startup the process from, if needed.
             *
             * @return A tuple containing the maximum variation for the
             *         ValueFunction and the computed ValueFunction.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<double, ValueFunction> operator()(const M & model, const std::vector<Belief> & bList, ValueFunction v = {});

        private:
            /**
             * @brief This function computes a VList composed the maximized cross-sums with respect to the provided beliefs.
             *
             * This function performs the job of accumulating the
             * information required to obtain the final policy. It
             * processes an action at a time.
             *
             * For each belief contained in the argument BeliefList, it
             * will create the optimal VEntry by cherry picking the best
             * projections for each observation. Finally it prunes the
             * resulting VList by removing duplicates.
             *
             * @param ProjectionsRow The type containing the projections to process.
             * @param projs A 1d container containing O elements: each a VList of projections for the respective observation.
             * @param a The action that this cross-sum is about.
             * @param bl The beliefs for which we are trying to find VEntries.
             *
             * @return The optimal cross-sum list for the given projections and BeliefList.
             */
            template <typename ProjectionsRow>
            VList crossSum(const ProjectionsRow & projs, size_t a, const std::vector<Belief> & bl);

            size_t S, A, O, beliefSize_;
            unsigned horizon_;
            double tolerance_;

            mutable RandomEngine rand_;
    };

    template <typename M, typename>
    std::tuple<double, ValueFunction> PBVI::operator()(const M & model, ValueFunction v) {
        // In this implementation we compute all beliefs in advance. This
        // is mostly due to the fact that I prefer counter parameters (how
        // many beliefs do you want?) versus timers (loop until time is
        // up). However, this is easily changeable, since the belief generator
        // can be called multiple times to increase the size of the belief
        // vector.
        BeliefGenerator bGen(model);
        return operator()(model, bGen(beliefSize_), v);
    }

    template <typename M, typename>
    std::tuple<double, ValueFunction> PBVI::operator()(const M & model, const std::vector<Belief> & beliefs, ValueFunction v) {
        // Initialize "global" variables
        S = model.getS();
        A = model.getA();
        O = model.getO();

        if (v.size() == 0)
            v = makeValueFunction(S);

        unsigned timestep = 0;

        Projecter projecter(model);

        // And off we go
        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);
        double variation = tolerance_ * 2; // Make it bigger
        while ( timestep < horizon_ && ( !useTolerance || variation > tolerance_ ) ) {
            ++timestep;

            // Compute all possible outcomes, from our previous results.
            // This means that for each action-observation pair, we are going
            // to obtain the same number of possible outcomes as the number
            // of entries in our initial vector w.
            auto projs = projecter(v.back());

            size_t finalWSize = 0;
            // In this method we split the work by action, which will then
            // be joined again at the end of the loop. This is not required,
            // but there does not seem to be a speed boost by not doing
            // so (not that I found one, if there is one I'd like to know!)
            for ( size_t a = 0; a < A; ++a ) {
                projs[a][0] = crossSum( projs[a], a, beliefs );
                finalWSize += projs[a][0].size();
            }
            VList w;
            w.reserve(finalWSize);

            for ( size_t a = 0; a < A; ++a )
                w.insert(std::end(w), std::make_move_iterator(std::begin(projs[a][0])), std::make_move_iterator(std::end(projs[a][0])));

            auto begin = std::begin(w);
            auto end   = std::end(w);
            auto bound = begin;
            for ( const auto & belief : beliefs )
                bound = extractBestAtPoint(belief, begin, bound, end, unwrap);

            w.erase(bound, std::end(w));

            // If you want to save as much memory as possible, do this.
            // It make take some time more though since it needs to reallocate
            // and copy stuff around.
            // w.shrink_to_fit();

            v.emplace_back(std::move(w));

            // Check convergence
            if ( useTolerance )
                variation = weakBoundDistance(v[v.size()-2], v.back());
        }

        return std::make_tuple(useTolerance ? variation : 0.0, v);
    }

    template <typename ProjectionsRow>
    VList PBVI::crossSum(const ProjectionsRow & projs, const size_t a, const std::vector<Belief> & bl) {
        VList result;
        result.reserve(bl.size());

        for ( const auto & b : bl )
            result.emplace_back(crossSumBestAtBelief(b, projs, a));

        const auto rbegin = std::begin(result);
        const auto rend   = std::end  (result);

        result.erase(extractDominated(rbegin, rend, unwrap), rend);

        return result;
    }
}

#endif
