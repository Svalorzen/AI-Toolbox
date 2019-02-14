#ifndef AI_TOOLBOX_POMDP_FAST_INFORMED_BOUND_HEADER_FILE
#define AI_TOOLBOX_POMDP_FAST_INFORMED_BOUND_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>

#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Utils.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the Fast Informed Bound algorithm.
     *
     * This class is useful in order to obtain a very simple upper bound for a
     * POMDP.
     *
     * This upper bound is computed as a simplification over the true
     * ValueFunction POMDP update (via Bellman Equation).
     *
     * I'm going to go through the whole derivation from scratch here since to
     * somebody reading it might turn out useful (and I have spent some time
     * trying to understand it so might as well write it up).
     *
     * We first start the derivation through the basic Bellman Equation for
     * POMDPs:
     *
     *     Q(b,a) = Sum_s R(s,a) * b(s) + gamma * Sum_o P(b'|b,a) * V(b')
     *     Q(b,a) = Sum_s R(s,a) * b(s) + gamma * Sum_o P(o|b,a) * V(b')
     *
     * This should be pretty self-explanatory: the value of a belief and an
     * action is equal to the reward you get by acting, plus the future
     * discounted rewards of whatever belief you end up in, times the
     * probability of ending up there.
     *
     * From here I just consider the second part (after gamma), since that's
     * the interesting stuff anyway. The rest remains the same.
     *
     *     Sum_o P(o|b,a) * V(b')
     *     Sum_o Sum_s P(o|s,a) * b(s) * V(b')
     *
     * Here b' is implied since it can be computed from b,a,o.
     *
     * So at this point we'd normally be done, but the thing is that this to
     * work this formula requires V(b) to be defined in every belief, and we'd
     * have to update it for every belief. This is kind of hard since there's
     * infinite beliefs to go over.
     *
     * So the trick is that we know that the ValueFunction is piecewise-linear
     * and convex, so we change the formula a bit in order to be able to use
     * the previous-step alphavectors to do the update.
     *
     *     Sum_o max_prev_alpha Sum_s' [ Sum_s P(s',o|s,a) * b(s) ] * prev_alpha(s')
     *
     * Here P(o|s,a) became P(s',o|s,a) simply because it has ended up inside
     * the Sum_s', to keep its value the same (this is probability math).
     *
     * So now instead of V, we look, for each observation, inside the best
     * previous alpha we can find for it, and sum over all its values (since
     * it's a vector).
     *
     * Now to the Fast Informed Bound. What we do is a simple shuffling of
     * terms, which increases the value:
     *
     *     Sum_o Sum_s max_prev_alpha Sum_s' [ P(s',o|s,a) * b(s) ] * prev_alpha(s')
     *
     * By moving the max inside the Sum_s, we increase the value of the
     * formula. From here on it's just algebra:
     *
     *     Sum_s b(s) Sum_o max_prev_alpha Sum_s' P(s',o|s,a) * prev_alpha(s')
     *     Q(b,a) = Sum_s b(s) * [ R(s,a) + gamma * Sum_o max_prev_alpha Sum_s' P(s',o|s,a) * prev_alpha(s') ]
     *
     * Finally, since with this method you produce Q(b,a), it means you'll
     * always produce A alphavectors. So you can write that as:
     *
     *     Q(b,a) = Sum_s b(s) * [ R(s,a) + gamma * Sum_o max_a' Sum_s' P(s',o|s,a) * Q(s',a') ]
     *     Q(s,a) = R(s,a) + gamma * Sum_o max_a' Sum_s' P(s',o|s,a) * Q(s',a')
     *
     * Which is the update we're doing in the code.
     */
    class FastInformedBound {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param horizon The maximum number of iterations to perform.
             * @param tolerance The tolerance factor to stop the value iteration loop.
             */
            FastInformedBound(unsigned horizon, double tolerance = 0.001);

            /**
             * @brief This function computes the Fast Informed Bound for the input POMDP.
             *
             * This function returns a QFunction since it's easier to work
             * with. If you want to use it to act within a POMDP, check out
             * QMDP which can transform it into a VList, and from there into a
             * ValueFunction.
             *
             * This method creates a SOSA matrix for the input model, and uses
             * it to create the bound.
             *
             * @param m The POMDP to be solved.
             * @param oldQ The QFunction to start iterating from.
             *
             * @return A tuple containing the maximum variation for the
             *         QFunction and the computed QFunction.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<double, MDP::QFunction> operator()(const M & m, const MDP::QFunction & oldQ = {});

            /**
             * @brief This function computes the Fast Informed Bound for the input POMDP.
             *
             * Internally, this method uses a SOSA matrix to improve its speed,
             * since otherwise it'd need to multiply the transition and
             * observation matrices over and over.
             *
             * Since we don't usually store SOSA matrices, the other operator()
             * computes it on the fly.
             *
             * In case you already have a POMDP with a pre-computed SOSA matrix
             * and don't need to recompute it, you can call this method
             * directly.
             *
             * You can use both sparse and dense Matrix4D for this method.
             *
             * @param m The POMDP to be solved.
             * @param sosa The SOSA matrix of the input POMDP.
             * @param oldQ The QFunction to start iterating from.
             *
             * @return A tuple containing the maximum variation for the
             *         QFunction and the computed QFunction.
             */
            template <typename M, typename SOSA, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<double, MDP::QFunction> operator()(const M & m, const SOSA & sosa, MDP::QFunction oldQ = {});

            /**
             * @brief This function sets the tolerance parameter.
             *
             * The tolerance parameter must be >= 0.0, otherwise the function
             * will throw an std::invalid_argument. The tolerance parameter sets
             * the convergence criterion. A tolerance of 0.0 forces the internal
             * loop to perform a number of iterations equal to the horizon
             * specified. Otherwise, FastInformedBound will stop as soon as the
             * difference between two iterations is less than the tolerance
             * specified.
             *
             * @param tolerance The new tolerance parameter.
             */
            void setTolerance(double tolerance);

            /**
             * @brief This function sets the horizon parameter.
             *
             * @param h The new horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function returns the currently set tolerance parameter.
             *
             * @return The currently set tolerance parameter.
             */
            double getTolerance() const;

            /**
             * @brief This function returns the current horizon parameter.
             *
             * @return The currently set horizon parameter.
             */
            unsigned getHorizon() const;

        private:
            size_t horizon_;
            double tolerance_;
    };

    template <typename M, typename>
    std::tuple<double, MDP::QFunction> FastInformedBound::operator()(const M & m, const MDP::QFunction & oldQ) {
        return operator()(m, makeSOSA(m), oldQ);
    }

    template <typename M, typename SOSA, typename>
    std::tuple<double, MDP::QFunction> FastInformedBound::operator()(const M & m, const SOSA & sosa, MDP::QFunction oldQ) {
        const auto & ir = [&]{
            if constexpr (is_model_eigen_v<M>) return m.getRewardFunction();
            else return computeImmediateRewards(m);
        }();
        auto newQ = MDP::QFunction(m.getS(), m.getA());

        if (oldQ.size() == 0) {
            oldQ.resize(m.getS(), m.getA());

            double max;
            using Tmp = remove_cv_ref_t<decltype(ir)>;
            if constexpr(std::is_base_of_v<Eigen::SparseMatrixBase<Tmp>, Tmp>)
                max = Eigen::Map<const Vector>(ir.valuePtr(), ir.size()).maxCoeff();
            else
                max = ir.maxCoeff();

            // Note that here we take the max over all IR: since we're
            // computing an upper bound, we want to assume that we're going to
            // do the best possible thing after each action forever.
            oldQ.fill(max / std::max(0.0001, 1.0 - m.getDiscount()));
        }

        unsigned timestep = 0;
        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);
        double variation = tolerance_ * 2; // Make it bigger
        while ( timestep < horizon_ && ( !useTolerance || variation > tolerance_ ) ) {
            ++timestep;
            newQ.setZero();
            // Q(s,a) = R(s,a) + gamma * Sum_o max_a' Sum_s' P(s',o|s,a) * Q(s',a')
            for (size_t a = 0; a < m.getA(); ++a)
                for (size_t o = 0; o < m.getO(); ++o)
                    newQ.col(a) += (sosa[a][o] * oldQ).rowwise().maxCoeff();
            newQ *= m.getDiscount();
            newQ += ir;

            if (useTolerance)
                variation = (oldQ - newQ).cwiseAbs().maxCoeff();

            std::swap(oldQ, newQ);
        }
        return std::make_tuple(useTolerance ? variation : 0.0, std::move(oldQ));
    }
}

#endif

