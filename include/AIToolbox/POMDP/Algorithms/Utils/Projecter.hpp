#ifndef AI_TOOLBOX_POMDP_PROJECTER_HEADER_FILE
#define AI_TOOLBOX_POMDP_PROJECTER_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class offers projecting facilities for Models.
     */
    template <typename M>
    class Projecter {
        static_assert(is_model_v<M>, "This class only works for POMDP models!");

        public:
            using ProjectionsTable          = boost::multi_array<VList, 2>;
            using ProjectionsRow            = boost::multi_array<VList, 1>;

            /**
             * @brief Basic constructor.
             *
             * This constructor initializes the internal immediate reward table and the
             * table containing what are the possible observations for the model (this
             * may speed up the computation of the projections).
             *
             * @param model The model that is used as a base for all projections.
             */
            Projecter(const M & model);

            /**
             * @brief This function returns all possible projections for the provided VList.
             *
             * @param w The list that needs to be projected.
             *
             * @return A 2d array of projection lists.
             */
            ProjectionsTable operator()(const VList & w);

            /**
             * @brief This function returns all possible projections for the provided VList and action.
             *
             * @param w The list that needs to be projected.
             * @param a The action used for projecting the list.
             *
             * @return A 1d array of projection lists.
             */
            ProjectionsRow operator()(const VList & w, size_t a);

        private:
            using PossibleObservationsTable = boost::multi_array<bool,  2>;

            /**
             * @brief This function precomputes which observations are possible from specific actions.
             */
            void computePossibleObservations();

            /**
             * @brief This function precomputes immediate rewards for the POMDP state-action pairs.
             */
            void computeImmediateRewards();

            const M & model_;
            size_t S, A, O;
            double discount_;

            Matrix2D immediateRewards_;
            PossibleObservationsTable possibleObservations_;
    };

    template <typename M>
    Projecter<M>::Projecter(const M& model) :
            model_(model), S(model_.getS()), A(model_.getA()), O(model_.getO()),
            discount_(model_.getDiscount()), possibleObservations_(boost::extents[A][O])
    {
        computePossibleObservations();
        computeImmediateRewards();
    }

    template <typename M>
    typename Projecter<M>::ProjectionsTable Projecter<M>::operator()(const VList & w) {
        ProjectionsTable projections( boost::extents[A][O] );

        for ( size_t a = 0; a < A; ++a )
            projections[a] = operator()(w, a);

        return projections;
    }

    template <typename M>
    typename Projecter<M>::ProjectionsRow Projecter<M>::operator()(const VList & w, const size_t a) {
        ProjectionsRow projections( boost::extents[O] );

        for ( size_t o = 0; o < O; ++o ) {
            // Here we put in just the immediate rewards so that the cross-summing step in the main
            // function works correctly. However we communicate via the boolean that pruning should
            // not be done at this step (since adding constants shouldn't do anything anyway).
            if ( !possibleObservations_[a][o] ) {
                // We add a parent id anyway in order to keep the code that cross-sums simple. However
                // note that this fake ID of 0 should never be used, so it should be safe to avoid
                // setting it to a special value like -1. If one really wants to check, he/she can
                // just look at the observation table and the belief and see if it makes sense.
                projections[o].emplace_back(immediateRewards_.row(a), a, VObs(1,0));
                continue;
            }

            // Otherwise we compute a projection for each ValueFunction supplied to us.
            MDP::Values vproj(S);
            for ( size_t i = 0; i < w.size(); ++i ) {
                const auto & v = w[i].values;
                // For each value function in the previous timestep, we compute the new value
                // if we performed action a and obtained observation o.
                // vproj_{a,o}[s] = R(s,a) / |O| + discount * sum_{s'} ( T(s,a,s') * O(s',a,o) * v_{t-1}(s') )
                if constexpr(is_model_eigen_v<M>) {
                    vproj = model_.getTransitionFunction(a) * (v.cwiseProduct(model_.getObservationFunction(a).col(o)));
                } else {
                    vproj.setZero();
                    for ( size_t s = 0; s < S; ++s )
                        for ( size_t s1 = 0; s1 < S; ++s1 )
                            vproj[s] += model_.getTransitionProbability(s,a,s1) * model_.getObservationProbability(s1,a,o) * v[s1];
                }
                // Set new projection with found value and previous V id.
                // projections[o].emplace_back(vproj, a, VObs(1,i));
                projections[o].emplace_back(vproj * discount_ + immediateRewards_.row(a).transpose(), a, VObs(1,i));
            }
        }
        return projections;
    }

    template <typename M>
    void Projecter<M>::computeImmediateRewards() {
        immediateRewards_ = [&]{
            if constexpr(MDP::is_model_eigen_v<M>)
                return model_.getRewardFunction().transpose();
            else
                return MDP::computeImmediateRewards(model_).transpose();
        }();
        // You can find out why this is divided in the incremental pruning paper =)
        // The idea is that at the end of all the cross sums it's going to add up to the correct value.
        immediateRewards_ /= static_cast<double>(O);
    }

    template <typename M>
    void Projecter<M>::computePossibleObservations() {
        for ( size_t a = 0; a < A; ++a )
            for ( size_t o = 0; o < O; ++o )
                for ( size_t s = 0; s < S; ++s ) // This NEEDS to be last!
                    if ( checkDifferentSmall(model_.getObservationProbability(s,a,o), 0.0) ) { possibleObservations_[a][o] = true; break; } // We only break the S loop!
    }
}

#endif
