#ifndef AI_TOOLBOX_MDP_DYNA2_HEADER_FILE
#define AI_TOOLBOX_MDP_DYNA2_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Algorithms/SARSAL.hpp>
#include <AIToolbox/Bandit/Policies/RandomPolicy.hpp>
#include <AIToolbox/MDP/Policies/BanditPolicyAdaptor.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the Dyna2 algorithm.
     *
     * This algorithm leverages the SARSAL algorithm in order to keep two
     * separate QFunctions: one permanent, and one transient.
     *
     * The permanent one contains the QFunction learned when actually
     * interacting with the real environment. The transient one is instead used
     * to learn against a generative model, so that it can explore.
     *
     * The transient one is overall always a sum of the permanent one and
     * whatever it learns during batch exploration. After each episode, the
     * transient memory should be cleared in order to avoid storing information
     * about states that it may never again encounter.
     *
     * Another advantage of clearing the memory is that, if the exploration
     * model is not perfect, imperfect information learned is also discarded.
     */
    template <typename M>
    class Dyna2 {
        static_assert(is_generative_model_v<M>, "This class only works for generative MDP models!");

        public:
            /**
             * @brief Basic constructor.
             *
             * @param m The model to be used to update the QFunction.
             * @param alpha The learning rate of the internal SARSAL methods.
             * @param lambda The lambda parameter for the eligibility traces.
             * @param tolerance The cutoff point for eligibility traces.
             * @param n The number of sampling passes to do on the model upon batchUpdateQ().
             */
            explicit Dyna2(const M & m, double alpha = 0.1, double lambda = 0.9, double tolerance = 0.001, unsigned n = 50);

            /**
             * @brief This function updates the internal QFunction.
             *
             * This function takes a single experience point and uses it to update
             * a QFunction. This is a very efficient method to keep the QFunction
             * up to date with the latest experience.
             *
             * In addition, the sampling list is updated so that batch
             * updating becomes possible as a second phase.
             *
             * The sampling list in Dyna2 is a simple list of all visited
             * state action pairs. This function is responsible for inserting
             * them in a set, keeping them unique.
             *
             * @param s The previous state.
             * @param a The action performed.
             * @param s1 The new state.
             * @param a1 The action performed in the new state.
             * @param rew The reward obtained.
             */
            void stepUpdateQ(size_t s, size_t a, size_t s1, size_t a1, double rew);

            /**
             * @brief This function updates a QFunction based on simulated experience.
             *
             * In Dyna2 we sample N times from already experienced
             * state-action pairs, and we update the resulting QFunction as
             * if this experience was actually real.
             *
             * The idea is that since we know which state action pairs we already
             * explored, we know that whose pairs are actually possible. Thus we
             * use the generative model to sample them again, and obtain a better
             * estimate of the QFunction.
             */
            void batchUpdateQ(size_t s);

            /**
             * @brief This function resets the transient QFunction to the permanent one.
             */
            void resetTransientLearning();

            /**
             * @brief This function sets the policy used to sample during batch updates.
             *
             * This function is provided separately in case you want to base
             * the policy on either the permanent or transient QFunctions,
             * which are internally owned and thus do not exist before this
             * class is actually created.
             *
             * This function takes ownership of the input policy, and destroys
             * the previous one.
             *
             * @param p The new policy to use during batch updates.
             */
            void setInternalPolicy(PolicyInterface * p);

            /**
             * @brief This function sets the new lambda parameter for the permanent SARSAL.
             *
             * This parameter determines how much to decrease updates for each
             * timestep in the past.
             *
             * \SA SARSAL
             *
             * The lambda parameter must be >= 0.0 and <= 1.0, otherwise the
             * function will throw an std::invalid_argument.
             *
             * @param l The new lambda parameter.
             */
            void setPermanentLambda(double l);

            /**
             * @brief This function returns the currently set lambda parameter for the permanent SARSAL.
             *
             * @return The currently set lambda parameter for the permanent SARSAL.
             */
            double getPermanentLambda() const;

            /**
             * @brief This function sets the new lambda parameter for the transient SARSAL.
             *
             * This parameter determines how much to decrease updates for each
             * timestep in the past.
             *
             * \SA SARSAL
             *
             * The lambda parameter must be >= 0.0 and <= 1.0, otherwise the
             * function will throw an std::invalid_argument.
             *
             * @param l The new lambda parameter.
             */
            void setTransientLambda(double l);

            /**
             * @brief This function returns the currently set lambda parameter for the transient SARSAL.
             *
             * @return The currently set lambda parameter for the permanent SARSAL.
             */
            double getTransientLambda() const;

            /**
             * @brief This function sets the current sample number parameter.
             *
             * @param n The new sample number parameter.
             */
            void setN(unsigned n);

            /**
             * @brief This function returns the currently set number of sampling passes during batchUpdateQ().
             *
             * @return The current number of updates().
             */
            unsigned getN() const;

            /**
             * @brief This function sets the trace cutoff parameter.
             *
             * This parameter determines when a trace is removed, as its
             * coefficient has become too small to bother updating its value.
             *
             * \sa SARSAL
             *
             * This sets the parameter for both the transient and permanent
             * SARSAL.
             *
             * @param t The new trace cutoff value.
             */
            void setTolerance(double t);

            /**
             * @brief This function returns the currently set trace cutoff parameter.
             *
             * @return The currently set trace cutoff parameter.
             */
            double getTolerance() const;

            /**
             * @brief This function returns a reference to the internal permanent QFunction.
             *
             * @return The internal permanent QFunction.
             */
            const QFunction & getPermanentQFunction() const;

            /**
             * @brief This function returns a reference to the internal transient QFunction.
             *
             * @return The internal transient QFunction.
             */
            const QFunction & getTransientQFunction() const;

            /**
             * @brief This function returns a reference to the referenced Model.
             *
             * @return The internal Model.
             */
            const M & getModel() const;

        private:
            unsigned N;
            const M & model_;
            SARSAL permanentLearning_;
            SARSAL transientLearning_;
            std::unique_ptr<PolicyInterface> internalPolicy_;
    };

    template <typename M>
    Dyna2<M>::Dyna2(const M & m, const double alpha, const double lambda, const double tolerance, const unsigned n) :
            N(n), model_(m),
            permanentLearning_(model_, alpha, lambda, tolerance),
            transientLearning_(model_, alpha, lambda, tolerance),
            internalPolicy_(new BanditPolicyAdaptor<Bandit::RandomPolicy>(model_.getS(), model_.getA()))
    {
    }

    template <typename M>
    void Dyna2<M>::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const size_t a1, const double rew) {
        // We copy the traces from the permanent SARSAL to the transient one so
        // that they will update their respective QFunctions in (nearly) the
        // same way.
        //
        // Note that this is not quite the same as it is stated in the paper.
        // Normally one would update only permanentLearning_, and transfer the
        // exact same changes directly to the QFunction of transientLearning_.
        //
        // They differ since the QFunction inside each method are different,
        // and so the updates won't exactly match. At the same time, after each
        // reset (or end of episodes) the transient memory should reset to the
        // permanent one, so this minor differences should go away.
        //
        // Ideally one would update directly the two QFunctions here, but this
        // would basically require re-implementing SARSAL both here and in the
        // batchUpdateQ method, which we avoid here for practicality.
        transientLearning_.setTraces(permanentLearning_.getTraces());
        permanentLearning_.stepUpdateQ(s, a, s1, a1, rew);
        transientLearning_.stepUpdateQ(s, a, s1, a1, rew);
    }

    template <typename M>
    void Dyna2<M>::batchUpdateQ(const size_t initS) {
        // This clearing may not be needed if this is called after stepUpdateQ
        // with the same s1 (since the set traces there will be correct then).
        // We do it anyway in case this method is called in different settings
        // and/or multiple times in a row.
        transientLearning_.clearTraces();

        size_t s = initS;
        size_t a = internalPolicy_->sampleAction(s);
        for ( unsigned i = 0; i < N; ++i ) {
            const auto [s1, rew] = model_.sampleSR(s, a);
            const size_t a1 = internalPolicy_->sampleAction(s1);

            transientLearning_.stepUpdateQ(s, a, s1, a1, rew);

            if (model_.isTerminal(s1)) {
                s = initS;
                a = internalPolicy_->sampleAction(s);
            } else {
                s = s1;
                a = a1;
            }
        }
    }

    template <typename M>
    void Dyna2<M>::resetTransientLearning() {
        transientLearning_.setQFunction(permanentLearning_.getQFunction());
    }
    template <typename M>
    void Dyna2<M>::setInternalPolicy(PolicyInterface * p) {
        internalPolicy_.reset(p);
    }

    template <typename M>
    unsigned Dyna2<M>::getN() const {
        return N;
    }

    template <typename M>
    void Dyna2<M>::setTolerance(const double t) {
        transientLearning_.setTolerance(t);
        permanentLearning_.setTolerance(t);
    }

    template <typename M>
    double Dyna2<M>::getTolerance() const {
        return permanentLearning_.getTolerance();
    }

    template <typename M>
    const QFunction & Dyna2<M>::getPermanentQFunction() const {
        return permanentLearning_.getQFunction();
    }

    template <typename M>
    const QFunction & Dyna2<M>::getTransientQFunction() const {
        return transientLearning_.getQFunction();
    }

    template <typename M>
    const M & Dyna2<M>::getModel() const {
        return model_;
    }

    template <typename M>
    void Dyna2<M>::setPermanentLambda(double l) { permanentLearning_.setLambda(l); }
    template <typename M>
    double Dyna2<M>::getPermanentLambda() const { return permanentLearning_.getLambda(); }
    template <typename M>
    void Dyna2<M>::setTransientLambda(double l) { transientLearning_.setLambda(l); }
    template <typename M>
    double Dyna2<M>::getTransientLambda() const { return transientLearning_.getLambda(); }
}

#endif
