#ifndef AI_TOOLBOX_POMDP_POLICY_HEADER_FILE
#define AI_TOOLBOX_POMDP_POLICY_HEADER_FILE

#include <tuple>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/PolicyInterface.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class represents a POMDP Policy.
     *
     * This class currently represents a basic Policy adaptor for a
     * POMDP::ValueFunction. What this class does is to extract the policy
     * tree contained within a POMDP::ValueFunction. The idea is that, at
     * each horizon, the ValueFunction contains a set of applicable
     * solutions (alpha vectors) for the POMDP. At each Belief point, only
     * one of those vectors applies.
     *
     * This class finds out at every belief which is the vector that
     * applies, and returns the appropriate action. At the same time, it
     * provides facilities to follow the chosen vector along the tree
     * (since future actions depend on the observations obtained by the
     * agent).
     */
    class Policy : public PolicyInterface<size_t, Belief, size_t> {
        public:
            using Base = PolicyInterface<size_t, Belief, size_t>;
            /**
             * @brief Basic constrctor.
             *
             * This constructor initializes the internal ValueFunction as
             * having only the horizon 0 no values solution. This is most
             * useful if the Policy needs to be read from a file.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             * @param o The number of possible observations the agent could make.
             */
            Policy(size_t s, size_t a, size_t o);

            /**
             * @brief Basic constrctor.
             *
             * This constructor copies the implied policy contained in a
             * ValueFunction.  Keep in mind that the policy stored within a
             * ValueFunction is non-stochastic in nature, since for each
             * state it can only save a single action.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             * @param o The number of possible observations the agent could make.
             * @param v The ValueFunction used as a basis for the Policy.
             */
            Policy(size_t s, size_t a, size_t o, const ValueFunction & v);

            // This may be implemented, but probably not since it would be mostly impossible to convert
            // from a POMDP policy format to another.
            // Policy(const PolicyInterface<Belief> & p);

            /**
             * @brief This function chooses a random action for belief b, following the policy distribution.
             *
             * Note that this will sample from the highest horizon that the
             * Policy was computed for.
             *
             * @param b The sampled belief of the policy.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction(const Belief & b) const override;

            /**
             * @brief This function chooses a random action for belief b when horizon steps are missing, following the policy distribution.
             *
             * There are a couple of differences between this sampling
             * function and the simpler version. The first one is that this
             * function is actually able to sample from different
             * timesteps, since this class is able to maintain a full
             * policy tree over time.
             *
             * The second difference is that it returns two values. The
             * first one is the requested action.  The second return value
             * is an id that allows the policy to compute more efficiently
             * the sampled action during the next timestep, if provided to
             * the Policy together with the obtained observation.
             *
             * @param b The sampled belief of the policy.
             * @param horizon The requested horizon, meaning the number of timesteps missing until
             * the end of the "episode". horizon 0 will return a valid, non-specified action.
             *
             * @return A tuple containing the chosen action, plus an id useful to sample an action
             * more efficiently at the next timestep, if required.
             */
            std::tuple<size_t, size_t> sampleAction(const Belief & b, unsigned horizon) const;

            /**
             * @brief This function chooses a random action after performing a sampled action and observing observation o, for a particular horizon.
             *
             * This sampling function is provided in case an already
             * sampled action has been performed, an observation
             * registered, and now a new action is needed for the next
             * timestep. Using this function is highly recommended, as no
             * belief update is necessary, and no lookup in a possibly very
             * long list of VEntries required.
             *
             * Note that this function works if and only if the horizon is
             * going to be 1 (one) less than the value used for the
             * previous sampling, otherwise anything could happen. This
             * does not mean that the calls depend on each other (the
             * function is "pure" in that sense), just that to obtain
             * meaningful values back the horizon should be decreased.
             *
             * To keep things simple, the id does not store internally the
             * needed horizon value, and you are requested to keep track of
             * it yourself.
             *
             * An example of usage for this function would be:
             *
             * ~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
             * horizon = 3;
             * // First sample
             * auto result = sampleAction(belief, horizon);
             * // We do the action, something happens, we get an observation.
             * size_t observation = performAction(std::get<0>(result));
             * --horizon;
             * // We sample again, after reducing the horizon, with the previous id.
             * result = sampleAction(std::get<1>(result), observation, horizon);
             * ~~~~~~~~~~~~~~~~~~~~~~~
             *
             * @param id An id returned from a previous call of sampleAction.
             * @param o The observation obtained after performing a previously sampled action.
             * @param horizon The new horizon, equal to the old sampled horizon - 1.
             *
             * @return A tuple containing the chosen action, plus an id useful to sample an action
             * more efficiently at the next timestep, if required.
             */
            std::tuple<size_t, size_t> sampleAction(size_t id, size_t o, unsigned horizon) const;

            /**
             * @brief This function returns the probability of taking the specified action in the specified belief.
             *
             * @param b The selected belief.
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified belief.
             */
            virtual double getActionProbability(const Belief & b, const size_t & a) const override;

            /**
             * @brief This function returns the probability of taking the specified action in the specified belief.
             *
             * @param b The selected belief.
             * @param a The selected action.
             * @param horizon The requested horizon, meaning the number of timesteps missing until
             * the end of the "episode".
             *
             * @return The probability of taking the selected action in the specified belief in the specified horizon.
             */
            double getActionProbability(const Belief & b, size_t a, unsigned horizon) const;

            /**
             * @brief This function returns the number of observations possible for the agent.
             *
             * @return The total number of observations.
             */
            size_t getO() const;

            /**
             * @brief This function returns the highest horizon available within this Policy.
             *
             * Note that all functions that accept an horizon as a
             * parameter DO NOT check the bounds of that variable. In
             * addition, note that while for S,A,O getters you get a number
             * that exceeds by 1 the values allowed (since counting starts
             * from 0), here the bound is actually included in the limit,
             * as horizon 0 does not really do anything.
             *
             * Example: getH() returns 5. This means that 5 is the highest
             * allowed parameter for an horizon in any other Policy method.
             *
             * @return The highest horizon policied.
             */
            size_t getH() const;

            /**
             * @brief This function returns the internally stored ValueFunction.
             *
             * @return The internally stored ValueFunction.
             */
            const ValueFunction & getValueFunction() const;

        private:
            // H holds the available max horizon for this Policy.
            size_t O, H;

            ValueFunction policy_;

            friend std::istream& operator>>(std::istream &is, Policy & p);
    };
}

#endif
