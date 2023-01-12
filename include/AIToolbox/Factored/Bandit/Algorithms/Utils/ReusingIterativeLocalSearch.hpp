#ifndef AI_TOOLBOX_FACTORED_BANDIT_REUSING_ITERATIVE_LOCAL_SEARCH_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_REUSING_ITERATIVE_LOCAL_SEARCH_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/LocalSearch.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class approximately finds the best joint action with Reusing Iterative Local Search.
     *
     * This class is mostly a wrapper around LocalSearch. The idea is to avoid
     * local optima by adding noise to the solution found by LocalSearch, or
     * alternatively restart from random points to see whether we can find a
     * better solution.
     *
     * In addition, we cache the best action found, so that we can re-use it as
     * a starting point if needed. The idea is that if the graph to solve has
     * changed in a relatively minor way, it is likely that the optimal
     * solution will be close to the one found previously. Note that this
     * caching is optional in case it is known that the graph changed
     * substantially (or we want to solve a different graph).
     */
    class ReusingIterativeLocalSearch {
        public:
            using Result = std::tuple<Action, double>;
            using Graph = LocalSearch::Graph;

            /**
             * @brief Basic constructor.
             *
             * The default parameters are provided mostly so that it's possible
             * to initialize RILS in classes internally without knowing the
             * explicit parameters.
             *
             * It's quite likely they won't work for your problem, so remember
             * to pass something that makes sense!
             *
             * @param resetActionProbability For each trial, the probability of testing a random action.
             * @param randomizeFactorProbability For each trial, the probability for each factor of being randomized from the current best.
             * @param trialNum The number of trials to perform before returning.
             * @param forceResetAction Whether force restarting from a random action rather than using the last returned best action.
             */
            ReusingIterativeLocalSearch(double resetActionProbability = 0.3, double randomizeFactorProbability = 0.1, unsigned trialNum = 10, bool forceResetAction = true);

            /**
             * @brief This function approximately finds the best Action-value pair for the provided Graph.
             *
             * On first call, this function optimizes over a single randomly
             * sampled initial action. Subsequently it will optimize using the
             * last best action as a starting point, unless it is explicitly
             * reset.
             *
             * \sa operator()(const Action &, Graph &, bool)
             *
             * @param A The action space of the agents.
             * @param graph The graph to perform RILS on.
             *
             * @return A tuple containing the best Action and its value over the input rules.
             */
            Result operator()(const Action & A, const Graph & graph);

            /**
             * @brief This function returns the currently set probability for testing a random action.
             */
            double getResetActionProbability() const;

            /**
             * @brief This function sets the probability for testing a random action.
             */
            void setResetActionProbability(double resetActionProbability);

            /**
             * @brief This function returns the currently set probability of randomizing each factor.
             */
            double getRandomizeFactorProbability() const;

            /**
             * @brief This function sets the probability of randomizing each factor.
             */
            void setRandomizeFactorProbability(double randomizeFactorProbability);

            /**
             * @brief This function returns the currently set number of trials to perform.
             */
            unsigned getTrialNum() const;

            /**
             * @brief This function sets the number of trials to perform.
             */
            void setTrialNum(unsigned trialNum);

            /**
             * @brief This function returns whether we always restart from a random action at each optimization.
             *
             * If this is false, we always start from the lastly returned best action.
             */
            bool getForceResetAction() const;

            /**
             * @brief This function sets whether we always restart from a random action at each optimization.
             */
            void setForceResetAction(bool forceResetAction);

        private:
            // Parameters
            double resetActionProbability_;
            double randomizeFactorProbability_;
            unsigned trialNum_;
            bool forceResetAction_;

            // Caches
            Action action_, newAction_;

            // Nested local search
            LocalSearch ls_;

            RandomEngine rnd_;
    };
};

#endif
