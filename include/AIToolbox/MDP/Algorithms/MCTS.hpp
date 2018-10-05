#ifndef AI_TOOLBOX_MDP_MCTS_HEADER_FILE
#define AI_TOOLBOX_MDP_MCTS_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

#include <unordered_map>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the MCTS online planner using UCB1.
     *
     * This algorithm is an online planner for MDPs. As an online planner,
     * it needs to have a generative model of the problem. This means that
     * it only needs a way to sample transitions and rewards from the
     * model, but it does not need to know directly the distribution
     * probabilities for them.
     *
     * MCTS plans for a single state at a time. It builds a tree structure
     * progressively and action values are deduced as averages of the
     * obtained rewards over rollouts. If the number of sample episodes is
     * high enough, it is guaranteed to converge to the optimal solution.
     *
     * At each rollout, we follow each action and resulting state within the
     * tree from root to leaves. During this path we chose actions using an
     * algorithm called UCT. What this does is privilege the most promising
     * actions, while guaranteeing that in the limit every action will still
     * be tried an infinite amount of times.
     *
     * Once we arrive to a leaf in the tree, we then expand it with a
     * single new node, representing a new state for the path we just
     * followed. We then proceed outside the tree following a random
     * policy, but this time we do not track which actions and states
     * we actually experience. The final reward obtained by this random
     * rollout policy is used to approximate the values for all nodes
     * visited in this rollout inside the tree, before leaving it.
     *
     * Since MCTS expands a tree, it can reuse work it has done if
     * multiple action requests are done in order. To do so, it simply asks
     * for the action that has been performed and its respective new state.
     * Then it simply makes that root branch the new root, and starts
     * again.
     */
    template <typename M>
    class MCTS {
        static_assert(is_generative_model_v<M>, "This class only works for generative MDP models!");

        public:
            using SampleBelief = std::vector<size_t>;

            struct StateNode;
            using StateNodes = std::unordered_map<size_t, StateNode>;

            struct ActionNode {
                StateNodes children;
                double V = 0.0;
                unsigned N = 0;
            };
            using ActionNodes = std::vector<ActionNode>;

            struct StateNode {
                StateNode() : N(0) {}
                ActionNodes children;
                unsigned N;
            };

            /**
             * @brief Basic constructor.
             *
             * @param m The MDP model that MCTS will operate upon.
             * @param iterations The number of episodes to run before completion.
             * @param exp The exploration constant. This parameter is VERY important to determine the final MCTS performance.
             */
            MCTS(const M& m, unsigned iterations, double exp);

            /**
             * @brief This function resets the internal graph and samples for the provided state and horizon.
             *
             * @param s The initial state for the environment.
             * @param horizon The horizon to plan for.
             *
             * @return The best action.
             */
            size_t sampleAction(size_t s, unsigned horizon);

            /**
             * @brief This function uses the internal graph to plan.
             *
             * This function can be called after a previous call to
             * sampleAction with a state. Otherwise, it will invoke it
             * anyway with the provided next state.
             *
             * If a graph is already present though, this function will
             * select the branch defined by the input action and
             * observation, and prune the rest. The search will be started
             * using the existing graph: this should make search faster.
             *
             * @param a The action taken in the last timestep.
             * @param s1 The state experienced after the action was taken.
             * @param horizon The horizon to plan for.
             *
             * @return The best action.
             */
            size_t sampleAction(size_t a, size_t s1, unsigned horizon);

            /**
             * @brief This function sets the number of performed rollouts in MCTS.
             *
             * @param iter The new number of rollouts.
             */
            void setIterations(unsigned iter);

            /**
             * @brief This function sets the new exploration constant for MCTS.
             *
             * This parameter is EXTREMELY important to determine MCTS
             * performance and, ultimately, convergence. In general it is
             * better to find it empirically, by testing some values and
             * see which one performs best. Tune this parameter, it really
             * matters!
             *
             * @param exp The new exploration constant.
             */
            void setExploration(double exp);

            /**
             * @brief This function returns the MDP generative model being used.
             *
             * @return The MDP generative model.
             */
            const M& getModel() const;

            /**
             * @brief This function returns a reference to the internal graph structure holding the results of rollouts.
             *
             * @return The internal graph.
             */
            const StateNode& getGraph() const;

            /**
             * @brief This function returns the number of iterations performed to plan for an action.
             *
             * @return The number of iterations.
             */
            unsigned getIterations() const;

            /**
             * @brief This function returns the currently set exploration constant.
             *
             * @return The exploration constant.
             */
            double getExploration() const;

        private:
            const M& model_;
            size_t S, A;
            unsigned iterations_, maxDepth_;
            double exploration_;

            StateNode graph_;

            mutable RandomEngine rand_;

            // Private Methods
            size_t runSimulation(size_t s, unsigned horizon);
            double simulate(StateNode & sn, size_t s, unsigned horizon);
            double rollout(size_t s, unsigned horizon);

            template <typename Iterator>
            Iterator findBestA(Iterator begin, Iterator end);

            template <typename Iterator>
            Iterator findBestBonusA(Iterator begin, Iterator end, unsigned count);
    };

    template <typename M>
    MCTS<M>::MCTS(const M& m, const unsigned iter, const double exp) :
            model_(m), S(model_.getS()), A(model_.getA()), iterations_(iter),
            exploration_(exp), graph_(), rand_(Impl::Seeder::getSeed()) {}

    template <typename M>
    size_t MCTS<M>::sampleAction(const size_t s, const unsigned horizon) {
        // Reset graph
        graph_ = StateNode();
        graph_.children.resize(A);

        return runSimulation(s, horizon);
    }

    template <typename M>
    size_t MCTS<M>::sampleAction(const size_t a, const size_t s1, const unsigned horizon) {
        auto & states = graph_.children[a].children;

        auto it = states.find(s1);
        if ( it == states.end() )
            return sampleAction(s1, horizon);

        // Here we need an additional step, because *it is contained by graph_.
        // If we just move assign, graph_ is first going to delete everything it
        // contains (included *it), and then we are going to move unallocated memory
        // into graph_! So we move *it outside of the graph_ hierarchy, so that
        // we can then assign safely.
        { auto tmp = std::move(it->second); graph_ = std::move(tmp); }

        // We resize here in case we didn't have time to sample the new
        // head node. In this case, the new head may not have children.
        // This would break the UCT call.
        graph_.children.resize(A);

        return runSimulation(s1, horizon);
    }

    template <typename M>
    size_t MCTS<M>::runSimulation(const size_t s, const unsigned horizon) {
        if ( !horizon ) return 0;

        maxDepth_ = horizon;

        for (unsigned i = 0; i < iterations_; ++i )
            simulate(graph_, s, 0);

        auto begin = std::begin(graph_.children);
        return std::distance(begin, findBestA(begin, std::end(graph_.children)));
    }

    template <typename M>
    double MCTS<M>::simulate(StateNode & sn, const size_t s, const unsigned depth) {
        // Head update
        sn.N++;

        auto begin = std::begin(sn.children);
        const size_t a = std::distance(begin, findBestBonusA(begin, std::end(sn.children), sn.N));

        auto [s1, rew] = model_.sampleSR(s, a);

        auto & aNode = sn.children[a];

        // We only go deeper if needed (maxDepth_ is always at least 1).
        if ( depth + 1 < maxDepth_ && !model_.isTerminal(s1) ) {
            const auto end = std::end(aNode.children);
            auto it = aNode.children.find(s1);

            double futureRew;
            if ( it == end ) {
                // Touch node to create it
                aNode.children[s1];
                futureRew = rollout(s1, depth + 1);
            }
            else {
                // Since most memory is allocated on the leaves,
                // we do not allocate on node creation but only when
                // we are actually descending into a node. If the node
                // already has memory this should not do anything in
                // any case.
                it->second.children.resize(A);
                futureRew = simulate( it->second, s1, depth + 1 );
            }

            rew += model_.getDiscount() * futureRew;
        }

        // Action update
        aNode.N++;
        aNode.V += ( rew - aNode.V ) / static_cast<double>(aNode.N);

        return rew;
    }

    template <typename M>
    double MCTS<M>::rollout(size_t s, unsigned depth) {
        double rew = 0.0, totalRew = 0.0, gamma = 1.0;

        std::uniform_int_distribution<size_t> generator(0, A-1);
        for ( ; depth < maxDepth_; ++depth ) {
            std::tie( s, rew ) = model_.sampleSR( s, generator(rand_) );
            totalRew += gamma * rew;

            if (model_.isTerminal(s))
                return totalRew;

            gamma *= model_.getDiscount();
        }
        return totalRew;
    }

    template <typename M>
    template <typename Iterator>
    Iterator MCTS<M>::findBestA(Iterator begin, Iterator end) {
        return std::max_element(begin, end, [](const ActionNode & lhs, const ActionNode & rhs){ return lhs.V < rhs.V; });
    }

    template <typename M>
    template <typename Iterator>
    Iterator MCTS<M>::findBestBonusA(Iterator begin, Iterator end, const unsigned count) {
        // Count here can be as low as 1.
        // Since log(1) = 0, and 0/0 = error, we add 1.0.
        const double logCount = std::log(count + 1.0);
        // We use this function to produce a score for each action. This can be easily
        // substituted with something else to produce different POMCP variants.
        const auto evaluationFunction = [this, logCount](const ActionNode & an){
            return an.V + exploration_ * std::sqrt( logCount / an.N );
        };

        auto bestIterator = begin++;
        double bestValue = evaluationFunction(*bestIterator);

        for ( ; begin < end; ++begin ) {
            double actionValue = evaluationFunction(*begin);
            if ( actionValue > bestValue ) {
                bestValue = actionValue;
                bestIterator = begin;
            }
        }

        return bestIterator;
    }

    template <typename M>
    void MCTS<M>::setIterations(const unsigned iter) {
        iterations_ = iter;
    }

    template <typename M>
    void MCTS<M>::setExploration(const double exp) {
        exploration_ = exp;
    }

    template <typename M>
    const M& MCTS<M>::getModel() const {
        return model_;
    }

    template <typename M>
    const typename MCTS<M>::StateNode& MCTS<M>::getGraph() const {
        return graph_;
    }

    template <typename M>
    unsigned MCTS<M>::getIterations() const {
        return iterations_;
    }

    template <typename M>
    double MCTS<M>::getExploration() const {
        return exploration_;
    }
}

#endif
