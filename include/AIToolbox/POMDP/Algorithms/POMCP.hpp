#ifndef AI_TOOLBOX_POMDP_POMCP_HEADER_FILE
#define AI_TOOLBOX_POMDP_POMCP_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

#include <unordered_map>
#include <iostream>

namespace AIToolbox {
    namespace POMDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_generative_model<M>::value>::type>
        class POMCP;
#endif

        /**
         * @brief This class represents the POMCP online planner using UCB1.
         *
         * This algorithm is an online planner for POMDPs. As an online
         * planner, it needs to have a generative model of the problem. This
         * means that it only needs a way to sample transitions, observations
         * and rewards from the model, but it does not need to know directly
         * the distribution probabilities for them.
         *
         * POMCP plans for a single belief at a time. It follows the logic of
         * Monte Carlo Tree Sampling, where a tree structure is build
         * progressively and action values are deduced as averages of the
         * obtained rewards over rollouts. If the number of sample episodes is
         * high enough, it is guaranteed to converge to the optimal solution.
         *
         * At each rollout, we follow each action and observation within the
         * tree from root to leaves. During this path we chose actions using an
         * algorithm called UCT. What this does is privilege the most promising
         * actions, while guaranteeing that in the limit every action will still
         * be tried an infinite amount of times.
         *
         * Once we arrive to a leaf in the tree, we then expand it with a
         * single new node, representing a new observation we just collected.
         * We then proceed outside the tree following a random policy, but this
         * time we do not track which actions and observations we actually
         * take/obtain. The final reward obtained by this random rollout policy
         * is used to approximate the values for all nodes visited in this
         * rollout inside the tree, before leaving it.
         *
         * Since POMCP expands a tree, it can reuse work it has done if
         * multiple action requests are done in order. To do so, it simply asks
         * for the action that has been performed and its respective obtained
         * observation. Then it simply makes that root branch the new root, and
         * starts again.
         *
         * In order to avoid performing belief updates between each
         * action/observation pair, which can be expensive, POMCP uses particle
         * beliefs. These approximate the beliefs at every step, and are used
         * to select states in the rollouts.
         *
         * A weakness of this implementation is that, as every particle
         * approximation of continuous values, it will lose particles in time.
         * To fight this a possibility is to implement a particle
         * reinvigoration method, which would introduce noise in the particle
         * beliefs in order to keep them "fresh" (possibly using domain
         * knowledge).
         */
        template <typename M>
        class POMCP<M> {
            public:
                using SampleBelief = std::vector<size_t>;

                struct BeliefNode;
                using BeliefNodes = std::unordered_map<size_t, BeliefNode>;

                struct ActionNode {
                    BeliefNodes children;
                    double V = 0.0;
                    unsigned N = 0;
                };
                using ActionNodes = std::vector<ActionNode>;

                struct BeliefNode {
                    BeliefNode(size_t A) : N(0) { children.resize(A); }
                    BeliefNode(size_t s, size_t A) : belief(1, s), N(0) { children.resize(A); }
                    ActionNodes children;
                    SampleBelief belief;
                    unsigned N;
                };

                /**
                 * @brief Basic constructor.
                 *
                 * @param m The POMDP model that POMCP will operate upon.
                 * @param beliefSize The size of the initial particle belief.
                 * @param iterations The number of episodes to run before completion.
                 * @param exp The exploration constant. This parameter is VERY important to determine the final POMCP performance.
                 */
                POMCP(const M& m, size_t beliefSize, unsigned iterations, double exp);

                /**
                 * @brief This function resets the internal graph and samples for the provided belief and horizon.
                 *
                 * In general it would be better if the belief did not contain
                 * any terminal states; although not necessary, it would
                 * prevent unnecessary work from being performed.
                 *
                 * @param b The initial belief for the environment.
                 * @param horizon The horizon to plan for.
                 *
                 * @return The best action.
                 */
                size_t sampleAction(const Belief& b, unsigned horizon);

                /**
                 * @brief This function uses the internal graph to plan.
                 *
                 * This function can be called after a previous call to
                 * sampleAction with a Belief. Otherwise, it will invoke it
                 * anyway with a random belief.
                 *
                 * If a graph is already present though, this function will
                 * select the branch defined by the input action and
                 * observation, and prune the rest. The search will be started
                 * using the existing graph: this should make search faster,
                 * and also not require any belief updates.
                 *
                 * NOTE: Currently there is no particle reinvigoration
                 * implemented, so for long horizons you can expect
                 * progressively degrading performances.
                 *
                 * @param a The action taken in the last timestep.
                 * @param o The observation received in the last timestep.
                 * @param horizon The horizon to plan for.
                 *
                 * @return The best action.
                 */
                size_t sampleAction(size_t a, size_t o, unsigned horizon);

                /**
                 * @brief This function sets the new size for initial beliefs created from sampleAction().
                 *
                 * Note that this parameter does not bound particle beliefs
                 * created within the tree by result of rollouts: only the ones
                 * directly created from true Beliefs.
                 *
                 * @param beliefSize The new particle belief size.
                 */
                void setBeliefSize(size_t beliefSize);

                /**
                 * @brief This function sets the number of performed rollouts in POMCP.
                 *
                 * @param iter The new number of rollouts.
                 */
                void setIterations(unsigned iter);

                /**
                 * @brief This function sets the new exploration constant for POMCP.
                 *
                 * This parameter is EXTREMELY important to determine POMCP
                 * performance and, ultimately, convergence. In general it is
                 * better to find it empirically, by testing some values and
                 * see which one performs best. Tune this parameter, it really
                 * matters!
                 *
                 * @param exp The new exploration constant.
                 */
                void setExploration(double exp);

                /**
                 * @brief This function returns the POMDP generative model being used.
                 *
                 * @return The POMDP generative model.
                 */
                const M& getModel() const;

                /**
                 * @brief This function returns a reference to the internal graph structure holding the results of rollouts.
                 *
                 * @return The internal graph.
                 */
                const BeliefNode& getGraph() const;

                /**
                 * @brief This function returns the initial particle size for converted Beliefs.
                 *
                 * @return The initial particle count.
                 */
                size_t getBeliefSize() const;

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
                size_t S, A, beliefSize_;
                unsigned iterations_, maxDepth_;
                double exploration_;

                SampleBelief sampleBelief_;
                BeliefNode graph_;

                mutable std::default_random_engine rand_;

                // Private Methods
                size_t runSimulation(unsigned horizon);
                double simulate(BeliefNode & b, size_t s, unsigned horizon);
                double rollout(size_t s, unsigned horizon);

                template <typename Iterator>
                Iterator findBestA(Iterator begin, Iterator end);

                template <typename Iterator>
                Iterator findBestBonusA(Iterator begin, Iterator end, unsigned count);

                SampleBelief makeSampledBelief(const Belief & b);
        };

        template <typename M>
        POMCP<M>::POMCP(const M& m, size_t beliefSize, unsigned iter, double exp) : model_(m), S(model_.getS()), A(model_.getA()), beliefSize_(beliefSize), iterations_(iter),
                                                                              graph_(A), exploration_(exp), rand_(Impl::Seeder::getSeed()) {}

        template <typename M>
        size_t POMCP<M>::sampleAction(const Belief& b, unsigned horizon) {
            // Reset graph
            graph_ = BeliefNode(A);
            graph_.belief = makeSampledBelief(b);

            return runSimulation(horizon);
        }

        template <typename M>
        size_t POMCP<M>::sampleAction(size_t a, size_t o, unsigned horizon) {
            auto & obs = graph_.children[a].children;

            auto it = obs.find(o);
            if ( it == obs.end() )
                return sampleAction(Belief(S, 1.0 / S), horizon);

            // Here we need an additional step, because *it is contained by graph_.
            // If we just move assign, graph_ is first going to delete everything it
            // contains (included *it), and then we are going to move unallocated memory
            // into graph_! So we move *it outside of the graph_ hierarchy, so that
            // we can then assign safely.
            { auto tmp = std::move(it->second); graph_ = std::move(tmp); }

            if ( ! graph_.belief.size() ) {
                std::cerr << "POMCP Lost track of the belief, restarting with uniform..\n";
                return sampleAction(Belief(S, 1.0 / S), horizon);
            }

            return runSimulation(horizon);
        }

        template <typename M>
        size_t POMCP<M>::runSimulation(unsigned horizon) {
            if ( !horizon ) return 0;

            maxDepth_ = horizon;
            std::uniform_int_distribution<size_t> generator(0, graph_.belief.size()-1);

            for (unsigned i = 0; i < iterations_; ++i )
                simulate(graph_, graph_.belief.at(generator(rand_)), 0);

            auto begin = std::begin(graph_.children);
            return std::distance(begin, findBestA(begin, std::end(graph_.children)));
        }

        template <typename M>
        double POMCP<M>::simulate(BeliefNode & b, size_t s, unsigned depth) {
            // Head update
            if ( depth > 0 ) b.belief.push_back(s);
            b.N++;

            auto begin = std::begin(b.children);
            size_t a = std::distance(begin, findBestBonusA(begin, std::end(b.children), b.N));

            size_t s1, o; double rew;
            std::tie(s1, o, rew) = model_.sampleSOR(s, a);

            auto & aNode = b.children[a];

            // We only go deeper if needed (maxDepth_ is always at least 1).
            if ( depth + 1 < maxDepth_ && !model_.isTerminal(s1) ) {
                auto end = std::end(aNode.children);
                auto ot = aNode.children.find(o);

                double futureRew;
                if ( ot == end ) {
                    aNode.children.emplace(std::piecewise_construct,
                                           std::forward_as_tuple(o),
                                           std::forward_as_tuple(s1, A));
                    futureRew = rollout(s1, depth + 1);
                }
                else {
                    futureRew = simulate( ot->second, s1, depth + 1 );
                }

                rew += model_.getDiscount() * futureRew;
            }

            // Action update
            aNode.N++;
            aNode.V += ( rew - aNode.V ) / static_cast<double>(aNode.N);

            return rew;
        }

        template <typename M>
        double POMCP<M>::rollout(size_t s, unsigned depth) {
            double rew = 0.0, totalRew = 0.0, gamma = 1.0;

            std::uniform_int_distribution<size_t> generator(0, A-1);
            for ( ; depth < maxDepth_; ++depth ) {
                std::tie( s, rew ) = model_.sampleSR( s, generator(rand_) );

                totalRew += gamma * rew;
                gamma *= model_.getDiscount();
            }
            return totalRew;
        }

        template <typename M>
        template <typename Iterator>
        Iterator POMCP<M>::findBestA(Iterator begin, Iterator end) {
            return std::max_element(begin, end, [](const ActionNode & lhs, const ActionNode & rhs){ return lhs.V < rhs.V; });
        }

        template <typename M>
        template <typename Iterator>
        Iterator POMCP<M>::findBestBonusA(Iterator begin, Iterator end, unsigned count) {
            // Count here can be as low as 1.
            // Since log(1) = 0, and 0/0 = error, we add 1.0.
            double logCount = std::log(count + 1.0);
            // We use this function to produce a score for each action. This can be easily
            // substituted with something else to produce different POMCP variants.
            auto evaluationFunction = [this, logCount](const ActionNode & an){
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
        typename POMCP<M>::SampleBelief POMCP<M>::makeSampledBelief(const Belief & b) {
            SampleBelief belief;
            belief.reserve(beliefSize_);

            for ( size_t i = 0; i < beliefSize_; ++i )
                belief.push_back(sampleProbability(S, b, rand_));

            return belief;
        }

        template <typename M>
        void POMCP<M>::setBeliefSize(size_t beliefSize) {
            beliefSize_ = beliefSize;
        }

        template <typename M>
        void POMCP<M>::setIterations(unsigned iter) {
            iterations_ = iter;
        }

        template <typename M>
        void POMCP<M>::setExploration(double exp) {
            exploration_ = exp;
        }

        template <typename M>
        const M& POMCP<M>::getModel() const {
            return model_;
        }

        template <typename M>
        const typename POMCP<M>::BeliefNode& POMCP<M>::getGraph() const {
            return graph_;
        }

        template <typename M>
        size_t POMCP<M>::getBeliefSize() const {
            return beliefSize_;
        }

        template <typename M>
        unsigned POMCP<M>::getIterations() const {
            return iterations_;
        }

        template <typename M>
        double POMCP<M>::getExploration() const {
            return exploration_;
        }
    }
}

#endif
