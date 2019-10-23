#ifndef AI_TOOLBOX_FACTORED_CPS_QUEUE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_CPS_QUEUE_HEADER_FILE

#include <random>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This class is used as the priority queue for CooperativePrioritizedSweeping.
     *
     * This class performs a similar work as that done by Tries, but in a much
     * more constrained way, so that it can be as fast as possible.
     *
     * This class assumes keys are always the parent sets of some variable as
     * represented in a DDN.
     *
     * When doing the reconstruction, we select a single rule from each node,
     * since all nodes's parents are by definition incompatible with each
     * other. We always the best possible rule, and then randomly iterate over
     * nodes, either picking their best possible rule if compatible or the best
     * available alternative after picking a random local action.
     */
    class CPSQueue {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor uses the inputs to construct the internal
             * representation for priority rules, following the structure of
             * the ddn.
             *
             * @param graph The ddn of the transition function of the problem.
             */
            CPSQueue(const DDNGraph & graph);

            /**
             * @brief This function updates the probability of the input parent set.
             *
             * This function takes ids directly to avoid having to pass through
             * the toIndexPartial() function.
             *
             * It increases the priority of the rule by 'p', and if necessary
             * updates the maxes for the associated action/node so they can be
             * more easily found later.
             *
             * @param i The id of the node.
             * @param a The id of the local joint action.
             * @param s The id of the local parent states.
             * @param p The priority to add.
             */
            void update(size_t i, size_t a, size_t s, double p);

            /**
             * @brief This function sets the input State and Action with the highest priority combination.
             *
             * The highest priority parent set is always picked. Then, we
             * randomly iterate over nodes, either picking their best possible
             * rule if compatible or the best available alternative after
             * picking a random local action.
             *
             * This is the best we can do, as picking the true highest
             * combination is NP-hard, and we want this to be as fast as
             * possible so we can do many batch updates in
             * CooperativePrioritizedSweeping.
             *
             * Note that some elements may not be picked. These will be left
             * with the value of the size of their respective space (so you can
             * find them and decide what to do with them).
             *
             * @param s The State to output, preallocated.
             * @param a The Action to output, preallocated.
             *
             * @return A set of Entry that match the input and each other, the Factors obtained by combining the input with the returned set.
             */
            void reconstruct(State & s, Action & a);

            /**
             * @brief This function returns the priority of the highest parent set of the selected node.
             *
             * @param i The id of the selected node.
             *
             * @return The priority of the highest parent set.
             */
            double getNodeMaxPriority(size_t i) const;

            /**
             * @brief This function returns how many non-zero priority parent sets there are.
             *
             * The result is pre-computed during updates and reconstructions,
             * so calling this function is always fast.
             *
             * @return The number of current non-zero priority rules.
             */
            unsigned getNonZeroPriorities() const;

        private:
            const DDNGraph & graph_;
            unsigned nonZeroPriorities_;

            std::vector<size_t> order_;

            struct ActionNode {
                Vector priorities;
                double maxV;
                size_t maxS;
            };
            struct Node {
                double maxV;
                size_t maxA;
                std::vector<size_t> order;
                std::vector<ActionNode> nodes;
            };

            std::vector<Node> nodes_;

            mutable std::ranlux24_base rand_; // Fastest engine possible, don't care about quality
    };
}

#endif
