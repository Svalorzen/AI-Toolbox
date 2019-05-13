#ifndef AI_TOOLBOX_MDP_UCB_HEADER_FILE
#define AI_TOOLBOX_MDP_UCB_HEADER_FILE

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/Algorithms/MCTS.hpp>

namespace AIToolbox::MDP {
    class UCB {
        public:
            template <typename Iterator>
            static Iterator findBestA(Iterator begin, Iterator end);

            template <typename Iterator>
            static Iterator findBestBonusA(Iterator begin, Iterator end, unsigned count, double exp);

            static void initializeActions(StateNode &parent, const Model &m);
    };

    template <typename Iterator>
    Iterator UCB::findBestA(Iterator begin, Iterator end) {
        return std::max_element(begin, end, [](const ActionNode & lhs, const ActionNode & rhs){ return lhs.V < rhs.V; });
    }

    template <typename Iterator>
    Iterator UCB::findBestBonusA(Iterator begin, Iterator end, const unsigned count, const double exp) {
        // Count here can be as low as 1.
        // Since log(1) = 0, and 0/0 = error, we add 1.0.
        const double logCount = std::log(count + 1.0);
        // We use this function to produce a score for each action. This can be easily
        // substituted with something else to produce different POMCP variants.
        const auto evaluationFunction = [exp, logCount](const ActionNode & an){
            return an.V + exp * std::sqrt( logCount / an.N );
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

    void UCB::initializeActions(StateNode &parent, const Model &m) {
        size_t A = m.getA();
        parent.children.resize(A);
        for (size_t i = 0; i < A; i++)
            parent.children.at(i).action = i;
    }
};

#endif
