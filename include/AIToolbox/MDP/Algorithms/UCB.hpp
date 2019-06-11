#ifndef AI_TOOLBOX_MDP_UCB_HEADER_FILE
#define AI_TOOLBOX_MDP_UCB_HEADER_FILE

#include <AIToolbox/MDP/Algorithms/MCTS.hpp>

namespace AIToolbox::MDP {
    class UCB {
        public:
            virtual ~UCB() = default;

            template <typename Iterator>
            Iterator findBestA(Iterator begin, Iterator end) const;

            template <typename Iterator>
            Iterator findBestBonusA(Iterator begin, Iterator end, unsigned count, double exp) const;

            template <typename M, typename ST = size_t, typename AT = size_t>
            void initializeActions(typename MCTS<M, UCB, ST, AT>::StateNode &parent, const ST &s, const M &m) const;

            template <typename M, typename ST = size_t, typename AT = size_t>
            AT getRandomAction(const ST &s, const M &m, RandomEngine &r) const;

    };

    template <typename Iterator>
    Iterator UCB::findBestA(Iterator begin, Iterator end) const {
        return std::max_element(begin, end, [](const auto & lhs, const auto & rhs){ return lhs.V < rhs.V; });
    }

    template <typename Iterator>
    Iterator UCB::findBestBonusA(Iterator begin, Iterator end, const unsigned count, const double exp) const {
        // Count here can be as low as 1.
        // Since log(1) = 0, and 0/0 = error, we add 1.0.
        const double logCount = std::log(count + 1.0);
        // We use this function to produce a score for each action. This can be easily
        // substituted with something else to produce different POMCP variants.
        const auto evaluationFunction = [exp, logCount](const auto & an){
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

    template <typename M, typename ST, typename AT>
    void UCB::initializeActions(typename MCTS<M, UCB, ST, AT>::StateNode &parent, const ST &, const M &m) const {
        if (parent.children.size() == 0) {
            size_t A = m.getA();
            parent.children.resize(A);
            for (size_t i = 0; i < A; i++)
                parent.children.at(i).action = i;
        }
    }

    template <typename M, typename ST, typename AT>
    AT UCB::getRandomAction(const ST&, const M &m, RandomEngine &r) const {
        std::uniform_int_distribution<size_t> generator(0, m.getA() - 1);
        return generator(r);
    }
};

#endif
