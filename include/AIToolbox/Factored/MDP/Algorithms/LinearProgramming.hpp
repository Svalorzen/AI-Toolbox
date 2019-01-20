#ifndef AI_TOOLBOX_FACTORED_MDP_LINEAR_PROGRAMMING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_LINEAR_PROGRAMMING_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>

namespace AIToolbox { class LP; }

namespace AIToolbox::Factored::MDP {

    class LinearProgramming {
        public:
            Vector operator()(const CooperativeModel & m, const FactoredVector & h, bool addConstantBasis) {
                const auto g = backProject(m.getS(), m.getA(), m.getTransitionFunction(), h);
                return *solveLP(m, g, h, addConstantBasis);
            }

        private:
            using Rule = std::pair<PartialValues, size_t>;
            using Rules = std::vector<Rule>;
            using Graph = FactorGraph<Rules>;

            std::optional<Vector> solveLP(const CooperativeModel & m, const Factored2DMatrix & g, const FactoredVector & h, bool addConstantBasis);
            void removeState(const Factors & F, Graph & graph, size_t s, LP & lp, std::vector<size_t> & finalFactors);
    };

}

#endif
