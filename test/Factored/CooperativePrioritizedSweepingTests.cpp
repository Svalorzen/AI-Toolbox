#define BOOST_TEST_MODULE Factored_MDP_CooperativePrioritizedSweeping
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/CooperativePrioritizedSweeping.hpp>
#include <AIToolbox/Factored/MDP/CooperativeExperience.hpp>
#include <AIToolbox/Factored/MDP/CooperativeRLModel.hpp>

#include "Utils/SysAdmin.hpp"

namespace aif = AIToolbox::Factored;
namespace fm = AIToolbox::Factored::MDP;

BOOST_AUTO_TEST_CASE( simple_rule_update ) {
    auto problem = makeSysAdminUniRing(2, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.3);

    fm::CooperativeExperience exp(problem.getS(), problem.getA(), problem.getTransitionFunction().nodes);
    fm::CooperativeRLModel model(exp, 0.9);

    std::vector<std::vector<size_t>> domains{
        {0, 1},
        {2, 3}
    };

    fm::CooperativePrioritizedSweeping ps(model, domains);

    aif::State s(model.getS().size());
    aif::Action a(model.getA().size());
    auto [s1, r] = model.sampleSRs(s, a);

    ps.stepUpdateQ(s, a, s1, r);
}
