#define BOOST_TEST_MODULE Factored_MDP_CooperativePrioritizedSweeping
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/CooperativePrioritizedSweeping.hpp>
#include <AIToolbox/Factored/MDP/CooperativeExperience.hpp>
#include <AIToolbox/Factored/MDP/CooperativeRLModel.hpp>

#include <AIToolbox/Factored/MDP/Policies/EpsilonPolicy.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>

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

    fm::QGreedyPolicy p(model.getS(), model.getA(), ps.getQFunction());
    fm::EpsilonPolicy ep(p);

    aif::State s(model.getS().size());
    aif::Rewards r(model.getS().size());
    r.setZero();
    for (size_t t = 0; t < 100; ++t) {
        std::cout << " #######\n";
        auto a = ep.sampleAction(s);
        auto [s1, x] = problem.sampleSR(s, a);

        for (size_t l = 1; l < model.getS().size(); l += 2)
            r[l] = s1[l] == 2;

        const auto & ids = exp.record(s, a, s1, r);
        model.sync(ids);

        ps.stepUpdateQ(s, a, s1, r);
        ps.batchUpdateQ();
        s = std::move(s1);

        std::cout << ps.getQFunction().bases[0].values << '\n';
        std::cout << ps.getQFunction().bases[1].values << '\n';
    }
}
