#define BOOST_TEST_MODULE Factored_MDP_CooperativePrioritizedSweeping
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/LinearProgramming.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/CooperativePrioritizedSweeping.hpp>
#include <AIToolbox/Factored/MDP/CooperativeExperience.hpp>
#include <AIToolbox/Factored/MDP/CooperativeRLModel.hpp>

#include <AIToolbox/Factored/MDP/Policies/EpsilonPolicy.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

namespace ai = AIToolbox;
namespace aif = AIToolbox::Factored;
namespace fm = AIToolbox::Factored::MDP;

BOOST_AUTO_TEST_CASE( simple_rule_update ) {
    auto problem = fm::makeSysAdminUniRing(2, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.3);

    fm::CooperativeExperience exp(problem.getS(), problem.getA(), problem.getTransitionFunction().nodes);
    fm::CooperativeRLModel model(exp, 0.95);

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
    for (size_t t = 0; t < 1000; ++t) {
        auto a = ep.sampleAction(s);
        auto [s1, x] = problem.sampleSR(s, a);
        (void)x;

        for (size_t l = 1; l < model.getS().size(); l += 2)
            r[l] = (s1[l] == fm::SysAdminEnums::Done);

        const auto & ids = exp.record(s, a, s1, r);
        model.sync(ids);

        ps.stepUpdateQ(s, a, s1, r);
        ps.batchUpdateQ();
        s = std::move(s1);
    }

    // Create and setup the bases to use for the ValueFunction.
    auto vf = fm::ValueFunction();

    for (size_t s = 0; s < problem.getS().size(); s += 2) {
        for (size_t i = 0; i < 9; ++i) {
            vf.values.bases.emplace_back(aif::BasisFunction{{s, s+1}, ai::Vector(9)});
            vf.values.bases.back().values.setZero();
            vf.values.bases.back().values[i] = 1.0;
        }
    }

    auto solver = fm::LinearProgramming();

    fm::QFunction q;
    std::tie(vf.weights, q) = solver(problem, vf.values);

    aif::PartialFactorsEnumerator se(problem.getS());
    aif::PartialFactorsEnumerator ae(problem.getA());
    double maxDiff = std::numeric_limits<double>::lowest();

    fm::QGreedyPolicy p2(model.getS(), model.getA(), q);
    while (se.isValid()) {
        ae.reset();
        while (ae.isValid()) {
            double currDiff = std::fabs(q.getValue(problem.getS(), problem.getA(), se->second, ae->second) -
                              ps.getQFunction().getValue(problem.getS(), problem.getA(), se->second, ae->second));
            if (currDiff > maxDiff)
                maxDiff = currDiff;
            ae.advance();
        }
        se.advance();
    }
    // This test is not very informative but not much we can do about it.. this
    // is mostly to see that the output at least makes somewhat sense.
    BOOST_TEST_INFO(maxDiff);
    BOOST_CHECK(maxDiff < 2);
}
