#define BOOST_TEST_MODULE Factored_MDP_LinearProgramming
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/MDP/Algorithms/LinearProgramming.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/Factored/MDP/Utils.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

BOOST_AUTO_TEST_CASE( solver ) {
    using namespace AIToolbox::Factored;
    using namespace AIToolbox::Factored::MDP;

    auto problem = makeSysAdminUniRing(2, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.3);

    // Create and setup the bases to use for the ValueFunction.
    auto vf = ValueFunction();

    for (size_t s = 0; s < problem.getS().size(); s += 2) {
        for (size_t i = 0; i < 9; ++i) {
            vf.values.bases.emplace_back(BasisFunction{{s, s+1}, AIToolbox::Vector(9)});
            vf.values.bases.back().values.setZero();
            vf.values.bases.back().values[i] = 1.0;
        }
    }

    auto solver = LinearProgramming();

    QFunction q;
    std::tie(vf.weights, q) = solver(problem, vf.values);

    // Since we have no information on what the weights should actually be,
    // here I'm comparing against the weights I got the first time I managed to
    // make this algorithm work correctly. This test is less about 100%
    // correctness, and more about warning me in case I touch something that
    // changes the result.

    // Check we got the correct number of weights.
    BOOST_CHECK_EQUAL(vf.weights.size(), vf.values.bases.size());

    AIToolbox::Vector solution(18);
    solution <<
        0.14477187137671121, 0.0000000000000000,
        0.0000000000000000, 0.57452249094478225,
        0.30396019119172546, 0.0000000000000000,
        0.0000000000000000, 0.0000000000000000,
        0.0000000000000000, 5.6461029837006631,
        5.5013311123207824, 5.5013311123237534,
        6.0758536032641155, 5.8052913035050926,
        5.5013311123270263, 5.5013311123214406,
        5.5013311123205346, 5.5013311123202353;

    for (size_t i = 0; i < 18; ++i)
        BOOST_CHECK_EQUAL(vf.weights[i], solution[i]);

    auto qSolution = bellmanBackup(problem, vf);

    // Here we check that the output QFunction is the same as the one we can
    // compute ourselves.
    BOOST_CHECK_EQUAL(qSolution.bases.size(), q.bases.size());
    for (size_t i = 0; i < qSolution.bases.size(); ++i) {
        const auto & sb = qSolution.bases[i];
        const auto & qb = q.bases[i];

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(sb.tag), std::end(sb.tag), std::begin(qb.tag), std::end(qb.tag));
        BOOST_CHECK_EQUAL_COLLECTIONS(
            std::begin(sb.actionTag), std::end(sb.actionTag),
            std::begin(qb.actionTag), std::end(qb.actionTag)
        );
        // This check is relatively fragile like this because it depends on
        // floating point comparisons and multiplication orderings.. for now we
        // leave it like this.
        BOOST_CHECK_EQUAL(sb.values, qb.values);
    }
}
