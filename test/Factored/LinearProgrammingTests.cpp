#define BOOST_TEST_MODULE Factored_MDP_LinearProgramming
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/MDP/Algorithms/LinearProgramming.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/Factored/MDP/Utils.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

#include "Utils/SysAdmin.hpp"

BOOST_AUTO_TEST_CASE( solver ) {
    auto problem = makeSysAdminUniRing(2, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.3);

    // Create and setup the bases to use for the ValueFunction.
    auto vf = afm::ValueFunction();

    // From the original paper it is not 100% clear whether we should create 3
    // separate bases per state element, or 9 bases per agent (Status x Load).
    //
    // Unfortunately I can't seem to be able to learn a reasonable policy with
    // the first approach, so we're doing the second - which actually learns
    // pretty nicely.

    // for (size_t s = 0; s < problem.getS().size(); ++s) {
    //     vf.values.bases.emplace_back(aif::BasisFunction{{s}, ai::Vector(3)});
    //     vf.values.bases.back().values << 1.0, 0.0, 0.0;

    //     vf.values.bases.emplace_back(aif::BasisFunction{{s}, ai::Vector(3)});
    //     vf.values.bases.back().values << 0.0, 1.0, 0.0;

    //     vf.values.bases.emplace_back(aif::BasisFunction{{s}, ai::Vector(3)});
    //     vf.values.bases.back().values << 0.0, 0.0, 1.0;
    // }

    for (size_t s = 0; s < problem.getS().size(); s += 2) {
        for (size_t i = 0; i < 9; ++i) {
            vf.values.bases.emplace_back(aif::BasisFunction{{s, s+1}, ai::Vector(9)});
            vf.values.bases.back().values.setZero();
            vf.values.bases.back().values[i] = 1.0;
        }
    }


    auto solver = afm::LinearProgramming();

    afm::QFunction q;
    std::tie(vf.weights, q) = solver(problem, vf.values);

    // Since we have no information on what the weights should actually be,
    // here I'm comparing against the weights I got the first time I managed to
    // make this algorithm work correctly. This test is less about 100%
    // correctness, and more about warning me in case I touch something that
    // changes the result.

    // Check we got the correct number of weights.
    BOOST_CHECK_EQUAL(vf.weights.size(), vf.values.bases.size());

    ai::Vector solution(18);
    solution <<
                5.7908748550780462238662948948331177234649658203125,       5.646102983700050259585623280145227909088134765625,
                5.64610298370053254046752044814638793468475341796875,      6.2206254746486262519056253950111567974090576171875,
                5.950063174889979933368522324599325656890869140625,        5.64610298369968877096880532917566597461700439453125,
                5.6461029837009473197895204066298902034759521484375,       5.64610298369642560345482706907205283641815185546875,
                5.646102983704164302025674260221421718597412109375,        0.0,
                -0.1447718713768413323350614518858492374420166015625,      -0.1447718713768439136035937053748057223856449127197265625,
                0.4297506195682692098358756993548013269901275634765625,    0.159188319814997980561344093075604178011417388916015625,
                -0.144771871376843941359169321003719232976436614990234375, -0.144771871376841609890817608174984343349933624267578125,
                -0.144771871376842720113842233331524766981601715087890625, -0.1447718713768416931575444550617248751223087310791015625;

    for (size_t i = 0; i < 18; ++i)
        BOOST_CHECK_EQUAL(vf.weights[i], solution[i]);

    vf.weights = solution;

    auto qSolution = afm::bellmanBackup(problem, vf);

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
