#define BOOST_TEST_MODULE Factored_MDP_CooperativeModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

BOOST_AUTO_TEST_CASE( construction ) {
    using namespace AIToolbox::Factored;
    using namespace AIToolbox::Factored::MDP;

    auto model = makeSysAdminBiRing(7, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);
    State s{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    State a{2, 2, 2, 2, 2, 2, 2};

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(model.getS()), std::end(model.getS()),
                                  std::begin(s), std::end(s));

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(model.getA()), std::end(model.getA()),
                                  std::begin(a), std::end(a));

    BOOST_CHECK_EQUAL(model.getTransitionFunction().nodes.size(), model.getS().size());
}

BOOST_AUTO_TEST_CASE( sampling ) {
    using namespace AIToolbox::Factored;
    using namespace AIToolbox::Factored::MDP;

    // Status transition params.
    double pFailBase = 0.1, pFailBonus = 0.2, pDeadBase = 0.3, pDeadBonus = 0.4;
    // Load transition params.
    double pLoad = 0.2, pDoneG = 0.2, pDoneF = 0.1;

    auto problem = makeSysAdminBiRing(5, pFailBase, pFailBonus, pDeadBase,
            pDeadBonus, pLoad, pDoneG, pDoneF);

    State s{
        0, 0,  // gi
        1, 1,  // fl
        1, 0,  // fi
        2, 2,  // dd
        0, 0   // gi
    };
    Action a{0, 0, 0, 0, 0};

    State s1 = s;
    std::vector<unsigned> counters{0,0,0,0,0};
    constexpr auto trials = 10000;
    double totReward = 0.0;
    for (size_t i = 0; i < trials; ++i) {
        auto rw = problem.sampleSR(s, a, &s1);
        auto [xxx, rw2] = problem.sampleSRs(s, a);
        (void)xxx;

        BOOST_CHECK_EQUAL(rw2.sum(), rw);
        BOOST_CHECK_EQUAL(rw2.size(), problem.getRewardFunction().bases.size());

        totReward += rw;

        // Status checks
        counters[0] += s1[0];
        BOOST_CHECK(s1[0] != 2);

        counters[1] += s1[2];
        BOOST_CHECK(s1[2] != 0);

        counters[2] += s1[4];
        BOOST_CHECK(s1[4] != 0);

        BOOST_CHECK(s1[6] == 2);

        counters[4] += s1[8];
        BOOST_CHECK(s1[8] != 2);

        // Load checks
        BOOST_CHECK(s1[1] != 2);
        BOOST_CHECK(s1[3] != 0);
        BOOST_CHECK(s1[5] != 2);
        BOOST_CHECK(s1[7] == 0);
        BOOST_CHECK(s1[9] != 2);
    }

    constexpr auto tolerance = 300;
    std::vector<double> solutions{
        trials * (pFailBase + pFailBonus / 2),
        trials + trials * (pDeadBase + pFailBonus / 2),
        trials + trials * (pDeadBase + pFailBonus / 2 + pDeadBonus / 2),
        0,
        trials * (pFailBase + pDeadBonus / 2)
    };

    BOOST_CHECK(counters[0] < solutions[0] + tolerance);
    BOOST_CHECK(counters[0] > solutions[0] - tolerance);

    BOOST_CHECK(counters[1] < solutions[1] + tolerance);
    BOOST_CHECK(counters[1] > solutions[1] - tolerance);

    BOOST_CHECK(counters[2] < solutions[2] + tolerance);
    BOOST_CHECK(counters[2] > solutions[2] - tolerance);

    BOOST_CHECK(counters[3] < solutions[3] + tolerance);
    BOOST_CHECK(counters[3] > solutions[3] - tolerance);

    BOOST_CHECK(totReward < 10000 * pDoneF + 100);
    BOOST_CHECK(totReward > 10000 * pDoneF - 100);
}
