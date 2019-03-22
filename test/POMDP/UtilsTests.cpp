#define BOOST_TEST_MODULE POMDP_Utils
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Utils.hpp>
#include "Utils/OldPOMDPModel.hpp"
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

#include <AIToolbox/POMDP/Environments/TigerProblem.hpp>

BOOST_AUTO_TEST_CASE( sosa ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;
    using namespace TigerProblemEnums;

    auto problem = makeTigerProblem();
    OldPOMDPModel<MDP::Model> oldProblem = problem;

    auto psosa = makeSOSA(problem);
    auto oldpsosa = makeSOSA(oldProblem);

    for (size_t a = 0; a < problem.getA(); ++a)
        for (size_t o = 0; o < problem.getO(); ++o)
            BOOST_CHECK(psosa[a][o] == oldpsosa[a][o]);

    BOOST_CHECK_EQUAL(psosa[A_LISTEN][TIG_RIGHT](TIG_RIGHT, TIG_RIGHT), 1.0 - listenError);
    BOOST_CHECK_EQUAL(psosa[A_LISTEN][TIG_RIGHT](TIG_RIGHT, TIG_LEFT),  0.0);
    BOOST_CHECK_EQUAL(psosa[A_LISTEN][TIG_RIGHT](TIG_LEFT, TIG_LEFT),   listenError);
    BOOST_CHECK_EQUAL(psosa[A_LISTEN][TIG_RIGHT](TIG_LEFT, TIG_RIGHT),  0.0);

    BOOST_CHECK_EQUAL(psosa[A_LISTEN][TIG_LEFT](TIG_RIGHT, TIG_RIGHT), listenError);
    BOOST_CHECK_EQUAL(psosa[A_LISTEN][TIG_LEFT](TIG_RIGHT, TIG_LEFT),  0.0);
    BOOST_CHECK_EQUAL(psosa[A_LISTEN][TIG_LEFT](TIG_LEFT, TIG_LEFT),   1.0 - listenError);
    BOOST_CHECK_EQUAL(psosa[A_LISTEN][TIG_LEFT](TIG_LEFT, TIG_RIGHT),  0.0);

    // All the rest is 0.25 since the observation when opening a door is random
    // (50/50), and the new state is too (50/50);
    for (size_t a = 1; a < problem.getA(); ++a)
        for (size_t o = 0; o < problem.getO(); ++o)
            for (size_t s = 0; s < problem.getS(); ++s)
                for (size_t s1 = 0; s1 < problem.getS(); ++s1)
                    BOOST_CHECK_EQUAL(psosa[a][o](s, s1), 0.25);
}

BOOST_AUTO_TEST_CASE( beliefUpdate ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    auto problem = makeTigerProblem();
    OldPOMDPModel<MDP::Model> oldProblem = problem;

    Belief b(2); b << 0.5, 0.5;
    Belief solution(2); solution << 0.85, 0.15;

    auto resultEigen = updateBelief(problem, b, 0, 0);
    auto resultOld   = updateBelief(oldProblem, b, 0, 0);

    for (size_t s = 0; s < problem.getS(); ++s)
        BOOST_CHECK_EQUAL(resultEigen[s], resultOld[s]);

    BOOST_CHECK(checkEqualProbability(resultEigen, solution));
}

BOOST_AUTO_TEST_CASE( beliefUpdateUnnormalized ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    auto problem = makeTigerProblem();
    OldPOMDPModel<MDP::Model> oldProblem = problem;

    Belief b(2); b << 0.5, 0.5;
    Belief solution(2); solution << 0.425, 0.075;

    auto resultEigen = updateBeliefUnnormalized(problem, b, 0, 0);
    auto resultOld   = updateBeliefUnnormalized(oldProblem, b, 0, 0);

    BOOST_CHECK(checkEqualProbability(resultEigen, resultOld));
    BOOST_CHECK(checkEqualProbability(resultEigen, solution));
}

BOOST_AUTO_TEST_CASE( beliefUpdatePartial ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    auto problem = makeTigerProblem();
    OldPOMDPModel<MDP::Model> oldProblem = problem;

    Belief b(2); b << 0.5, 0.5;

    auto partialEigen = updateBeliefPartial(problem, b, 0);
    auto partialOld   = updateBeliefPartial(oldProblem, b, 0);

    BOOST_CHECK(checkEqualProbability(partialEigen, partialOld));

    for (size_t o = 0; o < problem.getO(); ++o) {
        auto partialEigen1 = updateBeliefPartialNormalized(problem, partialEigen, 0, o);
        auto partialEigen2 = updateBeliefPartialUnnormalized(problem, partialEigen, 0, o);

        auto partialOld1 = updateBeliefPartialNormalized(oldProblem, partialEigen, 0, o);
        auto partialOld2 = updateBeliefPartialUnnormalized(oldProblem, partialEigen, 0, o);

        auto resultEigen1 = updateBelief(problem, b, 0, o);
        auto resultEigen2 = updateBeliefUnnormalized(problem, b, 0, o);

        BOOST_CHECK(checkEqualProbability(partialEigen1, partialOld1));
        BOOST_CHECK(checkEqualProbability(partialEigen2, partialOld2));

        BOOST_CHECK(checkEqualProbability(resultEigen1, partialEigen1));
        BOOST_CHECK(checkEqualProbability(resultEigen2, partialEigen2));
    }
}
