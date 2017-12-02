#define BOOST_TEST_MODULE POMDP_Utils
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Utils.hpp>
#include "Utils/OldPOMDPModel.hpp"
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

#include "Utils/TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( sosa ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

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
