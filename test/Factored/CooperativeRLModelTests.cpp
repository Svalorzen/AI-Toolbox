#define BOOST_TEST_MODULE Factored_MDP_CooperativeRLModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/MDP/CooperativeRLModel.hpp>

#include "Utils/SysAdmin.hpp"

BOOST_AUTO_TEST_CASE( construction ) {
    auto model = makeSysAdminBiRing(7, 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1);

    afm::CooperativeExperience exp(model.getS(), model.getA(), model.getTransitionFunction().nodes);
    afm::CooperativeRLModel rl(exp, 0.9);

    aif::State s{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    aif::State a{2, 2, 2, 2, 2, 2, 2};

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(model.getS()), std::end(model.getS()),
                                  std::begin(s), std::end(s));

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(model.getA()), std::end(model.getA()),
                                  std::begin(a), std::end(a));

    BOOST_CHECK_EQUAL(model.getTransitionFunction().nodes.size(), model.getS().size());
}
