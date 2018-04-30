#define BOOST_TEST_MODULE Factored_MDP_JointActionLearner
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Algorithms/QLearning.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/JointActionLearner.hpp>

namespace aif = AIToolbox::Factored;
namespace fm = AIToolbox::Factored::MDP;
using JAL = fm::JointActionLearner;

BOOST_AUTO_TEST_CASE( simple_test ) {
    constexpr size_t S = 3;
    const aif::Action A{2, 2, 2};

    constexpr double discount = 0.9;
    constexpr double learningRate = 0.1;

    fm::JointActionLearner l(S, A, 0, discount, learningRate);

    aif::Action a{0, 0, 0};

    l.stepUpdateQ(0, a, 1, 10.0);

    BOOST_CHECK_EQUAL(l.getSingleQFunction()(0,0), 1.0);

    l.stepUpdateQ(0, a, 1, 10.0);

    a[1] = 1;
    l.stepUpdateQ(0, a, 1, 6.0);

    BOOST_CHECK_EQUAL(l.getSingleQFunction()(0,0), (1.9 * 2.0 + 0.6) / 3.0);

    l.stepUpdateQ(2, a, 0, 10.0);

    BOOST_CHECK_EQUAL(l.getSingleQFunction()(2,0), 1.0 + 0.09 * 1.9);
}

