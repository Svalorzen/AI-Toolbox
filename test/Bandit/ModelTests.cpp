#define BOOST_TEST_MODULE Bandit_Model
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <random>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Bandit/Model.hpp>

BOOST_AUTO_TEST_CASE( construction ) {
    using namespace AIToolbox;

    // Test three possible ways to construct a uniform real distribution
    Bandit::Model<std::uniform_real_distribution<double>> tupleConstructor(
        std::tuple<>{},
        std::tuple<double>{-1.0},
        std::tuple<double,double>{1.0, 2.0}
    );

    // Build the same bandit, but with the same constructor
    Bandit::Model<std::uniform_real_distribution<double>> vectorConstructor(
        std::vector<std::tuple<double,double>>{
            {0.0, 1.0},
            {-1.0, 1.0},
            {1.0, 2.0}
        }
    );

    // Check that the distributions got the correct parameters
    BOOST_CHECK_EQUAL(tupleConstructor.getA(), 3);
    BOOST_CHECK_EQUAL(vectorConstructor.getA(), 3);

    BOOST_CHECK_EQUAL(tupleConstructor.getArms().size(), 3);
    BOOST_CHECK_EQUAL(vectorConstructor.getArms().size(), 3);

    BOOST_CHECK_EQUAL(tupleConstructor.getArms()[0].a(), 0.0);
    BOOST_CHECK_EQUAL(tupleConstructor.getArms()[0].b(), 1.0);
    BOOST_CHECK_EQUAL(tupleConstructor.getArms()[1].a(), -1.0);
    BOOST_CHECK_EQUAL(tupleConstructor.getArms()[1].b(), 1.0);
    BOOST_CHECK_EQUAL(tupleConstructor.getArms()[2].a(), 1.0);
    BOOST_CHECK_EQUAL(tupleConstructor.getArms()[2].b(), 2.0);

    BOOST_CHECK_EQUAL(vectorConstructor.getArms()[0].a(), 0.0);
    BOOST_CHECK_EQUAL(vectorConstructor.getArms()[0].b(), 1.0);
    BOOST_CHECK_EQUAL(vectorConstructor.getArms()[1].a(), -1.0);
    BOOST_CHECK_EQUAL(vectorConstructor.getArms()[1].b(), 1.0);
    BOOST_CHECK_EQUAL(vectorConstructor.getArms()[2].a(), 1.0);
    BOOST_CHECK_EQUAL(vectorConstructor.getArms()[2].b(), 2.0);
}

BOOST_AUTO_TEST_CASE( sampling ) {
    using namespace AIToolbox;

    Bandit::Model<std::uniform_real_distribution<double>> bandit(
        std::vector<std::tuple<double,double>>{
            {0.0, 1.0},
            {-1.0, 1.0},
            {1.0, 2.0}
        }
    );

    BOOST_CHECK_EQUAL(bandit.getArms().size(), 3);

    for (size_t a = 0; a < 3; ++a) {
        for (auto i = 0; i < 100; ++i) {
            double r = bandit.sampleR(a);
            BOOST_CHECK(r > bandit.getArms()[a].min());
            BOOST_CHECK(r < bandit.getArms()[a].max());
        }
    }
}
