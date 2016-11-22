#define BOOST_TEST_MODULE Global
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils.hpp>

BOOST_AUTO_TEST_CASE( veccmp ) {
    using namespace AIToolbox;

    using Test = std::tuple<std::vector<double>, std::vector<double>, int>;

    std::vector<Test> data {
        Test({1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, 0),
        Test({0.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, -1),
        Test({1.0, 1.0, 3.0}, {1.0, 2.0, 3.0}, -1),
        Test({1.0, 2.0, 2.0}, {1.0, 2.0, 3.0}, -1),
        Test({1.0, 2.0, 3.0}, {0.0, 2.0, 3.0}, 1),
        Test({1.0, 2.0, 3.0}, {1.0, 1.0, 3.0}, 1),
        Test({1.0, 2.0, 3.0}, {1.0, 2.0, 2.0}, 1),
    };

    for (const auto & test : data) {
        Vector lhs = Vector::Map(std::get<0>(test).data(), 3);
        Vector rhs = Vector::Map(std::get<1>(test).data(), 3);

        BOOST_CHECK_EQUAL(AIToolbox::veccmp(lhs, rhs), std::get<2>(test));
    }
}
