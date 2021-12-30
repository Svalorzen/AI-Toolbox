#define BOOST_TEST_MODULE MDP_DYNA2
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <AIToolbox/MDP/Environments/CornerProblem.hpp>

BOOST_AUTO_TEST_CASE( cornerProblem ) {
    using namespace AIToolbox::MDP;

    const std::vector<std::pair<size_t, size_t>> configs = {
        {4, 4},
        {5, 5}
    };

    for (const auto & config : configs) {
        BOOST_TEST_INFO("Width: " << config.first << ", " <<
                        "Height: " << config.second << '\n');

        GridWorld grid(config.first, config.second);
        auto model = makeCornerProblem(grid);

        BOOST_CHECK(true);
    }
}
