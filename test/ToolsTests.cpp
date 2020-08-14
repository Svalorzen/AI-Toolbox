#define BOOST_TEST_MODULE Tools
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Tools/Statistics.hpp>
#include <AIToolbox/Utils/Core.hpp>

BOOST_AUTO_TEST_CASE( mean_variance ) {
    std::vector<std::vector<double>> data {
        {19, 11, 8, 7, 7, 20, 0, 5, 4, 13},
        {10, 5, 3, 12, 20, 5, 19, 5, 18, 6},
        {26, 30, 49, 33, 13, 36, 20, 31, 22, 18},
        {20, 9, 4, 6, 10, 11, 12, 11, 12, 10}
    };

    std::array<std::vector<double>, 4> truth {{
        {18.75,  13.75,  16.  ,  14.5 ,  12.5 ,  18.  ,  12.75,  13.  ,  14.  ,  11.75}, // means
        {18.75,  32.5 ,  48.5 ,  63.  ,  75.5 ,  93.5 , 106.25, 119.25, 133.25, 145.  }, // cum means
        {6.601767440112787, 11.116804097101529, 22.105806175452337, 12.609520212918492,  5.567764362830022, 13.490737563232042, 9.215023964519391, 12.328828005937952,  7.831560082980487, 5.057996968497839}, // std
        {6.601767440112787, 17.09775813764288 , 38.61346915261564 , 50.37856687124    , 50.34878350069642 , 62.740205078827515, 67.17328337962944 , 79.20595516331669 , 84.79927279562406 , 88.79564553888139}, // cum std
    }};

    const size_t length = data[0].size();

    AIToolbox::Statistics stats(length);

    for (const auto & d : data) {
        for (size_t i = 0; i < length; ++i)
            stats.record(d[i], i);
    }

    auto output = stats.process();

    BOOST_CHECK_EQUAL(output.size(), length);

    for (size_t i = 0; i < length; ++i) {
        const auto [mean, cummean, std, cumstd] = output[i];

        BOOST_CHECK(AIToolbox::checkEqualGeneral(mean,    truth[0][i]));
        BOOST_CHECK(AIToolbox::checkEqualGeneral(cummean, truth[1][i]));
        BOOST_CHECK(AIToolbox::checkEqualGeneral(std,     truth[2][i]));
        BOOST_CHECK(AIToolbox::checkEqualGeneral(cumstd,  truth[3][i]));
    }
}
