#define BOOST_TEST_MODULE UtilsPrune
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Prune.hpp>

BOOST_AUTO_TEST_CASE( dominationPrune ) {
    using namespace AIToolbox;

    std::vector<std::vector<Vector>> data {{
        (Vector(2) <<   7.5975 , -96.9025).finished(),
        (Vector(2) <<  -8.0775 ,  -8.0775).finished(),
        (Vector(2) <<   6.03   , -16.96).finished(),
        (Vector(2) <<   7.29576, -28.3518).finished(),
        (Vector(2) <<   4.01968,  -9.78738).finished(),
        (Vector(2) << -81.2275 , -81.2275).finished(),
        (Vector(2) << -96.9025 ,   7.5975).finished(),
        (Vector(2) << -82.795  ,  -1.285).finished(),
        (Vector(2) << -81.5292 , -12.6768).finished(),
        (Vector(2) << -84.8053 ,   5.88762).finished(),
        (Vector(2) <<  -1.285  , -82.795).finished(),
        (Vector(2) << -16.96   ,   6.03).finished(),
        (Vector(2) <<  -2.8525 ,  -2.8525).finished(),
        (Vector(2) <<  -1.58674, -14.2443).finished(),
        (Vector(2) <<  -4.86282,   4.32012).finished(),
        (Vector(2) <<   5.88762, -84.8053).finished(),
        (Vector(2) <<  -9.78738,   4.01968).finished(),
        (Vector(2) <<   4.32012,  -4.86282).finished(),
        (Vector(2) <<   5.58587, -16.2546).finished(),
        (Vector(2) <<   2.3098 ,   2.3098).finished(),
        (Vector(2) << -12.6768 , -81.5292).finished(),
        (Vector(2) << -28.3518 ,   7.29576).finished(),
        (Vector(2) << -14.2443 ,  -1.58674).finished(),
        (Vector(2) << -12.9786 , -12.9786).finished(),
        (Vector(2) << -16.2546 ,   5.58587).finished(),
    }, {
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) << -100.000 ,  10.0000).finished(),
        (Vector(2) <<  10.0000 , -100.000).finished(),
    }, {
        // Test duplicates
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
    }};

    std::vector<std::vector<Vector>> solutions {{
        (Vector(2) <<   7.5975 , -96.9025).finished(),
        (Vector(2) << -16.2546 ,   5.58587).finished(),
        (Vector(2) <<   6.03   , -16.96).finished(),
        (Vector(2) <<   7.29576, -28.3518).finished(),
        (Vector(2) << -28.3518 ,   7.29576).finished(),
        (Vector(2) <<   2.3098 ,   2.3098).finished(),
        (Vector(2) << -96.9025 ,   7.5975).finished(),
        (Vector(2) <<   5.58587, -16.2546).finished(),
        (Vector(2) <<   4.32012,  -4.86282).finished(),
        (Vector(2) <<  -4.86282,   4.32012).finished(),
        (Vector(2) <<  -16.96  ,   6.03).finished(),
    }, {
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) << -100.000 ,  10.0000).finished(),
        (Vector(2) <<  10.0000 , -100.000).finished(),
    }, {
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
    }};

    for (size_t i = 0; i < data.size(); ++i) {
        auto & d = data[i];
        auto & s = solutions[i];

        auto test = extractDominated(2, std::begin(d), std::end(d));
        auto comparer = [](const auto & lhs, const auto & rhs) {
            return veccmp(lhs, rhs) < 0;
        };

        std::sort(std::begin(s), std::end(s), comparer);
        std::sort(std::begin(d), test, comparer);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(s), std::end(s),
                                      std::begin(d), test);
    }
}
