#define BOOST_TEST_MODULE UtilsPolytope
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Polytope.hpp>

BOOST_AUTO_TEST_CASE( extractBestUsefulPointsTest ) {
    using namespace AIToolbox;

    std::vector<Point> points = {
        (Point(2) << 0.969799   , 0.0302013).finished(),
        (Point(2) << 0.85       , 0.15).finished(),
        (Point(2) << 0.00546559 , 0.994534).finished(),
        (Point(2) << 0.15       , 0.85).finished(),
        (Point(2) << 0.5        , 0.5).finished(),
        (Point(2) << 0.0302013  , 0.969799).finished(),
        (Point(2) << 0.994534   , 0.00546559).finished(),
    };

    std::vector<Hyperplane> vl = {
        (Hyperplane(2) << 3, 3).finished(),
        (Hyperplane(2) << 4, 1).finished(),
        (Hyperplane(2) << 1, 4).finished(),
        (Hyperplane(2) << 5, -5).finished(),
        (Hyperplane(2) << -5, 5).finished(),
    };

    auto bound = extractBestUsefulPoints(std::begin(points), std::end(points), std::begin(vl), std::end(vl));

    BOOST_CHECK_EQUAL(std::distance(std::begin(points), bound), vl.size());

    BOOST_CHECK(checkEqualSmall((*bound)[0], 0.969799) || checkEqualSmall((*bound)[0], 0.0302013));
    ++bound;
    BOOST_CHECK(checkEqualSmall((*bound)[0], 0.969799) || checkEqualSmall((*bound)[0], 0.0302013));
}

BOOST_AUTO_TEST_CASE( naive_vertex_enumeration ) {
    using namespace AIToolbox;

    std::vector<Vector> alphas = {
        {(Vector(3) << 1.0, 0.0, 0.0).finished()},
        {(Vector(3) << 0.0, 1.0, 0.0).finished()},
        {(Vector(3) << 0.0, 0.0, 1.0).finished()},
    };

    std::vector<std::pair<Vector, double>> solutions = {
        {(Vector(3) << 1.0/3.0, 1.0/3.0, 1.0/3.0).finished(), 1.0/3.0},
        {(Vector(3) << 0.5, 0.5, 0.0).finished(), 0.5},
        {(Vector(3) << 0.0, 0.5, 0.5).finished(), 0.5},
        {(Vector(3) << 0.5, 0.0, 0.5).finished(), 0.5},
    };

    // We look for all vertices from all possible angles.  We are going to get
    // duplicates for now, but that's not a problem as long as all vertices are
    // enumerated.
    auto vertices = findVerticesNaive(std::begin(alphas), std::end(alphas), std::begin(alphas), std::end(alphas));

    // Now we check against the solution, both ways: all vertices in the
    // solution must be somewhere in the new list, and all vertices in the new
    // list must be in the solution.
    for (const auto & v : vertices) {
        bool found = false;
        for (const auto & s : solutions) {
            if (veccmpSmall(v.first, s.first) == 0 && checkEqualSmall(v.second, s.second)) {
                found = true;
                break;
            }
        }
        BOOST_CHECK(found);
    }

    for (const auto & s : solutions) {
        bool found = false;
        for (const auto & v : vertices) {
            if (veccmpSmall(v.first, s.first) == 0 && checkEqualSmall(v.second, s.second)) {
                found = true;
                break;
            }
        }
        BOOST_CHECK(found);
    }
}

BOOST_AUTO_TEST_CASE( optimistic_value_discovery ) {
    using namespace AIToolbox;

    std::vector<std::pair<Vector, double>> beliefs = {
        {(Vector(3) << 1.0, 0.0, 0.0).finished(), 10.0},
        {(Vector(3) << 0.0, 1.0, 0.0).finished(), 5.0},
        {(Vector(3) << 0.0, 0.0, 1.0).finished(), -10.0},
    };

    Vector b(3);
    b.fill(1.0/3.0);

    constexpr double solution = (10.0 + 5.0 - 10.0) / 3.0;

    const auto v = computeOptimisticValue(b, std::begin(beliefs), std::end(beliefs));

    BOOST_CHECK(checkEqualGeneral(v, solution));
}
