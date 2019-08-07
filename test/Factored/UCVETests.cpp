#define BOOST_TEST_MODULE Factored_Bandit_UCVE

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/UCVE.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace fm = AIToolbox::Factored;
namespace fb = fm::Bandit;

BOOST_AUTO_TEST_CASE( testing ) {
    fm::Action A{2,2,2,2,2};

    constexpr double logtA = 11.9829;

    fb::UCVE::Factor ucveVectors;
    // FIXME: C++ 20 fix for aggregates
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.194357          , 0.0031348  }, {{0,1},{0,0}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.0669014         , 0.0140845  }, {{0,1},{1,0}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.25              , 0.000273598}, {{0,1},{0,1}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.224084          , 0.00104712 }, {{0,1},{1,1}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.183535          , 0.00302115 }, {{1,2},{0,0}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.25              , 0.000269906}, {{1,2},{1,0}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.0466102         , 0.0169492  }, {{1,2},{0,1}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.225414          , 0.00110497 }, {{1,2},{1,1}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.193182          , 0.0227273  }, {{2,3},{0,0}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.0697674         , 0.0232558  }, {{2,3},{1,0}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.25              , 0.000250501}, {{2,3},{0,1}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.225299          , 0.00108578 }, {{2,3},{1,1}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.19186           , 0.0232558  }, {{3,4},{0,0}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.25              , 0.0263158  }, {{3,4},{1,0}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.0511364         , 0.0227273  }, {{3,4},{0,1}}});
    ucveVectors.emplace_back(fb::UCVE::Entry{{0.224256          , 0.000205128}, {{3,4},{1,1}}});

    fb::UCVE ucve;
    auto [a, v] = ucve(A, logtA, ucveVectors);

    // Solve via bruteforce
    fm::PartialAction bestAction;
    fb::UCVE::V bestValue(2); bestValue.setZero();
    double bestV = 0.0;

    auto value = [](const auto & v) {
        return v[0] + std::sqrt(0.5 * v[1] * logtA);
    };

    fb::UCVE::V helper(2);
    fm::PartialFactorsEnumerator jointActions(A);
    while (jointActions.isValid()) {
        auto & jointAction = *jointActions;
        helper.setZero();

        for (const auto & e : ucveVectors)
            if (fm::match(e.tag, jointAction))
                helper += e.v;

        double tmpV = value(helper);
        if (tmpV > bestV) {
            bestValue = helper;
            bestAction = jointAction;
            bestV = tmpV;
        }

        jointActions.advance();
    }

    // Check solutions match
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a), std::end(a),
                                  std::begin(bestAction.second), std::end(bestAction.second));
    BOOST_CHECK_EQUAL(v, bestValue);
}
