#define BOOST_TEST_MODULE FactoredMDP_UCVE

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/FactoredMDP/Utils.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/Utils/UCVE.hpp>

namespace fm = AIToolbox::FactoredMDP;

BOOST_AUTO_TEST_CASE( testing ) {
    fm::Action A{2,2,2,2,2};

    constexpr double logtA = 11.9829;

    fm::UCVE::Entries ucveVectors;
    ucveVectors.emplace_back(fm::PartialAction{{0,1},{0,0}}, fm::UCVE::V{0.194357          , 0.0031348  });
    ucveVectors.emplace_back(fm::PartialAction{{0,1},{1,0}}, fm::UCVE::V{0.0669014         , 0.0140845  });
    ucveVectors.emplace_back(fm::PartialAction{{0,1},{0,1}}, fm::UCVE::V{0.25              , 0.000273598});
    ucveVectors.emplace_back(fm::PartialAction{{0,1},{1,1}}, fm::UCVE::V{0.224084          , 0.00104712 });
    ucveVectors.emplace_back(fm::PartialAction{{1,2},{0,0}}, fm::UCVE::V{0.183535          , 0.00302115 });
    ucveVectors.emplace_back(fm::PartialAction{{1,2},{1,0}}, fm::UCVE::V{0.25              , 0.000269906});
    ucveVectors.emplace_back(fm::PartialAction{{1,2},{0,1}}, fm::UCVE::V{0.0466102         , 0.0169492  });
    ucveVectors.emplace_back(fm::PartialAction{{1,2},{1,1}}, fm::UCVE::V{0.225414          , 0.00110497 });
    ucveVectors.emplace_back(fm::PartialAction{{2,3},{0,0}}, fm::UCVE::V{0.193182          , 0.0227273  });
    ucveVectors.emplace_back(fm::PartialAction{{2,3},{1,0}}, fm::UCVE::V{0.0697674         , 0.0232558  });
    ucveVectors.emplace_back(fm::PartialAction{{2,3},{0,1}}, fm::UCVE::V{0.25              , 0.000250501});
    ucveVectors.emplace_back(fm::PartialAction{{2,3},{1,1}}, fm::UCVE::V{0.225299          , 0.00108578 });
    ucveVectors.emplace_back(fm::PartialAction{{3,4},{0,0}}, fm::UCVE::V{0.19186           , 0.0232558  });
    ucveVectors.emplace_back(fm::PartialAction{{3,4},{1,0}}, fm::UCVE::V{0.25              , 0.0263158  });
    ucveVectors.emplace_back(fm::PartialAction{{3,4},{0,1}}, fm::UCVE::V{0.0511364         , 0.0227273  });
    ucveVectors.emplace_back(fm::PartialAction{{3,4},{1,1}}, fm::UCVE::V{0.224256          , 0.000205128});

    fm::UCVE ucve(A, logtA);
    auto a_v = ucve(ucveVectors);
}
