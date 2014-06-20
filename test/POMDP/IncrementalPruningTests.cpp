#define BOOST_TEST_MODULE POMDP_Incremental_Pruning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include "TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox;

    auto model = makeTigerProblem();
    model.setDiscount(0.95);

    // We solve the problem for an horizon of 15
    // (enough to find out everything is working
    // correclty). In addition, for higher horizons
    // floating point precision results in this library
    // obtaining more VEntries than Cassandra's solver
    // (all but the new ones are the same as his). This
    // is probably due to this library using a higher
    // precision floating point error.
    unsigned horizon = 15;
    POMDP::IncrementalPruning solver(horizon);
    auto solution = solver(model);

    BOOST_CHECK_EQUAL(std::get<0>(solution), true);

    auto & vf = std::get<1>(solution);
    auto vlist = vf[horizon];

    // This is the correct solution
    POMDP::VList truth = {
        std::make_tuple(MDP::Values({-91.2960462266272685383228236, 18.7039537733727385671045340 }), 1u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-18.6647017255443259386993304, 15.6400519533182436049401076 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-9.2894374007652391611600251 , 15.2372532254717185651315958 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-9.1073394270104568448687132 , 15.2281474451412623949408953 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-9.0719322042323611299252661 , 15.2255070494616866483283957 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-9.0672250658686337487779383 , 15.2249840340879405431451232 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-8.3962682348594448455969541 , 15.1437884651934897561886828 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-6.7962580607883573691196943 , 14.9335465202520900618310407 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-6.7682403332796141626204189 , 14.9297173844606003711987796 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-6.7671170519224235206934281 , 14.9295586137774591861671070 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-6.6978832295572425792329341 , 14.9185440901028592008970008 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-6.6354675954011774763330322 , 14.9063263567442980672694830 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({3.5978798439707659895248071  , 12.6727487351471701515492896 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({3.5992293947774589568666670  , 12.6724513959643871885418775 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({3.6317770490051213272408859  , 12.6650349386826608366618530 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({6.0145190916743329623272984  , 12.1003254654811005508463495 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({6.7813661312330246744295437  , 11.9029852210666327039234602 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({6.7861614182552472129827947  , 11.9016412132853162120227353 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({6.8103730465170482233361327  , 11.8915302737326413762275479 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({6.8937259587727552911928797  , 11.8396199916215500991256704 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({9.1762520832364025125116314  , 10.1851615813664171383834400 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({9.3272070958679975660743366  , 10.0684107617843388027267792 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({9.3329596904102434251626619  , 10.0635752364107489142952545 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({9.7284247446776745960050903  , 9.7284247446776745960050903  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({10.0635752364107489142952545 , 9.3329596904102434251626619  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({10.0684107617843388027267792 , 9.3272070958679975660743366  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({10.1851615813664171383834400 , 9.1762520832364025125116314  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({11.8396199916215500991256704 , 6.8937259587727552911928797  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({11.8915302737326413762275479 , 6.8103730465170482233361327  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({11.9016412132853162120227353 , 6.7861614182552472129827947  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({11.9029852210666327039234602 , 6.7813661312330246744295437  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({12.1003254654811005508463495 , 6.0145190916743329623272984  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({12.6650349386826608366618530 , 3.6317770490051213272408859  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({12.6724513959643871885418775 , 3.5992293947774589568666670  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({12.6727487351471701515492896 , 3.5978798439707659895248071  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({14.9063263567442980672694830 , -6.6354675954011774763330322 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({14.9185440901028592008970008 , -6.6978832295572425792329341 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({14.9295586137774591861671070 , -6.7671170519224235206934281 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({14.9297173844606003711987796 , -6.7682403332796141626204189 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({14.9335465202520900618310407 , -6.7962580607883573691196943 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({15.1437884651934897561886828 , -8.3962682348594448455969541 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({15.2249840340879405431451232 , -9.0672250658686337487779383 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({15.2255070494616866483283957 , -9.0719322042323611299252661 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({15.2281474451412623949408953 , -9.1073394270104568448687132 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({15.2372532254717185651315958 , -9.2894374007652391611600251 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({15.6400519533182436049401076 , -18.6647017255443259386993304}), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({18.7039537733727385671045340 , -91.2960462266272685383228236}), 2u, POMDP::VObs(0)),
    };

    // Make sure we can actually compare them
    std::sort(std::begin(vlist), std::end(vlist));
    std::sort(std::begin(truth), std::end(truth));

    BOOST_CHECK_EQUAL(vlist.size(), truth.size());
    // We check each entry by itself to avoid checking observations
    for ( size_t i = 0; i < vlist.size(); ++i ) {
        BOOST_CHECK_EQUAL(std::get<POMDP::ACTION>(vlist[i]), std::get<POMDP::ACTION>(truth[i]));

        auto & values      = std::get<POMDP::VALUES>(vlist[i]);
        auto & truthValues = std::get<POMDP::VALUES>(truth[i]);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(values), std::end(values), std::begin(truthValues), std::end(truthValues));
    }
}

BOOST_AUTO_TEST_CASE( undiscountedHorizon ) {
    using namespace AIToolbox;
    // NOTE: This test has been added since I noticed that the action results
    // for the undiscounted tiger problem for an horizon of 2 gave me different
    // results from both Cassandra's code and what is published in the literature.
    // In particular, there is a single ValueFunction which suggests to act, while
    // in the literature usually in this step all ValueFunctions point to the
    // listening action. This alternative solution is actually correct, as in an
    // undiscounted scenario it doesn't matter, if the belief in a state is high
    // enough, whether we act now and listen later, or vice-versa.

    auto model = makeTigerProblem();
    model.setDiscount(1.0);

    unsigned horizon = 2;
    POMDP::IncrementalPruning solver(horizon);
    auto solution = solver(model);

    BOOST_CHECK_EQUAL(std::get<0>(solution), true);

    auto & vf = std::get<1>(solution);
    auto vlist = vf[horizon];

    // This is the correct solution
    POMDP::VList truth = {
        // Action 10 here (which does not exist) is used to mark the values for which both listening or acting is a correct
        // action. We will not test those.
        std::make_tuple(MDP::Values({-101.0000000000000000000000000, 9.0000000000000000000000000   }), 10u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-16.8500000000000014210854715 , 7.3499999999999996447286321   }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-2.0000000000000000000000000  , -2.0000000000000000000000000  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({7.3499999999999996447286321   , -16.8500000000000014210854715 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({9.0000000000000000000000000   , -101.0000000000000000000000000}), 10u, POMDP::VObs(0)),
    };

    // Make sure we can actually compare them
    std::sort(std::begin(vlist), std::end(vlist));
    std::sort(std::begin(truth), std::end(truth));

    BOOST_CHECK_EQUAL(vlist.size(), truth.size());
    // We check each entry by itself to avoid checking observations
    for ( size_t i = 0; i < vlist.size(); ++i ) {
        // Avoid checking actions with multiple possible answers.
        if ( std::get<POMDP::ACTION>(truth[i]) != 10u )
            BOOST_CHECK_EQUAL(std::get<POMDP::ACTION>(vlist[i]), std::get<POMDP::ACTION>(truth[i]));

        auto & values      = std::get<POMDP::VALUES>(vlist[i]);
        auto & truthValues = std::get<POMDP::VALUES>(truth[i]);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(values), std::end(values), std::begin(truthValues), std::end(truthValues));
    }
}
