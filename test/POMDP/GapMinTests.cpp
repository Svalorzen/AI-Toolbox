#define BOOST_TEST_MODULE POMDP_GapMin
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/GapMin.hpp>
#include <AIToolbox/POMDP/Types.hpp>

#include "Utils/TigerProblem.hpp"
#include <AIToolbox/Impl/Logging.hpp>

#include <iostream>

enum class Color {
    FG_DEFAULT = 39,
    FG_BLACK = 30,
    FG_RED = 31,
    FG_GREEN = 32,
    FG_YELLOW = 33,
    FG_BLUE = 34,
    FG_MAGENTA = 35,
    FG_CYAN = 36,
    FG_LIGHT_GRAY = 37,
    FG_DARK_GRAY = 90,
    FG_LIGHT_RED = 91,
    FG_LIGHT_GREEN = 92,
    FG_LIGHT_YELLOW = 93,
    FG_LIGHT_BLUE = 94,
    FG_LIGHT_MAGENTA = 95,
    FG_LIGHT_CYAN = 96,
    FG_WHITE = 97,
    BG_RED = 41,
    BG_GREEN = 42,
    BG_BLUE = 44,
    BG_DEFAULT = 49
};

std::ostream & operator<<(std::ostream& os, const Color& mod) {
    return os << "\033[" << static_cast<int>(mod) << "m";
}

void print(int severity, const char * msg) {
    if (severity != 101) return;
    // switch (severity) {
    //     case AI_SEVERITY_ERROR:   std::cout << Color::FG_RED; break;
    //     case AI_SEVERITY_WARNING: std::cout << Color::FG_YELLOW; break;
    //     case AI_SEVERITY_INFO:    std::cout << Color::FG_DEFAULT; break;
    //     case AI_SEVERITY_DEBUG:   std::cout << Color::FG_LIGHT_GRAY; break;
    // }
    std::cout << msg << std::endl;
}

using MModel = AIToolbox::MDP::Model;
using PModel = AIToolbox::POMDP::Model<MModel>;

PModel chengD35() {
    constexpr size_t S = 3, A = 3, O = 3;

    MModel::TransitionTable t(A);
    MModel::RewardTable r(S, A);
    PModel::ObservationTable o(A);

    for (size_t a = 0; a < A; ++a) {
        t[a] = AIToolbox::Matrix2D(S, S);
        o[a] = AIToolbox::Matrix2D(S, O);
    }

    t[0] <<
        0.445, 0.222, 0.333,
        0.500, 0.173, 0.327,
        0.204, 0.553, 0.243;

    t[1] <<
        0.234, 0.064, 0.702,
        0.549, 0.218, 0.233,
        0.061, 0.466, 0.473;

    t[2] <<
        0.535, 0.313, 0.152,
        0.114, 0.870, 0.016,
        0.325, 0.360, 0.315;

    o[0] <<
        0.686, 0.182, 0.132,
        0.138, 0.786, 0.076,
        0.279, 0.083, 0.638;

    o[1] <<
        0.698, 0.131, 0.171,
        0.283, 0.624, 0.093,
        0.005, 0.202, 0.793;

    o[2] <<
        0.567, 0.234, 0.199,
        0.243, 0.641, 0.116,
        0.186, 0.044, 0.770;

    r <<
        5.2, 0.8, 9.0,
        4.6, 6.8, 9.3,
        4.1, 6.9, 0.8;

    return PModel(AIToolbox::NO_CHECK, O, std::move(o), AIToolbox::NO_CHECK, S, A, std::move(t), std::move(r), 0.999);
}

PModel ejs4() {
    constexpr size_t S = 3, A = 2, O = 2;

    MModel::TransitionTable t(A);
    MModel::RewardTable r(S, A);
    PModel::ObservationTable o(A);

    for (size_t a = 0; a < A; ++a) {
        t[a] = AIToolbox::Matrix2D(S, S);
        o[a] = AIToolbox::Matrix2D(S, O);
    }
    t[0] <<
        0.1, 0.1, 0.8,
        0.2, 0.5, 0.3,
        0.7, 0.1, 0.2;

    t[1] <<
        0.1, 0.8, 0.1,
        0.7, 0.1, 0.2,
        0.1, 0.9, 0.0;

    o[0] <<
        0.7, 0.3,
        0.1, 0.9,
        0.4, 0.6;

    o[1] <<
        0.2, 0.8,
        0.4, 0.6,
        0.3, 0.7;

    r <<
        -1.0, 0.0,
         0.0,-1.0,
         0.0, 0.0;

    return PModel(AIToolbox::NO_CHECK, O, std::move(o), AIToolbox::NO_CHECK, S, A, std::move(t), std::move(r), 0.999);
}


BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox::POMDP;
#if AI_LOGGING_ENABLED == 1
    AIToolbox::AILogger = print;
#endif

    GapMin gm(0.005, 3);

    //auto model = makeTigerProblem();
    //model.setDiscount(0.95);

    auto model = chengD35();
    // auto model = ejs4();

    Belief initialBelief(model.getS());
    initialBelief.fill(1.0 / model.getS());

    const auto [lb, ub, vlist, qfun] = gm(model, initialBelief);

    std::cout << "LB = " << lb << "; UB = " << ub << "\n";
    std::cout << "Gap = " << ub - lb << "\n";
    std::cout << "VList entries =\n" << vlist.size() << "\n";
    std::cout << "QFun =\n" << qfun << "\n";
    (void)vlist;
}
