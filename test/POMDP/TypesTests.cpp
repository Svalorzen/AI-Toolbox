#define BOOST_TEST_MODULE POMDP_Types
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include "Utils/OldPOMDPModel.hpp"
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

BOOST_AUTO_TEST_CASE( positives ) {
    namespace MDP = AIToolbox::MDP;
    using namespace AIToolbox::POMDP;

    static_assert(IsGenerativeModel<Model<MDP::Model>>);
    static_assert(IsModel<Model<MDP::Model>>);
    static_assert(IsModelEigen<Model<MDP::Model>>);

    static_assert(IsGenerativeModel<OldPOMDPModel<MDP::Model>>);
    static_assert(IsModel<OldPOMDPModel<MDP::Model>>);
    static_assert(!IsModelEigen<OldPOMDPModel<MDP::Model>>);
}

BOOST_AUTO_TEST_CASE( negatives ) {
    namespace MDP = AIToolbox::MDP;
    using namespace AIToolbox::POMDP;

    static_assert(!IsGenerativeModel<MDP::Model>);
    static_assert(!IsGenerativeModel<int>);
    static_assert(!IsGenerativeModel<void*>);

    static_assert(!IsModel<MDP::Model>);
    static_assert(!IsModel<int>);
    static_assert(!IsModel<void*>);

    static_assert(!IsModelEigen<MDP::Model>);
    static_assert(!IsModelEigen<int>);
    static_assert(!IsModelEigen<void*>);
}
