#define BOOST_TEST_MODULE MDP_Types
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include "Utils/OldMDPModel.hpp"
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Experience.hpp>

BOOST_AUTO_TEST_CASE( positives ) {
    namespace POMDP = AIToolbox::POMDP;
    using namespace AIToolbox::MDP;

    static_assert(IsGenerativeModel<Model>);
    static_assert(IsModel<Model>);
    static_assert(IsModelEigen<Model>);

    static_assert(IsGenerativeModel<OldMDPModel>);
    static_assert(IsModel<OldMDPModel>);
    static_assert(!IsModelEigen<OldMDPModel>);

    static_assert(IsGenerativeModel<POMDP::Model<Model>>);
    static_assert(IsModel<POMDP::Model<Model>>);
    static_assert(IsModelEigen<POMDP::Model<Model>>);

    static_assert(IsGenerativeModel<POMDP::Model<OldMDPModel>>);
    static_assert(IsModel<POMDP::Model<OldMDPModel>>);
    static_assert(!IsModelEigen<POMDP::Model<OldMDPModel>>);

    static_assert(IsExperience<Experience>);
    static_assert(IsExperienceEigen<Experience>);
}

BOOST_AUTO_TEST_CASE( negatives ) {
    namespace AI = AIToolbox;
    using namespace AIToolbox::MDP;

    static_assert(!AI::IsGenerativeModel<Experience>);
    static_assert(!AI::IsGenerativeModel<int>);
    static_assert(!AI::IsGenerativeModel<void*>);

    static_assert(!IsGenerativeModel<Experience>);
    static_assert(!IsGenerativeModel<int>);
    static_assert(!IsGenerativeModel<void*>);

    static_assert(!IsModel<Experience>);
    static_assert(!IsModel<int>);
    static_assert(!IsModel<void*>);

    static_assert(!IsModelEigen<Experience>);
    static_assert(!IsModelEigen<int>);
    static_assert(!IsModelEigen<void*>);

    static_assert(!IsExperience<Model>);
    static_assert(!IsExperience<int>);
    static_assert(!IsExperience<void*>);

    static_assert(!IsExperienceEigen<Model>);
    static_assert(!IsExperienceEigen<int>);
    static_assert(!IsExperienceEigen<void*>);
}
