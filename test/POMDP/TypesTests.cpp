#define BOOST_TEST_MODULE POMDP_Types
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "Utils/OldPOMDPModel.hpp"
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

BOOST_AUTO_TEST_CASE( positives ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    BOOST_CHECK(is_generative_model<Model<MDP::Model>>::value);
    BOOST_CHECK(is_model<Model<MDP::Model>>::value);
    BOOST_CHECK(is_model_eigen<Model<MDP::Model>>::value);
    BOOST_CHECK(!is_model_not_eigen<Model<MDP::Model>>::value);

    BOOST_CHECK(is_generative_model<OldPOMDPModel<MDP::Model>>::value);
    BOOST_CHECK(is_model<OldPOMDPModel<MDP::Model>>::value);
    BOOST_CHECK(!is_model_eigen<OldPOMDPModel<MDP::Model>>::value);
    BOOST_CHECK(is_model_not_eigen<OldPOMDPModel<MDP::Model>>::value);
}

BOOST_AUTO_TEST_CASE( negatives ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    BOOST_CHECK(!is_generative_model<MDP::Model>::value);
    BOOST_CHECK(!is_generative_model<int>::value);
    BOOST_CHECK(!is_generative_model<void*>::value);

    BOOST_CHECK(!is_model<MDP::Model>::value);
    BOOST_CHECK(!is_model<int>::value);
    BOOST_CHECK(!is_model<void*>::value);

    BOOST_CHECK(!is_model_eigen<MDP::Model>::value);
    BOOST_CHECK(!is_model_eigen<int>::value);
    BOOST_CHECK(!is_model_eigen<void*>::value);

    BOOST_CHECK(!is_model_not_eigen<MDP::Model>::value);
    BOOST_CHECK(!is_model_not_eigen<int>::value);
    BOOST_CHECK(!is_model_not_eigen<void*>::value);
}

