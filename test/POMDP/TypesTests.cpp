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
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    BOOST_CHECK(is_generative_model_v<Model<MDP::Model>>);
    BOOST_CHECK(is_model_v<Model<MDP::Model>>);
    BOOST_CHECK(is_model_eigen_v<Model<MDP::Model>>);
    BOOST_CHECK(!is_model_not_eigen_v<Model<MDP::Model>>);

    BOOST_CHECK(is_generative_model_v<OldPOMDPModel<MDP::Model>>);
    BOOST_CHECK(is_model_v<OldPOMDPModel<MDP::Model>>);
    BOOST_CHECK(!is_model_eigen_v<OldPOMDPModel<MDP::Model>>);
    BOOST_CHECK(is_model_not_eigen_v<OldPOMDPModel<MDP::Model>>);
}

BOOST_AUTO_TEST_CASE( negatives ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    BOOST_CHECK(!is_generative_model_v<MDP::Model>);
    BOOST_CHECK(!is_generative_model_v<int>);
    BOOST_CHECK(!is_generative_model_v<void*>);

    BOOST_CHECK(!is_model_v<MDP::Model>);
    BOOST_CHECK(!is_model_v<int>);
    BOOST_CHECK(!is_model_v<void*>);

    BOOST_CHECK(!is_model_eigen_v<MDP::Model>);
    BOOST_CHECK(!is_model_eigen_v<int>);
    BOOST_CHECK(!is_model_eigen_v<void*>);

    BOOST_CHECK(!is_model_not_eigen_v<MDP::Model>);
    BOOST_CHECK(!is_model_not_eigen_v<int>);
    BOOST_CHECK(!is_model_not_eigen_v<void*>);
}
