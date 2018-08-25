#define BOOST_TEST_MODULE MDP_Types
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include "Utils/OldMDPModel.hpp"
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Experience.hpp>

BOOST_AUTO_TEST_CASE( positives ) {
    using namespace AIToolbox;
    using namespace AIToolbox::MDP;

    BOOST_CHECK(is_generative_model_v<Model>);
    BOOST_CHECK(is_model_v<Model>);
    BOOST_CHECK(is_model_eigen_v<Model>);
    BOOST_CHECK(!is_model_not_eigen_v<Model>);

    BOOST_CHECK(is_generative_model_v<OldMDPModel>);
    BOOST_CHECK(is_model_v<OldMDPModel>);
    BOOST_CHECK(!is_model_eigen_v<OldMDPModel>);
    BOOST_CHECK(is_model_not_eigen_v<OldMDPModel>);

    BOOST_CHECK(is_generative_model_v<POMDP::Model<Model>>);
    BOOST_CHECK(is_model_v<POMDP::Model<Model>>);
    BOOST_CHECK(is_model_eigen_v<POMDP::Model<Model>>);
    BOOST_CHECK(!is_model_not_eigen_v<POMDP::Model<Model>>);

    BOOST_CHECK(is_generative_model_v<POMDP::Model<OldMDPModel>>);
    BOOST_CHECK(is_model_v<POMDP::Model<OldMDPModel>>);
    BOOST_CHECK(!is_model_eigen_v<POMDP::Model<OldMDPModel>>);
    BOOST_CHECK(is_model_not_eigen_v<POMDP::Model<OldMDPModel>>);

    BOOST_CHECK(is_experience_v<Experience>);
}

BOOST_AUTO_TEST_CASE( negatives ) {
    using namespace AIToolbox;
    using namespace AIToolbox::MDP;

    BOOST_CHECK(!is_generative_model_v<Experience>);
    BOOST_CHECK(!is_generative_model_v<int>);
    BOOST_CHECK(!is_generative_model_v<void*>);

    BOOST_CHECK(!is_model_v<Experience>);
    BOOST_CHECK(!is_model_v<int>);
    BOOST_CHECK(!is_model_v<void*>);

    BOOST_CHECK(!is_model_eigen_v<Experience>);
    BOOST_CHECK(!is_model_eigen_v<int>);
    BOOST_CHECK(!is_model_eigen_v<void*>);

    BOOST_CHECK(!is_model_not_eigen_v<Experience>);
    BOOST_CHECK(!is_model_not_eigen_v<int>);
    BOOST_CHECK(!is_model_not_eigen_v<void*>);

    BOOST_CHECK(!is_experience_v<Model>);
    BOOST_CHECK(!is_experience_v<int>);
    BOOST_CHECK(!is_experience_v<void*>);
}
