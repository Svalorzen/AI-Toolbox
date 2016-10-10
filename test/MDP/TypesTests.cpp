#define BOOST_TEST_MODULE MDP_Types
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include "Utils/OldMDPModel.hpp"
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Experience.hpp>

BOOST_AUTO_TEST_CASE( positives ) {
    using namespace AIToolbox;
    using namespace AIToolbox::MDP;

    BOOST_CHECK(is_generative_model<Model>::value);
    BOOST_CHECK(is_model<Model>::value);
    BOOST_CHECK(is_model_eigen<Model>::value);
    BOOST_CHECK(!is_model_not_eigen<Model>::value);

    BOOST_CHECK(is_generative_model<OldMDPModel>::value);
    BOOST_CHECK(is_model<OldMDPModel>::value);
    BOOST_CHECK(!is_model_eigen<OldMDPModel>::value);
    BOOST_CHECK(is_model_not_eigen<OldMDPModel>::value);

    BOOST_CHECK(is_generative_model<POMDP::Model<Model>>::value);
    BOOST_CHECK(is_model<POMDP::Model<Model>>::value);
    BOOST_CHECK(is_model_eigen<POMDP::Model<Model>>::value);
    BOOST_CHECK(!is_model_not_eigen<POMDP::Model<Model>>::value);

    BOOST_CHECK(is_generative_model<POMDP::Model<OldMDPModel>>::value);
    BOOST_CHECK(is_model<POMDP::Model<OldMDPModel>>::value);
    BOOST_CHECK(!is_model_eigen<POMDP::Model<OldMDPModel>>::value);
    BOOST_CHECK(is_model_not_eigen<POMDP::Model<OldMDPModel>>::value);

    BOOST_CHECK(is_experience<Experience>::value);
}

BOOST_AUTO_TEST_CASE( negatives ) {
    using namespace AIToolbox;
    using namespace AIToolbox::MDP;

    BOOST_CHECK(!is_generative_model<Experience>::value);
    BOOST_CHECK(!is_generative_model<int>::value);
    BOOST_CHECK(!is_generative_model<void*>::value);

    BOOST_CHECK(!is_model<Experience>::value);
    BOOST_CHECK(!is_model<int>::value);
    BOOST_CHECK(!is_model<void*>::value);

    BOOST_CHECK(!is_model_eigen<Experience>::value);
    BOOST_CHECK(!is_model_eigen<int>::value);
    BOOST_CHECK(!is_model_eigen<void*>::value);

    BOOST_CHECK(!is_model_not_eigen<Experience>::value);
    BOOST_CHECK(!is_model_not_eigen<int>::value);
    BOOST_CHECK(!is_model_not_eigen<void*>::value);

    BOOST_CHECK(!is_experience<Model>::value);
    BOOST_CHECK(!is_experience<int>::value);
    BOOST_CHECK(!is_experience<void*>::value);
}
