#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeModel::CooperativeModel(State s, Action a, FactoredDDN transitions, FactoredMatrix2D rewards, const double discount) :
            S(std::move(s)), A(std::move(a)), discount_(discount),
            transitions_(std::move(transitions)), rewards_(std::move(rewards)),
            rand_(Impl::Seeder::getSeed())
    {
        // Now we validate both the transition function and the rewards.
        if (transitions_.nodes.size() != S.size())
            throw std::invalid_argument("Input DDN has an incorrect number of nodes!");

        TagErrors error;
        for (size_t s = 0; s < S.size(); ++s) {
            const auto & node = transitions_.nodes[s];

            std::tie(error, std::ignore) = checkTag(A, node.actionTag);
            switch (error) {
                case TagErrors::NoElements:
                    throw std::invalid_argument("Input DDN contains action tags with no elements!");
                case TagErrors::TooManyElements:
                    throw std::invalid_argument("Input DDN contains action tags with too many elements!");
                case TagErrors::IdTooHigh:
                    throw std::invalid_argument("Input DDN references action IDs too high for the action space!");
                case TagErrors::NotSorted:
                    throw std::invalid_argument("Input DDN contains action tags that are not sorted!");
                case TagErrors::Duplicates:
                    throw std::invalid_argument("Input DDN contains duplicate action tags entries!");
                default:;
            }

            // Check number of nodes is correct.
            if (node.nodes.size() != factorSpacePartial(node.actionTag, A))
                throw std::invalid_argument("Input DDN has an incorrect number of bayesian nodes for the specified action tag!");

            for (size_t a = 0; a < node.nodes.size(); ++a) {
                const auto & subnode = node.nodes[a];

                std::tie(error, std::ignore) = checkTag(S, subnode.tag);
                switch (error) {
                    case TagErrors::NoElements:
                        throw std::invalid_argument("Input DDN contains subnode tags with no elements!");
                    case TagErrors::TooManyElements:
                        throw std::invalid_argument("Input DDN contains subnode tags with too many elements!");
                    case TagErrors::IdTooHigh:
                        throw std::invalid_argument("Input DDN references state IDs too high for the state space!");
                    case TagErrors::NotSorted:
                        throw std::invalid_argument("Input DDN contains state tags that are not sorted!");
                    case TagErrors::Duplicates:
                        throw std::invalid_argument("Input DDN contains duplicate state tags entries!");
                    default:;
                }

                // Check size of matrix is correct
                if (subnode.matrix.cols() != static_cast<long>(S[s]))
                    throw std::invalid_argument("Input DDN contains matrices with incorrect number of columns!");

                if (subnode.matrix.rows() != static_cast<long>(factorSpacePartial(subnode.tag, S)))
                    throw std::invalid_argument("Input DDN contains matrices with incorrect number of rows!");

                // Check each row is a probability.
                for (int i = 0; i < subnode.matrix.rows(); ++i)
                    if (!isProbability(S[s], subnode.matrix.row(i)))
                        throw std::invalid_argument("Input DDN contains invalid probabilities!");
            }
        }

        for (const auto & r : rewards_.bases) {
            auto [error, id] = checkTag(A, r.actionTag);
            switch (error) {
                case TagErrors::NoElements:
                    throw std::invalid_argument("Input reward function contains action tags with no elements!");
                case TagErrors::TooManyElements:
                    throw std::invalid_argument("Input reward function contains action tags with too many elements!");
                case TagErrors::IdTooHigh:
                    throw std::invalid_argument("Input reward function references action IDs too high for the action space!");
                case TagErrors::NotSorted:
                    throw std::invalid_argument("Input reward function contains action tags that are not sorted!");
                case TagErrors::Duplicates:
                    throw std::invalid_argument("Input reward function contains duplicate action tags entries!");
                default:;
            }
            std::tie(error, id) = checkTag(S, r.tag);
            switch (error) {
                case TagErrors::NoElements:
                    throw std::invalid_argument("Input reward function contains subnode tags with no elements!");
                case TagErrors::TooManyElements:
                    throw std::invalid_argument("Input reward function contains subnode tags with too many elements!");
                case TagErrors::IdTooHigh:
                    throw std::invalid_argument("Input reward function references state IDs too high for the state space!");
                case TagErrors::NotSorted:
                    throw std::invalid_argument("Input reward function contains state tags that are not sorted!");
                case TagErrors::Duplicates:
                    throw std::invalid_argument("Input reward function contains duplicate state tags entries!");
                default:;
            }

            // Check size of matrix is correct
            if (r.values.cols() != static_cast<long>(factorSpacePartial(r.actionTag, A)))
                throw std::invalid_argument("Input reward function contains bases with incorrect number of columns!");

            if (r.values.rows() != static_cast<long>(factorSpacePartial(r.tag, S)))
                throw std::invalid_argument("Input reward function contains bases with incorrect number of rows!");
        }
    }

    std::tuple<State, double> CooperativeModel::sampleSR(const State & s, const Action & a) const {
        State s1(S.size());
        const double reward = sampleSR(s, a, &s1);

        return std::make_tuple(s1, reward);
    }

    double CooperativeModel::sampleSR(const State & s, const Action & a, State * s1p) const {
        State & s1 = *s1p;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto actionId = toIndexPartial(transitions_[i].actionTag, A, a);

            const auto & node = transitions_[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, S, s);

            const size_t newS = sampleProbability(S[i], node.matrix.row(parentId), rand_);

            s1[i] = newS;
        }

        return rewards_.getValue(S, A, s, a);
    }

    std::tuple<State, Rewards> CooperativeModel::sampleSRs(const State & s, const Action & a) const {
        std::tuple<State, Rewards> retval;
        auto & [s1, rews] = retval;

        s1.resize(S.size());
        rews.resize(rewards_.bases.size());

        for (size_t i = 0; i < S.size(); ++i) {
            const auto actionId = toIndexPartial(transitions_[i].actionTag, A, a);

            const auto & node = transitions_[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, S, s);

            const size_t newS = sampleProbability(S[i], node.matrix.row(parentId), rand_);

            s1[i] = newS;
        }

        for (size_t i = 0; i < rewards_.bases.size(); ++i) {
            const auto & e = rewards_.bases[i];
            const auto fid = toIndexPartial(e.tag, S, s);
            const auto aid = toIndexPartial(e.actionTag, A, a);

            rews[i] = e.values(fid, aid);
        }

        return retval;
    }

    double CooperativeModel::getTransitionProbability(const State & s, const Action & a, const State & s1) const {
        return transitions_.getTransitionProbability(S, A, s, a, s1);
    }

    double CooperativeModel::getExpectedReward(const State & s, const Action & a, const State &) const {
        return rewards_.getValue(S, A, s, a);
    }

    const State & CooperativeModel::getS() const { return S; }
    const Action & CooperativeModel::getA() const { return A; }
    double CooperativeModel::getDiscount() const { return discount_; }
    const FactoredDDN & CooperativeModel::getTransitionFunction() const { return transitions_; }
    const FactoredMatrix2D & CooperativeModel::getRewardFunction() const { return rewards_; }
}

