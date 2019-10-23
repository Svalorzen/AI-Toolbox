#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeModel::CooperativeModel(DDNGraph graph, FactoredDDN::TransitionMatrix transitions, FactoredMatrix2D rewards, const double discount) :
            discount_(discount),
            graph_(std::move(graph)),
            transitions_({graph_, std::move(transitions)}), rewards_(std::move(rewards)),
            rand_(Impl::Seeder::getSeed())
    {
        // Now we validate both the transition function and the rewards.
        // The DDN graph we can already trust since it's a class and not a
        // struct, but we can still check that all the nodes have been pushed.
        const auto & S = graph_.getS();
        const auto & A = graph_.getA();
        if (S.size() == 0)
            throw std::invalid_argument("Input DDN has empty state space in its DDNGraph");
        if (A.size() == 0)
            throw std::invalid_argument("Input DDN has empty action space in its DDNGraph");
        if (graph_.getNodes().size() != S.size())
            throw std::invalid_argument("Input DDN has an incorrect number of nodes in its DDNGraph!");

        if (transitions_.transitions.size() != S.size()) {
            throw std::invalid_argument("Input DDN has an incorrect number of transition nodes!");
        }

        for (size_t i = 0; i < S.size(); ++i) {
            if (static_cast<size_t>(transitions_.transitions[i].rows()) != graph_.getSize(i)) {
                throw std::invalid_argument("Input DDN contains matrices with incorrect number of rows!");
            }
            if (static_cast<size_t>(transitions_.transitions[i].cols()) != graph_.getS()[i]) {
                throw std::invalid_argument("Input DDN contains matrices with incorrect number of columns!");
            }

            // Check each row is a probability.
            for (size_t j = 0; j < graph_.getSize(i); ++j)
                if (!isProbability(S[i], transitions_.transitions[i].row(j)))
                    throw std::invalid_argument("Input DDN contains invalid probabilities!");
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
        const auto & S = graph_.getS();

        State s1(S.size());
        const double reward = sampleSR(s, a, &s1);

        return std::make_tuple(s1, reward);
    }

    double CooperativeModel::sampleSR(const State & s, const Action & a, State * s1p) const {
        const auto & tProbs = transitions_.transitions;
        const auto & S = graph_.getS();

        State & s1 = *s1p;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = graph_.getId(s, a, i);

            s1[i] = sampleProbability(S[i], tProbs[i].row(j), rand_);
        }

        return rewards_.getValue(S, graph_.getA(), s, a);
    }

    std::tuple<State, Rewards> CooperativeModel::sampleSRs(const State & s, const Action & a) const {
        const auto & tProbs = transitions_.transitions;
        const auto & S = graph_.getS();

        std::tuple<State, Rewards> retval;
        auto & [s1, rews] = retval;

        s1.resize(S.size());
        rews.resize(rewards_.bases.size());

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = graph_.getId(s, a, i);

            s1[i] = sampleProbability(S[i], tProbs[i].row(j), rand_);
        }

        for (size_t i = 0; i < rewards_.bases.size(); ++i) {
            const auto & e = rewards_.bases[i];
            const auto fid = toIndexPartial(e.tag, S, s);
            const auto aid = toIndexPartial(e.actionTag, graph_.getA(), a);

            rews[i] = e.values(fid, aid);
        }

        return retval;
    }

    double CooperativeModel::getTransitionProbability(const State & s, const Action & a, const State & s1) const {
        return transitions_.getTransitionProbability(s, a, s1);
    }

    double CooperativeModel::getExpectedReward(const State & s, const Action & a, const State &) const {
        return rewards_.getValue(graph_.getS(), graph_.getA(), s, a);
    }

    const State & CooperativeModel::getS() const { return graph_.getS(); }
    const Action & CooperativeModel::getA() const { return graph_.getA(); }
    double CooperativeModel::getDiscount() const { return discount_; }
    const FactoredDDN & CooperativeModel::getTransitionFunction() const { return transitions_; }
    const FactoredMatrix2D & CooperativeModel::getRewardFunction() const { return rewards_; }
    const DDNGraph & CooperativeModel::getGraph() const { return graph_; }
}

