#include <AIToolbox/Factored/MDP/CooperativeThompsonModel.hpp>

#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeThompsonModel::CooperativeThompsonModel(const CooperativeExperience & exp, const double discount)
            : experience_(exp), discount_(discount), transitions_({experience_.getGraph(), {}})
    {
        const auto & S = experience_.getS();
        auto & tProbs = transitions_.transitions;

        tProbs.reserve(S.size());
        rewards_.reserve(S.size());

        for (size_t i = 0; i < S.size(); ++i) {
            const auto d1 = experience_.getGraph().getSize(i);
            const auto d2 = S[i];

            tProbs.emplace_back(d1, d2);
            rewards_.emplace_back(d1);
        }
        sync();
    }

    void CooperativeThompsonModel::sync() {
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            for (size_t j = 0; j < getGraph().getSize(i); ++j) {
                syncRow(i, j);
            }
        }
    }

    void CooperativeThompsonModel::sync(const State & s, const Action & a) {
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            syncRow(i, j);
        }
    }

    void CooperativeThompsonModel::sync(const CooperativeExperience::Indeces & indeces) {
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = indeces[i];

            syncRow(i, j);
        }
    }

    void CooperativeThompsonModel::syncRow(const size_t i, const size_t j) {
        const auto & S = experience_.getS();
        const auto & vtable  = experience_.getVisitsTable();
        const auto & rmatrix = experience_.getRewardMatrix();
        const auto & m2matrix = experience_.getM2Matrix();
        auto & tProbs = transitions_.transitions;

        sampleDirichletDistribution(
            // Here we add the Jeffreys prior
            //
            // Ideally this shouldn't allocate, as the casting and sum
            // should simply create a wrapper Eigen object which is passed
            // by reference, so should be still as efficient as doing it by
            // hand.
            vtable[i].row(j).head(S[i]).array().cast<double>() + 0.5,
            rand_, tProbs[i].row(j)
        );

        const auto totalVisits = vtable[i](j, S[i]);
        if (totalVisits < 2) {
            rewards_[i][j] = rmatrix[i][j];
        } else {
            std::student_t_distribution<double> dist(totalVisits - 1);
            rewards_[i][j] = rmatrix[i][j] + dist(rand_) * std::sqrt(m2matrix[i][j] / (totalVisits * (totalVisits - 1)));
        }
    }

    std::tuple<State, double> CooperativeThompsonModel::sampleSR(const State & s, const Action & a) const {
        State s1(s.size());
        const double reward = sampleSR(s, a, &s1);

        return std::make_tuple(s1, reward);
    }

    std::tuple<State, Rewards> CooperativeThompsonModel::sampleSRs(const State & s, const Action & a) const {
        const auto & S = experience_.getS();

        State s1(S.size());
        Rewards rs(S.size());

        sampleSRs(s, a, &s1, &rs);

        return std::make_tuple(s1, rs);
    }

    double CooperativeThompsonModel::sampleSR(const State & s, const Action & a, State * s1p) const {
        assert(s1p);

        const auto & S = experience_.getS();
        auto & tProbs = transitions_.transitions;

        State & s1 = *s1p;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            s1[i] = sampleProbability(S[i], tProbs[i].row(j), rand_);
        }

        return getExpectedReward(s, a, s1);
    }

    void CooperativeThompsonModel::sampleSRs(const State & s, const Action & a, State * s1p, Rewards * rews) const {
        assert(s1p);
        assert(rews);

        const auto & S = experience_.getS();
        auto & tProbs = transitions_.transitions;

        State & s1 = *s1p;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            s1[i] = sampleProbability(S[i], tProbs[i].row(j), rand_);
        }

        getExpectedRewards(s, a, s1, rews);
    }

    double CooperativeThompsonModel::getTransitionProbability(const State & s, const Action & a, const State & s1) const {
        return transitions_.getTransitionProbability(s, a, s1);
    }

    double CooperativeThompsonModel::getExpectedReward(const State & s, const Action & a, const State &) const {
        const auto & S = experience_.getS();

        double retval = 0.0;
        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            retval += rewards_[i][j];
        }

        return retval;
    }

    Rewards CooperativeThompsonModel::getExpectedRewards(const State & s, const Action & a, const State & s1) const {
        const auto & S = experience_.getS();

        Rewards rews(S.size());

        getExpectedRewards(s, a, s1, &rews);

        return rews;
    }

    void CooperativeThompsonModel::getExpectedRewards(const State & s, const Action & a, const State &, Rewards * rewsp) const {
        assert(rewsp);

        const auto & S = experience_.getS();

        auto & rews = *rewsp;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            rews[i] = rewards_[i][j];
        }
    }

    void CooperativeThompsonModel::setDiscount(const double d) { discount_ = d; }
    double CooperativeThompsonModel::getDiscount() const { return discount_; }

    const State & CooperativeThompsonModel::getS() const { return experience_.getS(); }
    const Action & CooperativeThompsonModel::getA() const { return experience_.getA(); }
    const CooperativeExperience & CooperativeThompsonModel::getExperience() const { return experience_; }
    const CooperativeThompsonModel::TransitionMatrix & CooperativeThompsonModel::getTransitionFunction() const { return transitions_; }
    const CooperativeThompsonModel::RewardMatrix & CooperativeThompsonModel::getRewardFunction() const { return rewards_; }
    const DDNGraph & CooperativeThompsonModel::getGraph() const { return experience_.getGraph(); }
}
