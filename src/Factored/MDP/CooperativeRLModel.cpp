#include <AIToolbox/Factored/MDP/CooperativeRLModel.hpp>

#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeRLModel::CooperativeRLModel(const CooperativeExperience & exp, double discount, bool)
            : experience_(exp), discount_(discount)
    {
        const auto & vnodes = experience_.getVisitTable().nodes;

        transitions_.nodes.resize(vnodes.size());
        rewards_.resize(vnodes.size());

        auto & tnodes = transitions_.nodes;

        for (size_t i = 0; i < vnodes.size(); ++i) {
            tnodes[i].actionTag = vnodes[i].actionTag;

            tnodes[i].nodes.resize(vnodes[i].nodes.size());
            rewards_[i].resize(vnodes[i].nodes.size());

            for (size_t j = 0; j < vnodes[i].nodes.size(); ++j) {
                tnodes[i].nodes[j].tag = vnodes[i].nodes[j].tag;

                tnodes[i].nodes[j].matrix.resize(
                    vnodes[i].nodes[j].matrix.rows(),
                    vnodes[i].nodes[j].matrix.cols() - 1 // vnodes also has the overall sum column
                );
                tnodes[i].nodes[j].matrix.setZero();
                tnodes[i].nodes[j].matrix.col(0).fill(1.0);

                rewards_[i][j].resize(vnodes[i].nodes[j].matrix.rows());
                rewards_[i][j].setZero();
            }
        }
    }

    void CooperativeRLModel::sync(const State & s, const Action & a) {
        const auto & vnodes = experience_.getVisitTable().nodes;
        const auto & rnodes = experience_.getRewardTable().nodes;

        auto & tnodes = transitions_.nodes;
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            const auto actionId = toIndexPartial(tnodes[i].actionTag, getA(), a);

            const auto & node = tnodes[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, getS(), s);

            const auto totalVisits = vnodes[i].nodes[actionId].matrix(parentId, S[i]+1);

            tnodes[i].nodes[actionId].matrix.row(parentId) =
                vnodes[i].nodes[actionId].matrix.row(parentId).head(S[i]) / totalVisits;

            rewards_[i][actionId][parentId] = rnodes[i].nodes[actionId].matrix(parentId, S[i]+1) / totalVisits;
        }
    }

    void CooperativeRLModel::sync(const CooperativeExperience::Indeces & indeces) {
        const auto & vnodes = experience_.getVisitTable().nodes;
        const auto & rnodes = experience_.getRewardTable().nodes;

        auto & tnodes = transitions_.nodes;
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            const auto [actionId, parentId] = indeces[i];

            const auto totalVisits = vnodes[i].nodes[actionId].matrix(parentId, S[i]+1);

            tnodes[i].nodes[actionId].matrix.row(parentId) =
                vnodes[i].nodes[actionId].matrix.row(parentId).head(S[i]) / totalVisits;

            rewards_[i][actionId][parentId] = rnodes[i].nodes[actionId].matrix(parentId, S[i]+1) / totalVisits;
        }
    }

    std::tuple<State, double> CooperativeRLModel::sampleSR(const State & s, const Action & a) const {
        State s1(s.size());
        const double reward = sampleSR(s, a, &s1);

        return std::make_tuple(s1, reward);
    }

    double CooperativeRLModel::sampleSR(const State & s, const Action & a, State * s1p) const {
        const auto & tnodes = transitions_.nodes;
        State & s1 = *s1p;

        for (size_t i = 0; i < s.size(); ++i) {
            const auto actionId = toIndexPartial(tnodes[i].actionTag, getA(), a);

            const auto & node = tnodes[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, getS(), s);

            const size_t newS = sampleProbability(getS()[i], node.matrix.row(parentId), rand_);

            s1[i] = newS;
        }

        return getExpectedReward(s, a, s1);
    }

    double CooperativeRLModel::getTransitionProbability(const State & s, const Action & a, const State & s1) const {
        return transitions_.getTransitionProbability(getS(), getA(), s, a, s1);
    }

    double CooperativeRLModel::getExpectedReward(const State & s, const Action & a, const State &) const {
        double retval = 0.0;

        for (size_t i = 0; i < transitions_.nodes.size(); ++i) {
            const auto actionId = toIndexPartial(transitions_[i].actionTag, getA(), a);

            const auto & node = transitions_[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, getS(), s);

            retval += rewards_[i][actionId][parentId];
        }

        return retval;
    }

    void CooperativeRLModel::setDiscount(double d) { discount_ = d; }
    double CooperativeRLModel::getDiscount() const { return discount_; }

    const State & CooperativeRLModel::getS() const { return experience_.getS(); }
    const Action & CooperativeRLModel::getA() const { return experience_.getA(); }
    const CooperativeExperience & CooperativeRLModel::getExperience() const { return experience_; }
}
