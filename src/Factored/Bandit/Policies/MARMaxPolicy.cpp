#include <AIToolbox/Factored/Bandit/Policies/MARMaxPolicy.hpp>

#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    MARMaxPolicy::MARMaxPolicy(const Experience & exp, Vector ranges, double epsilon, double delta, bool optimistic) :
            Base(exp.getA()), exp_(exp), ranges_(std::move(ranges)),
            epsilon_(epsilon), delta_(delta), optimistic_(optimistic),
            values_(exp_.getRewardMatrix()), graph_(getA().size()),
            canRecommend_(false),
            currentAction_(exp.getA().size(), 0)
    {
        assert(exp_.getDependencies().size() == static_cast<size_t>(ranges_.size()));

        if (epsilon < 0.0 || epsilon_ > 1.0) throw std::invalid_argument("Epsilon parameter must be in [0,1]");
        if (delta_ <= 0.0 || delta_ > 1.0) throw std::invalid_argument("Delta parameter must be in (0,1]");

        // m >= ln(2/delta) * sum(range^2) / 2 * eps^2 * sum(range)^2
        {
            double sumR = 0.0, sumRSquared = 0.0;
            for (auto r : ranges_) {
                sumR += r;
                sumRSquared += r * r;
            }
            m_ = std::max(
                1.0,
                std::ceil( (std::log( 2.0 / delta_ ) * sumRSquared) / (2.0 * epsilon_ * epsilon_ * sumR * sumR ) )
            );
        }

        for (size_t i = 0; i < static_cast<size_t>(ranges_.size()); ++i) {
            // Initialize values
            values_.bases[i].values.fill(ranges_[i]);

            // Initialize graph
            auto it = graph_.getFactor(exp_.getDependencies()[i]);
            const size_t size = exp_.getRewardMatrix().bases[i].values.size();

            if (it->getData().size() == 0) {
                it->getData().reserve(size);
                for (size_t j = 0; j < size; ++j)
                    it->getData().emplace_back(j, VariableElimination::Factor{ranges_[i], {}});
            } else {
                for (size_t j = 0; j < size; ++j)
                    it->getData()[j].second.first += ranges_[i];
            }
        }
    }

    double MARMaxPolicy::getActionProbability(const Action & a) const {
        return a == currentAction_;
    }

    Action MARMaxPolicy::sampleAction() const {
        return currentAction_;
    }

    bool MARMaxPolicy::canRecommendAction() const {
        return canRecommend_;
    }

    Action MARMaxPolicy::recommendAction() const {
        return currentAction_;
    }

    void MARMaxPolicy::stepUpdateQ(const Experience::Indeces & indeces) {
        // Update graph from experience to see which means we're allowed to
        // copy over to our graph.
        for (size_t i = 0; i < indeces.size(); ++i) {
            // For each local reward factor, figure out the factor in the graph
            // we need to update.
            auto fIt = graph_.getFactor(exp_.getDependencies()[i]);

            const auto id = indeces[i];

            const auto n = exp_.getVisitsTable()[i][id];
            const auto q = exp_.getRewardMatrix().bases[i].values[id];

            // Value stored for this exact local reward function.
            auto & value = values_.bases[i].values[id];
            // Value stored together with other LRF that depend on the same agents.
            auto & gValue = fIt->getData()[id].second.first;

            // Here we first compute the update for "value" since it's the
            // simplest one (just the individual value).
            // From this, we compute a diff that we apply to the gValue, so
            // that its cumulative sum over possibly multiple local reward
            // functions stays consistent.
            //
            // Would have been nice to only need and update gValue but
            // unfortunately it can't be done given the way that the
            // FactorGraph class works.
            if (n >= m_) {
                const double diff = q - value;

                value = q;
                gValue += diff;
            } else if (optimistic_) {
                // This is the MAVMax variant
                const double newValue = (n * q + (m_ - n) * ranges_[i]) / m_;
                const double diff = newValue - value;

                value = newValue;
                gValue += diff;
            }
        }

        // Compute here action to select/recommend next timestep.
        {
            auto g2 = graph_;

            VariableElimination ve;
            double unusedV;
            std::tie(currentAction_, unusedV) = ve(exp_.getA(), g2);
        }

        // Check whether the currently selected action can be recommended.
        canRecommend_ = true;
        for (size_t i = 0; i < static_cast<size_t>(ranges_.size()); ++i) {
            const auto & group = exp_.getDependencies()[i];

            const auto id = toIndexPartial(group, exp_.getA(), currentAction_);

            const auto & v = exp_.getVisitsTable()[i];
            if (v[id] < m_) {
                canRecommend_ = false;
                break;
            }
        }
    }

    double MARMaxPolicy::getEpsilon() const { return epsilon_; }
    double MARMaxPolicy::getDelta() const { return delta_; }
    unsigned MARMaxPolicy::getM() const { return m_; }
    const Experience & MARMaxPolicy::getExperience() const { return exp_; }
}
