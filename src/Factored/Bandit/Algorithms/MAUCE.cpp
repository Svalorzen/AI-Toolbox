#include <AIToolbox/Factored/Bandit/Algorithms/MAUCE.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

#include <AIToolbox/Impl/Logging.hpp>

namespace AIToolbox::Factored::Bandit {
    MAUCE::MAUCE(Action aa, const std::vector<std::pair<double, std::vector<size_t>>> & rangesAndDependencies) :
            A(std::move(aa)), timestep_(0),
            averages_(A), logA_(0.0)
    {
        // Compute log(|A|) without needing to compute |A| which may be too
        // big. We'll use it later to obtain log(t |A|)
        for (const auto a : A)
            logA_ += std::log(a);

        // Build single rules for each dependency group.
        // This allows us to allocate the rules_ only once, and to just
        // update their values at each timestep.
        for (const auto & dependency : rangesAndDependencies) {
            PartialFactorsEnumerator enumerator(A, dependency.second);
            while (enumerator.isValid()) {
                const auto & pAction = *enumerator;

                averages_.emplace(pAction, Average{ 0.0, 0, dependency.first * dependency.first });
                rules_.emplace_back(pAction, UCVE::V());

                enumerator.advance();
            }
        }
    }

    Action MAUCE::stepUpdateQ(const Action & a, const Rewards & rew) {
        AI_LOGGER(AI_SEVERITY_INFO, "Updating averages...");

        // Update all averages with what we've learned this step.  Note: We
        // know that the factors are going to be in the correct order when
        // looping here since we are looping in the same order in which we
        // have inserted them into the averages_ container, and filter
        // returns sorted lists. So we can correctly match the rewards with
        // the agent groups!
        size_t i = 0;
        auto filtered = averages_.filter(a);
        for (auto & avg : filtered)
            avg.value += (rew[i++] - avg.value) / (++avg.count);

        // Build the vectors to pass to UCVE
        AI_LOGGER(AI_SEVERITY_INFO, "Building vectors...");
        for (size_t i = 0; i < averages_.size(); ++i) {
            const double count = averages_[i].count ? averages_[i].count : 0.00001;
            std::get<1>(rules_[i])[0] = averages_[i].value;
            std::get<1>(rules_[i])[1] = averages_[i].rangeSquared / count;
        }

        // Update the timestep, and finish computing log(t |A|) for this
        // timestep.
        ++timestep_;
        const auto logtA = logA_ + std::log(timestep_);

        // Create and run UCVE
        AI_LOGGER(AI_SEVERITY_INFO, "Now running UCVE...");
        UCVE ucve(A, logtA);
        auto a_v = ucve(rules_);
        AI_LOGGER(AI_SEVERITY_INFO, "Done.");

        // We convert the output (PartialAction) to a normal action.
        return toFactors(A.size(), std::get<0>(a_v));
    }

    FactoredContainer<QFunctionRule> MAUCE::getQFunctionRules() const {
        FactoredContainer<QFunctionRule>::ItemsContainer container;

        for (size_t i = 0; i < averages_.size(); ++i)
            container.emplace_back(std::get<0>(rules_[i]), averages_[i].value);

        return FactoredContainer<QFunctionRule>(averages_.getTrie(), std::move(container));
    }

    unsigned MAUCE::getTimestep() const { return timestep_; }
    void MAUCE::setTimestep(unsigned t) { timestep_ = t; }
}
