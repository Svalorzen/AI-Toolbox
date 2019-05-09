#include <AIToolbox/Factored/MDP/Policies/EpsilonPolicy.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    EpsilonPolicy::EpsilonPolicy(const EpsilonBase::Base & p, double epsilon) :
            EpsilonBase::Base(p.getS(), p.getA()), EpsilonBase(p, epsilon)
    {
        randomDistribution_.reserve(getA().size());
        for (size_t i = 0; i < getA().size(); ++i)
            randomDistribution_.emplace_back(0, getA()[i] - 1);
    }

    Action EpsilonPolicy::sampleRandomAction() const {
        Action a;
        a.reserve(getA().size());

        for (size_t i = 0; i < getA().size(); ++i)
            a.push_back(randomDistribution_[i](rand_));

        return a;
    }

    double EpsilonPolicy::getRandomActionProbability() const {
        return 1.0 / factorSpace(A);
    }
}
