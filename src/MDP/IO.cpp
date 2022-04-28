#include <AIToolbox/MDP/IO.hpp>

#include <iostream>

#include <AIToolbox/Utils/IO.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/MDP/Policies/Policy.hpp>

#include <AIToolbox/Tools/CassandraParser.hpp>
#include <AIToolbox/Logging.hpp>

namespace AIToolbox::MDP {
    Model parseCassandra(std::istream & input) {
        CassandraParser parser;

        const auto & [S, A, T, R, discount] = parser.parseMDP(input);

        return Model(S, A, T, R, discount);
    }

    std::ostream & operator<<(std::ostream & os, const Experience & exp) {
        os << exp.getTimesteps() << '\n';
        write(os, exp.getVisitsTable());
        write(os, exp.getRewardMatrix());
        write(os, exp.getM2Matrix());

        return os;
    }

    std::ostream & operator<<(std::ostream & os, const SparseExperience & exp) {
        os << exp.getTimesteps() << '\n';
        write(os, exp.getVisitsTable());
        write(os, exp.getRewardMatrix());
        write(os, exp.getM2Matrix());

        return os;
    }

    std::ostream & operator<<(std::ostream & os, const Model & model) {
        write(os, model.getDiscount());
        write(os, model.getTransitionFunction());
        write(os, model.getRewardFunction());

        return os;
    }

    std::ostream & operator<<(std::ostream & os, const SparseModel & model) {
        write(os, model.getDiscount());
        write(os, model.getTransitionFunction());
        write(os, model.getRewardFunction());

        return os;
    }

    // Global discrete policy writer
    std::ostream& operator<<(std::ostream &os, const PolicyInterface & p) {
        write(os, p.getPolicy());

        return os;
    }

    // Experience reader
    std::istream& operator>>(std::istream &is, Experience & exp) {
        Experience e(exp.getS(), exp.getA());

        if (!(is >> e.timesteps_))
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Experience timesteps.");

        auto visits = e.getVisitsTable();
        if (!read(is, visits)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Experience visits table.");
            return is;
        } else
            e.setVisitsTable(visits);

        auto rewards = e.getRewardMatrix();
        if (!read(is, rewards)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Experience rewards matrix.");
            return is;
        } else
            e.setRewardMatrix(rewards);

        auto m2 = e.getM2Matrix();
        if (!read(is, m2)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Experience m2 matrix.");
            return is;
        } else
            e.setM2Matrix(m2);

        exp = std::move(e);
        return is;
    }

    // SparseExperience reader
    std::istream& operator>>(std::istream &is, SparseExperience & exp) {
        SparseExperience e(exp.getS(), exp.getA());

        if (!(is >> e.timesteps_))
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseExperience timesteps.");

        auto visits = e.getVisitsTable();
        if (!read(is, visits)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseExperience visits table.");
            return is;
        } else
            e.setVisitsTable(visits);

        auto rewards = e.getRewardMatrix();
        if (!read(is, rewards)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseExperience rewards matrix.");
            return is;
        } else
            e.setRewardMatrix(rewards);

        auto m2 = e.getM2Matrix();
        if (!read(is, m2)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseExperience m2 matrix.");
            return is;
        } else
            e.setM2Matrix(m2);

        exp = std::move(e);
        return is;
    }

    // MDP::Model reader
    std::istream& operator>>(std::istream &is, Model & m) {
        Model in(m.getS(), m.getA());

        double discount;
        if (!(is >> discount)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Model discount.");
            return is;
        } else
            in.setDiscount(discount);

        auto transitions = in.getTransitionFunction();
        if (!read(is, transitions)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Model transition function.");
            return is;
        } else {
            try {
                in.setTransitionFunction(transitions);
            } catch (const std::invalid_argument &) {
                AI_LOGGER(AI_SEVERITY_ERROR, "The transition function for Model did not contain valid probabilities.");
                is.setstate(std::ios::failbit);
                return is;
            }
        }

        auto rewards = in.getRewardFunction();
        if (!read(is, rewards)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Model reward function.");
            return is;
        } else
            in.setRewardFunction(rewards);

        m = std::move(in);
        return is;
    }

    // MDP::SparseModel reader
    std::istream& operator>>(std::istream &is, SparseModel & m) {
        SparseModel in(m.getS(), m.getA());

        double discount;
        if (!(is >> discount)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseModel discount.");
            return is;
        } else
            in.setDiscount(discount);

        auto transitions = in.getTransitionFunction();
        if (!read(is, transitions)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseModel transition function.");
            return is;
        } else {
            try {
                in.setTransitionFunction(transitions);
            } catch (const std::invalid_argument &) {
                AI_LOGGER(AI_SEVERITY_ERROR, "The transition function for SparseModel did not contain valid probabilities.");
                is.setstate(std::ios::failbit);
                return is;
            }
        }

        auto rewards = in.getRewardFunction();
        if (!read(is, rewards)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseModel reward function.");
            return is;
        } else
            in.setRewardFunction(rewards);

        m = std::move(in);
        return is;
    }

    // MDP::Policy reader
    std::istream& operator>>(std::istream &is, Policy &p) {
        Policy::PolicyMatrix pMatrix(p.getS(), p.getA());
        if (!read(is, pMatrix)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Policy matrix.");
            return is;
        }

        if (!isProbability(pMatrix)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Policy matrix does not contain valid probabilities.");
            is.setstate(std::ios::failbit);
            return is;
        }

        p.policy_ = std::move(pMatrix);
        return is;
    }
}
