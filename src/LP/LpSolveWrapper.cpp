#include <AIToolbox/LP.hpp>

#include <type_traits>

#include <lpsolve/lp_lib.h>

namespace AIToolbox {
    constexpr bool conversionNeeded = !std::is_same<REAL, double>::value;

    template <bool realConversionNeeded>
    struct ConversionArray {
        ConversionArray(const size_t) {}
        void resize(size_t) {}
        double * conversionData(const Vector &) { return nullptr; }
    };

    template <>
    struct ConversionArray<true> {
        ConversionArray(const size_t vars) : conv_(new REAL[vars + 1]) {}
        REAL * conversionData(const Vector & row) {
            for (int v = 0; v < row.size(); ++v )
                conv_[v+1] = static_cast<REAL>(row[v]);
            return conv_.get();
        }
        void resize(const size_t vars) {
            conv_.reset(new REAL[vars + 1]);
        }
        std::unique_ptr<REAL[]> conv_;
    };

    struct LP::LP_impl : private ConversionArray<conversionNeeded> {
        LP_impl(size_t vars);
        void resize(size_t vars);
        // Used to switch from double to REAL if needed.
        using ConversionArray<conversionNeeded>::conversionData;

        std::unique_ptr<lprec, void(*)(lprec*)> lp_;
        std::unique_ptr<double[]> data_;
    };

    LP::LP_impl::LP_impl(const size_t vars) :
            ConversionArray(vars), lp_(make_lp(0, vars), delete_lp),
            data_(new double[vars + 1])
    {
        // Make lp shut up. Could redirect stream to /dev/null if needed.
        set_verbose(lp_.get(), SEVERE /*or CRITICAL*/);

        set_simplextype(lp_.get(), SIMPLEX_DUAL_DUAL);

        // This makes adding row constraints faster, but then we'd have to turn
        // it off before solving.. and can never turn it on again..
        // set_add_rowmode(lp.get(), true);

        // Not included in Debian package, speeds around 3x, but also crashes
        // set_BFP(lp.get(), "../../libbfp_etaPFI.so");
    }

    void LP::LP_impl::resize(const size_t vars) {
        data_.reset(new double[vars + 1]);
        ConversionArray<conversionNeeded>::resize(vars);
    }

    constexpr int toLpSolveConstraint(LP::Constraint c) {
        if (c == LP::Constraint::LessEqual)
            return LE;
        if (c == LP::Constraint::GreaterEqual)
            return GE;
        return EQ;
    }

    LP::~LP() = default;

    // Row is initialized from 1 since lp_solve reads element from 1 onwards
    LP::LP(const size_t varNumber) :
            pimpl_(new LP_impl(varNumber)), row(pimpl_->data_.get()+1, varNumber),
            varNumber_(varNumber), maximize_(false) {}

    void LP::setObjective(const size_t n, const bool maximize) {
        set_obj(pimpl_->lp_.get(), n+1, 1.0);
        if (maximize)
            set_maxim(pimpl_->lp_.get());
        else
            set_minim(pimpl_->lp_.get());
        maximize_ = maximize;
    }

    void LP::pushRow(const Constraint c, const double value) {
        if constexpr (conversionNeeded) {
            add_constraint(pimpl_->lp_.get(), pimpl_->conversionData(row), toLpSolveConstraint(c), static_cast<REAL>(value));
        } else {
            add_constraint(pimpl_->lp_.get(), pimpl_->data_.get(), toLpSolveConstraint(c), static_cast<REAL>(value));
        }
    }

    // TODO: Implement this version of pushRow to improve performances.
    // void LP::pushRow(const std::vector<int> & ids, const Constraint c, const double value) {
    //     add_constraintex(pimpl_->lp_.get(), ids.size(), pimpl_->data_.get(), ids.data(), toLpSolveConstraint(c), static_cast<REAL>(value));
    // }

    void LP::popRow() {
        del_constraint(pimpl_->lp_.get(), get_Nrows(pimpl_->lp_.get()));
    }

    size_t LP::addColumn() {
        ++varNumber_;
        // Add element to row
        pimpl_->resize(varNumber_);
        // Reassign MAP to new row
        new (&row) Eigen::Map<Vector>(pimpl_->data_.get()+1, varNumber_);
        // Add new empty column to LP
        add_columnex(pimpl_->lp_.get(), 0, NULL, NULL);

        return varNumber_;
    }

    void LP::setUnbounded(const size_t n) {
        set_unbounded(pimpl_->lp_.get(), n+1);
    }

    std::optional<Vector> LP::solve(const size_t variables) {
        auto lp = pimpl_->lp_.get();
        // lp_solve uses the result of the previous runs to bootstrap
        // the new solution. Sometimes this breaks down for some reason,
        // so we just avoid it - it does not really even give a performance
        // boost..
        default_basis(lp);


        // print_lp(lp.get());
        const auto result = ::solve(lp);

        REAL * vp;
        get_ptr_variables(lp, &vp);
        const REAL value = get_objective(lp);

        // We have found a witness point if we have found a belief for which the value
        // of the supplied ValueFunction is greater than ALL others. Thus we just need
        // to verify that the variable we have maximixed is actually greater than 0.
        const bool isSolved = !( result > 1 || (maximize_ && value <= 0.0) || (!maximize_ && value >= 0.0) );


        std::optional<Vector> solution;


        if ( isSolved )
            solution = Eigen::Map<Vector>(vp, variables);


        return solution;
    }

    void LP::resize(const size_t rows) {
        resize_lp(pimpl_->lp_.get(), rows, row.size());
    }
}
