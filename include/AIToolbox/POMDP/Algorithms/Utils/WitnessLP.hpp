#ifndef AI_TOOLBOX_POMDP_LP_HEADER_FILE
#define AI_TOOLBOX_POMDP_LP_HEADER_FILE

#include <cstddef>
#include <memory>
#include <vector>

#include <AIToolbox/POMDP/Types.hpp>

#include <lpsolve/lp_types.h>

namespace AIToolbox {
    namespace POMDP {
        class WitnessLP {
            public:
                WitnessLP(size_t S);

                void addRow(const std::vector<double> & v, int constrType);
                void popRow();

                void setDeltaCoefficient(REAL value);
                REAL getDeltaCoefficient() const;

                std::tuple<bool, POMDP::Belief> solve();

                void setRowNr(size_t rows);

            private:
                size_t S;
                int cols;
                std::unique_ptr<lprec, void(*)(lprec*)> lp;
                std::unique_ptr<REAL[]> row;
        };
    }
}

#endif
