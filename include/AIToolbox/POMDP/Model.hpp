#ifndef AI_TOOLBOX_POMDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_POMDP_MODEL_HEADER_FILE

#include <AIToolbox/Utils.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<MDP::is_model<M>::value>::type>
        class Model;
#endif

        template <typename M>
        class Model<M> {
            public:
                using ObservationTable = Table3D;

                template <typename O>
                Model(const M & underlyingMDP, size_t o, const O & table);

                const M & getMDP() const;
            private:
                const M & mdp_;
                // void updateBelief(Belief * b);

                size_t O;
                ObservationTable observations_;
        };
    }
}

#endif
