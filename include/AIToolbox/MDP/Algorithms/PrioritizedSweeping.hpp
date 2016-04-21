#ifndef AI_TOOLBOX_MDP_PRIORITIZED_SWEEPING_HEADER_FILE
#define AI_TOOLBOX_MDP_PRIORITIZED_SWEEPING_HEADER_FILE

#include <AIToolbox/MDP/Algorithms/PrioritizedSweepingImpl/PrioritizedSweepingGeneral.hpp>
#include <AIToolbox/MDP/Algorithms/PrioritizedSweepingImpl/PrioritizedSweepingEigen.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class represents the PrioritizedSweeping algorithm.
         *
         * This class is simply a placeholder which selects automatically
         * the best implementation depending on your model. If your model
         * supports Eigen, then the Eigen version is selected, which performs
         * several times better than the most general one.
         *
         * Both implementation have exactly the same API and are used
         * in the exact same way.
         *
         * This class in itself cannot be instantiated.
         *
         * Check each of the implementations' documentations to know more.
         */
        template <typename T, typename U = void>
        class PrioritizedSweeping {
            // We prevent this class from being instantiated, if the model does not support either implementation.
            static_assert(!std::is_same<U,U>::value, "Type T is not valid for this template");
        };

#ifndef DOXYGEN_SKIP
        template <typename T>
        class PrioritizedSweeping<T, typename std::enable_if<is_model<T>::value && !is_model_eigen<T>::value>::type> : public PrioritizedSweepingGeneral<T> {
            using PrioritizedSweepingGeneral<T>::PrioritizedSweepingGeneral;
        };

        template <typename T>
        class PrioritizedSweeping<T, typename std::enable_if<is_model_eigen<T>::value>::type> : public PrioritizedSweepingEigen<T> {
            using PrioritizedSweepingEigen<T>::PrioritizedSweepingEigen;
        };
#endif
    }
}
#endif
