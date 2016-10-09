#ifndef AI_TOOLBOX_POMDP_PROJECTER_HEADER_FILE
#define AI_TOOLBOX_POMDP_PROJECTER_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/ProjecterImpl/ProjecterGeneral.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/ProjecterImpl/ProjecterEigen.hpp>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This class offers projecting facilities for Models.
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
        class Projecter {
            // We prevent this class from being instantiated, if the model does not support either implementation.
            static_assert(!std::is_same<U,U>::value, "Type T is not valid for this template");
        };

#ifndef DOXYGEN_SKIP
        template <typename T>
        class Projecter<T, typename std::enable_if<is_model_not_eigen<T>::value>::type> : public ProjecterGeneral<T> {
            using ProjecterGeneral<T>::ProjecterGeneral;
        };

        template <typename T>
        class Projecter<T, typename std::enable_if<is_model_eigen<T>::value>::type> : public ProjecterEigen<T> {
            using ProjecterEigen<T>::ProjecterEigen;
        };
#endif
    }
}

#endif
