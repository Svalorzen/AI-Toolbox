#ifndef AI_TOOLBOX_FACTORED_MDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_UTILS_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This function applies a one-step backup on the input ValueFunction.
     *
     * @param m The model used to do the backup.
     * @param v The ValueFunction to backup.
     *
     * @return The QFunction resulting from the backup.
     */
    QFunction bellmanBackup(const CooperativeModel & m, const ValueFunction & v);
}

#endif
