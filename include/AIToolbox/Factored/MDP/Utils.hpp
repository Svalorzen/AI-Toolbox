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


    /**
     * @brief This function creates a new factored QFunction from the given graph and basis domain.
     *
     * The basisDomains is a set of state feature id groups. Each group should
     * be unique.
     *
     * For each such group, this function will identify the ids of the parent
     * features and agents as defined by the input graph, and a new basis
     * function is created inside the output QFunction. The parent
     * features/agents ids are then set as the tags of a new basis function.
     *
     * @param graph The DDNGraph containing the information about the factored MDP structure.
     * @param basisDomains The required domains for each of the outputted basis functions of the QFunction.
     *
     * @return A newly built QFunction.
     */
    QFunction makeQFunction(const DDNGraph & graph, const std::vector<std::vector<size_t>> & basisDomains);
}

#endif
