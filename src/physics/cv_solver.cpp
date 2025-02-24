#include "cv.hpp"

namespace madrona::phys::cv {

namespace tasks {
void solveCPU(Context &ctx,
                     CVSolveData &cv_sing)
{
    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = getStateManager(ctx);

    // Call the solver
    if (cv_sing.cvxSolve && cv_sing.cvxSolve->fn) {
        cv_sing.cvxSolve->totalNumDofs = cv_sing.totalNumDofs;
        cv_sing.cvxSolve->numContactPts = cv_sing.numContactPts;
        cv_sing.cvxSolve->h = cv_sing.h;
        cv_sing.cvxSolve->mass = cv_sing.mass;
        cv_sing.cvxSolve->free_acc = cv_sing.freeAcc;
        cv_sing.cvxSolve->vel = cv_sing.vel;
        cv_sing.cvxSolve->J_c = cv_sing.J_c;
        cv_sing.cvxSolve->J_e = cv_sing.J_e;
        cv_sing.cvxSolve->numEqualityRows = cv_sing.numRowsJe;
        cv_sing.cvxSolve->mu = cv_sing.mu;
        cv_sing.cvxSolve->penetrations = cv_sing.penetrations;
        cv_sing.cvxSolve->eqResiduals = cv_sing.eqResiduals;
        cv_sing.cvxSolve->diagApprox_c = cv_sing.diagApprox_c;
        cv_sing.cvxSolve->diagApprox_e = cv_sing.diagApprox_e;

        cv_sing.cvxSolve->callSolve.store_release(1);
        while (cv_sing.cvxSolve->callSolve.load_acquire() != 2);
        cv_sing.cvxSolve->callSolve.store_relaxed(0);

        float *res = cv_sing.cvxSolve->resPtr;

        if (res) {
            BodyGroupMemory *all_memories = state_mgr->getWorldComponents<
                BodyGroupArchetype, BodyGroupMemory>(world_id);
            BodyGroupProperties *all_properties = state_mgr->getWorldComponents<
                BodyGroupArchetype, BodyGroupProperties>(world_id);

            CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

            // Update the body accelerations
            uint32_t processed_dofs = 0;
            for (CountT i = 0; i < num_grps; ++i) {
                BodyGroupMemory &m = all_memories[i];
                BodyGroupProperties &p = all_properties[i];

                for (CountT j = 0; j < all_properties[i].numBodies; j++) {
                    BodyOffsets offsets = m.offsets(p)[j];
                    float *dqv = m.dqv(p) + offsets.velOffset;

                    for (CountT k = 0; k < BodyOffsets::getDofTypeDim(offsets.dofType); k++) {
                        dqv[k] = res[processed_dofs];
                        processed_dofs++;
                    }
                }
            }
        }
    }
}
}

TaskGraphNodeID setupSolveTasks(TaskGraphBuilder &builder,
                                TaskGraphNodeID prev)
{
    auto cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::solveCPU,
            CVSolveData
        >>({prev});

    return cur_node;
}

}
