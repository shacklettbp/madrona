#include <madrona/state.hpp>
#include <madrona/physics.hpp>
#include <madrona/context.hpp>
#include <madrona/cvphysics.hpp>
#include <madrona/taskgraph.hpp>

#include "physics_impl.hpp"

using namespace madrona::math;
using namespace madrona::base;

namespace madrona::phys::cv {

struct Contact : Archetype<
    ContactConstraint,
    ContactTmpState
> {};

struct Joint : Archetype<
    JointConstraint
> {};

struct CVRigidBodyState : Bundle<
    CVPhysicalComponent
> {};

struct CVSingleton {
    // Just to have something to loop over for CPU solver
};

namespace tasks {

// We are going to need a different way of solving on GPU mode
#ifdef MADRONA_GPU_MODE
struct GaussMinimizationNode : NodeBase {
    GaussMinimizationNode(StateManager *state_mgr);

    void solve(int32_t invocation_idx);

    static TaskGraph::NodeID addToGraph(
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);

    DofObjectPosition *positions;
    DofObjectVelocity *velocities;
    DofObjectNumDofs *numDofs;
    ObjectID *objIDs;

    // This is generated from narrowphase.
    ContactConstraint *contacts;

    // World offsets of the contact constraints.
    int32_t *contactWorldOffsets;

    // World offsets of the positions / velocities.
    int32_t *dofObjectWorldOffsets;
    int32_t *dofObjectWorldCounts;
};

GaussMinimizationNode::GaussMinimizationNode(
        StateManager *s)
    : positions(s->getArchetypeComponent<DofObjectArchetype,
            DofObjectPosition>()),
      velocities(s->getArchetypeComponent<DofObjectArchetype,
            DofObjectVelocity>()),
      numDofs(s->getArchetypeComponent<DofObjectArchetype,
            DofObjectNumDofs>()),
      objIDs(s->getArchetypeComponent<DofObjectArchetype,
            ObjectID>()),
      contacts(s->getArchetypeComponent<Contact,
            ContactConstraint>()),
      contactWorldOffsets(s->getArchetypeWorldOffsets<Contact>()),
      dofObjectWorldOffsets(s->getArchetypeWorldOffsets<DofObjectArchetype>()),
      dofObjectWorldCounts(s->getArchetypeWorldCounts<DofObjectArchetype>())
{
}

void GaussMinimizationNode::solve(int32_t invocation_idx)
{
    uint32_t total_resident_warps = (blockDim.x * gridDim.x) / 32;

    uint32_t total_num_worlds = mwGPU::GPUImplConsts::get().numWorlds;
    uint32_t world_idx = (blockDim.x * blockIdx.x + threadIdx.x) / 32;

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;



    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();
    auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id;

    enum PhaseSMemBytes {

    };

    // TODO: do the magic here!
    // Note, the refs that are in the ContactConstraint are not to the
    // DOF object but to the entity which holds the position / rotation
    // components. This just means we have to inefficiently go through another
    // indirection to get the Dof positions / velocities of affected objects
    // (with that indirection being CVPhysicalComponent).
    // We can fix this by having the contact generation directly output
    // the locs of the DOF objects instead but that we can do in the future.
    assert(blockDim.x == consts::numMegakernelThreads);

    while (world_idx < total_num_worlds) {
        { // Compute data for mass matrix
            ObjectManager &state_mgr =
                mwGPU::getStateManager()->getSingleton<ObjectData>(
                        world_idx).mgr;

            uint32_t num_phys_objs = dofObjectWorldCounts[world_idx];
            uint32_t phys_objs_offset = dofObjectWorldOffsets[world_idx];

            uint32_t current_phys_obj = lane_id;

            uint32_t per_world_alloc = num_phys_objs *
                ((3 * 3 * 4) + // Inverse generalized mass rotation components
                 // TODO
                 0);

            // TODO: Possibly merge this over threadblock to avoid
            // device <-> cpu communication.
            void *per_world_bytes =
                mwGPU::TmpAllocator::get().alloc(per_world_alloc);

            while (current_phys_obj < num_phys_objs) {
                DofObjectPosition gen_pos = positions[current_phys_obj +
                                                      phys_objs_offset];
                ObjectID obj_id = objIDs[current_phys_obj +
                                         phys_objs_offset];

                const RigidBodyMetadata &metadata =
                    obj_mgr.metadata[obj_id.idx];

                Vector3 inv_inertia = metadata.invInertiaTensor;

                Quat rot_quat = {
                    gen_pos.q[3],
                    gen_pos.q[4],
                    gen_pos.q[5],
                    gen_pos.q[6],
                };

                Mat3x3 body_jacob = Mat3x3::fromQuat(rot_quat);
                Diag3x3 inv_inertia = Diag3x3::fromVec(
                    inv_inertia);

                Mat3x3 inv_rot_mass = body_jacob * inv_inertia;
                inv_rot_mass = body_jacob * inv_rot_mass.transpose();

                *((Mat3x3 *)per_world_bytes + current_phys_obj) = inv_rot_mass;

                current_phys_obj += 32;
            }

            __syncwarp();
        }


        world_idx += total_resident_warps;
    }
}

TaskGraph::NodeID GaussMinimizationNode::addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> deps)
{
    using namespace mwGPU;

    // First, we must sort the physics DOF object archetypes by world.
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<
            DofObjectArchetype, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    StateManager *state_mgr = getStateManager();

    auto data_id = builder.constructNodeData<GaussMinimizationNode>(
            state_mgr);
    auto &gauss_data = builder.getDataRef(data_id);

    // For now, we are going with the persistent threads approach where each
    // thread block is going to process a world.
    uint32_t num_invocations = (uint32_t)gridDim.x;

    TaskGraph::NodeID solve_node = builder.addNodeFn<
        &GaussMinimizationNode::solve>(data_id, { post_sort_reset_tmp },
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                // This is the thread block dimension
                consts::numMegakernelThreads);

    return solve_node;
}
#else
static void gaussMinimizeFn(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectNumDofs &numDofs,
                            DofObjectTmpState &tmp_state,
                            ObjectID &objID)
{
    // TODO: Gather all the physics objects and do the minimization
    StateManager *state_mgr = ctx.getStateManager();
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    // Phase 1: compute the mass matrix data
    RigidBodyMetadata &metadata = obj_mgr.metadata[objID.idx];
    float invMass = metadata.mass.invMass;
    Vector3 invInertia = metadata.mass.invInertiaTensor;

    Diag3x3 inv_inertia_body = Diag3x3::fromVec(invInertia);
    Diag3x3 inertia_body = inv_inertia_body.inv();
    Quat rot_quat = {
        position.q[3],
        position.q[4],
        position.q[5],
        position.q[6],
    };
    Vector3 omega = {
        velocity.qv[3],
        velocity.qv[4],
        velocity.qv[5],
    };

    Mat3x3 rot_mat = Mat3x3::fromQuat(rot_quat);

    // Inertia and Inverse Inertia in world frame
    Mat3x3 I_world_frame = rot_mat * inertia_body;
    I_world_frame = rot_mat * I_world_frame.transpose();
    Mat3x3 inv_I_world_frame = rot_mat * inv_inertia_body;
    inv_I_world_frame = rot_mat * inv_I_world_frame.transpose();

    // Step 2: compute the external and gyroscopic forces
    // Step 2.1: gravity
    Vector3 force_gravity = physics_state.g / invMass;
    Vector3 trans_forces = force_gravity;

    // Step 2.2: gyroscopic moments
    Vector3 gyro_moment = -(omega.cross(I_world_frame * omega));
    Vector3 rot_moments = gyro_moment;

    // Infinite mass object, just zero out the forces
    if(invMass == 0) {
        trans_forces = Vector3::zero();
        rot_moments = Vector3::zero();
    }

    tmp_state.invMass = invMass;
    tmp_state.invInertia = inv_I_world_frame;
    tmp_state.externalForces = trans_forces;
    tmp_state.externalMoment = rot_moments;
}

static void processContacts(Context &ctx,
                            ContactConstraint &contact,
                            ContactTmpState &tmp_state)
{
    CVPhysicalComponent ref = ctx.get<CVPhysicalComponent>(
            contact.ref);
    CVPhysicalComponent alt = ctx.get<CVPhysicalComponent>(
            contact.alt);

    auto &ref_pos = ctx.get<DofObjectPosition>(ref.physicsEntity);
    auto &alt_pos = ctx.get<DofObjectPosition>(alt.physicsEntity);
    Vector3 ref_com = Vector3(ref_pos.q[0], ref_pos.q[1], ref_pos.q[2]);
    Vector3 alt_com = Vector3(alt_pos.q[0], alt_pos.q[1], alt_pos.q[2]);

    // Create a coordinate system for the contact
    Vector3 n = contact.normal.normalize();
    Vector3 t{};

    Vector3 x_axis = {1.f, 0.f, 0.f};
    if(n.cross(x_axis).length() > 0.01f) {
        t = n.cross(x_axis).normalize();
    } else {
        t = n.cross({0.f, 1.f, 0.f}).normalize();
    }
    Vector3 s = n.cross(t).normalize();

    // Get the average contact
    float penetration_sum = 0.f;
    Vector3 avg_contact = Vector3::zero();
    float max_depth = -FLT_MAX;
    for(CountT i = 0; i < contact.numPoints; ++i) {
        Vector4 point = contact.points[i];
        penetration_sum += point.w;
        avg_contact += point.w * point.xyz();

        max_depth = std::max(max_depth, point.w);
    }
    avg_contact /= penetration_sum;

    // Compute the relative positions of the contact
    Vector3 rRefComToPt = avg_contact - ref_com;
    Vector3 rAltComToPt = avg_contact - alt_com;

    tmp_state.n = n;
    tmp_state.t = t;
    tmp_state.s = s;
    tmp_state.rRefComToPt = rRefComToPt;
    tmp_state.rAltComToPt = rAltComToPt;
    tmp_state.maxDepth = max_depth;

    // Get friction coefficient
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    CountT objID_i = ctx.get<ObjectID>(ref.physicsEntity).idx;
    CountT objID_j = ctx.get<ObjectID>(alt.physicsEntity).idx;
    RigidBodyMetadata &metadata_i = obj_mgr.metadata[objID_i];
    RigidBodyMetadata &metadata_j = obj_mgr.metadata[objID_j];
    tmp_state.mu = std::min(metadata_i.friction.muS,
                            metadata_j.friction.muS);
}

template <typename MatrixT>
static inline Mat3x3 rightMultiplyCross(const MatrixT &m,
                                        const Vector3 &v)
{
    return {
        {
                { -m[2][0]*v.y + m[1][0]*v.z, -m[2][1]*v.y + m[1][1]*v.z, -m[2][2]*v.y + m[1][2]*v.z },
               { m[2][0]*v.x - m[0][0]*v.z, m[2][1]*v.x - m[0][1]*v.z, m[2][2]*v.x - m[0][2]*v.z },
               { -m[1][0]*v.x + m[0][0]*v.y, -m[1][1]*v.x + m[0][1]*v.y, -m[1][2]*v.x + m[0][2]*v.y }
            }
    };
}

static void solveSystem(Context &ctx,
                        CVSingleton &)
{
    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = ctx.getStateManager();
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    CountT num_contacts = state_mgr->numRows<Contact>(world_id);
    CountT num_bodies = state_mgr->numRows<DofObjectArchetype>(world_id);

    ContactConstraint *contacts = state_mgr->getWorldComponents<
        Contact, ContactConstraint>(world_id);
    ContactTmpState *contacts_tmp_state = state_mgr->getWorldComponents<
        Contact, ContactTmpState>(world_id);
    DofObjectTmpState *tmp_states = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectTmpState>(world_id);
    DofObjectVelocity *body_vels = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectVelocity>(world_id);

    CountT num_dof = 6 * num_bodies;

    // Compute b, v (stack values for each body)
    float *btr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * num_dof);
    float *vptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * num_dof);
    for(CountT i = 0; i < num_bodies; ++i) {
        DofObjectTmpState &body_tmp_state = tmp_states[i];
        DofObjectVelocity &body_vel = body_vels[i];
        CountT body_idx = i * 6;
        btr[body_idx] = body_tmp_state.externalForces[0];
        btr[body_idx + 1] = body_tmp_state.externalForces[1];
        btr[body_idx + 2] = body_tmp_state.externalForces[2];
        btr[body_idx + 3] = body_tmp_state.externalMoment[0];
        btr[body_idx + 4] = body_tmp_state.externalMoment[1];
        btr[body_idx + 5] = body_tmp_state.externalMoment[2];
        vptr[body_idx] = body_vel.qv[0];
        vptr[body_idx + 1] = body_vel.qv[1];
        vptr[body_idx + 2] = body_vel.qv[2];
        vptr[body_idx + 3] = body_vel.qv[3];
        vptr[body_idx + 4] = body_vel.qv[4];
        vptr[body_idx + 5] = body_vel.qv[5];
    }

    CountT jacob_size = (3 * num_contacts) * (num_dof);

    float *jacob_ptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * jacob_size);
    memset(jacob_ptr, 0, sizeof(float) * jacob_size);
    // Lambda for setting in col major order
    auto j_entry = [jacob_ptr, num_contacts, num_dof](int col, int row) -> float & {
        assert(col < num_dof);
        assert(row < 3 * num_contacts);

        return jacob_ptr[row + col * (3 * num_contacts)];
    };

    for (CountT cont_idx = 0; cont_idx < num_contacts; ++cont_idx) {
        ContactConstraint &contact = contacts[cont_idx];
        ContactTmpState &contact_tmp_state = contacts_tmp_state[cont_idx];

        // Get the loc of the two bodies
        CVPhysicalComponent &ref_phys_comp = ctx.get<CVPhysicalComponent>(
                contact.ref);
        CountT ref_idx = ctx.loc(ref_phys_comp.physicsEntity).row;

        CVPhysicalComponent &alt_phys_comp = ctx.get<CVPhysicalComponent>(
                contact.alt);
        CountT alt_idx = ctx.loc(alt_phys_comp.physicsEntity).row;

        CountT row_start = cont_idx * 3;
        CountT ref_col_start = ref_idx * 6;
        CountT alt_col_start = alt_idx * 6;

        Mat3x3 ref_linear_cT;
        Mat3x3 alt_linear_cT;

        // J = [... -C^T, C^T r_i^x, ... C^T, -C^T r_j^x ...]
        for (int i = 0; i < 3; ++i) {
            ref_linear_cT[i][0] = j_entry(ref_col_start + i, row_start) = -contact_tmp_state.n[i];
            ref_linear_cT[i][1] = j_entry(ref_col_start + i, row_start+1) = -contact_tmp_state.t[i];
            ref_linear_cT[i][2] = j_entry(ref_col_start + i, row_start+2) = -contact_tmp_state.s[i];

            alt_linear_cT[i][0] = j_entry(alt_col_start + i, row_start) = contact_tmp_state.n[i];
            alt_linear_cT[i][1] = j_entry(alt_col_start + i, row_start+1) = contact_tmp_state.t[i];
            alt_linear_cT[i][2] = j_entry(alt_col_start + i, row_start+2) = contact_tmp_state.s[i];
        }

        // C^T r_i^x, C^T r_j^x
        ref_linear_cT = rightMultiplyCross(ref_linear_cT,
                                          -contact_tmp_state.rRefComToPt); // need negative to cancel first one
        alt_linear_cT = rightMultiplyCross(alt_linear_cT,
                                          -contact_tmp_state.rAltComToPt);

        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                j_entry(ref_col_start + 3 + col, row_start + row) = ref_linear_cT[col][row];
                j_entry(alt_col_start + 3 + col, row_start + row) = alt_linear_cT[col][row];
            }
        }
    }

    // Create M^{-1}
    CountT M_inv_size = num_dof * num_dof;
    float *M_inv_ptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * M_inv_size);
    memset(M_inv_ptr, 0, sizeof(float) * M_inv_size);

    auto M_inv_entry = [M_inv_ptr, num_dof](int col, int row) -> float & {
        assert(row < num_dof);
        assert(col < num_dof);
        return M_inv_ptr[row + col * num_dof];
    };

    for(CountT body_idx = 0; body_idx < num_bodies; ++body_idx) {
        DofObjectTmpState &tmp_state = tmp_states[body_idx];
        CountT idx_start = body_idx * 6;

        // Diagonal 1 / mass entries
        for (int i = 0; i < 3; ++i) {
            M_inv_entry(idx_start + i, idx_start + i) = tmp_state.invMass;
        }
        // Inverse inertia entries
        for (int i = 0; i < 3; ++i) {
            M_inv_entry(idx_start + i + 3, idx_start + 3) = tmp_state.invInertia[i][0];
            M_inv_entry(idx_start + i + 3, idx_start + 4) = tmp_state.invInertia[i][1];
            M_inv_entry(idx_start + i + 3, idx_start + 5) = tmp_state.invInertia[i][2];
        }
    }

    // Build A=J_c M^{-1} J_c^T
    // First, compute J_c M^{-1} (3 * num_contacts x num_dof)
    float *J_M_inv_ptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * (3 * num_contacts) * num_dof);
    memset(J_M_inv_ptr, 0, sizeof(float) * (3 * num_contacts) * num_dof);
    // We can re-use the lambda earlier since they share same num rows
    auto jm_inv_entry = [J_M_inv_ptr, num_contacts, num_dof](int col, int row) -> float & {
        assert(row < 3 * num_contacts);
        assert(col < num_dof);

        return J_M_inv_ptr[row + col * (3 * num_contacts)];
    };
    for(CountT i = 0; i < 3 * num_contacts; ++i) {
        for(CountT j = 0; j < num_dof; ++j) {
            for(CountT k = 0; k < num_dof; ++k) {
                // (k, j) should be (j, k)
                jm_inv_entry(j, i) += j_entry(k, i) * M_inv_entry(j, k);
            }
        }
    }

    // Then, compute A = J_c M^{-1} J_c^T
    CountT A_size = (3 * num_contacts) * (3 * num_contacts);
    float *A_ptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * A_size);
    memset(A_ptr, 0, sizeof(float) * A_size);
    auto A_entry = [A_ptr, num_contacts](int col, int row) -> float & {
        assert(row < 3 * num_contacts);
        assert(col < 3 * num_contacts);

        return A_ptr[row + col * (3 * num_contacts)];
    };
    for(CountT i = 0; i < 3 * num_contacts; ++i) {
        for(CountT j = 0; j < 3 * num_contacts; ++j) {
            for(CountT k = 0; k < num_dof; ++k) {
                A_entry(j, i) += jm_inv_entry(k, i) * j_entry(k, j);
            }
        }
    }

    // Compute v0 = J_c (v + h M^{-1} b)
    // First compute v + h M^{-1} b
    float *v_h_M_inv_b = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * 6 * num_bodies);
    for(CountT i = 0; i < 6 * num_bodies; ++i) {
        v_h_M_inv_b[i] = vptr[i];
        for(CountT j = 0; j < 6 * num_bodies; ++j) {
            v_h_M_inv_b[i] += physics_state.h * M_inv_entry(j, i) * btr[j];
        }
    }

    // Finally, compute v0 (multiply by J_c)
    float *v0 = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * 3 * num_contacts);
    for(CountT i = 0; i < 3 * num_contacts; ++i) {
        v0[i] = 0.f;
        for(CountT j = 0; j < 6 * num_bodies; ++j) {
            v0[i] += j_entry(j, i) * v_h_M_inv_b[j];
        }
    }

    // Start building f_C
    float *f_C = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * 3 * num_contacts);
    memset(f_C, 0, sizeof(float) * 3 * num_contacts);

    // Begin solving for f_C
    for(CountT i = 0; i < 3 * num_contacts; i += 3) {
        f_C[i] = 100.f; // initial guess
    }

    float* g = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * 3 * num_contacts);

    // Populates gradient and returns the norm
    float *v_C = (float *)state_mgr->tmpAlloc(world_id, sizeof(float) * num_contacts * 3);

    auto grad = [&] (float *gradient, const float* f) -> float {
        memset(v_C, 0, sizeof(float) * num_contacts * 3);

        // v_C = A f_C + v0
        for(CountT i = 0; i < 3 * num_contacts; ++i) {
            v_C[i] = v0[i];
            for(CountT j = 0; j < 3 * num_contacts; ++j) {
                v_C[i] += A_entry(j, i) * f[j];
            }
        }

        // Gradient of main objective is v_C
        for(CountT i = 0; i < 3 * num_contacts; ++i) {
            gradient[i] = v_C[i];
        }

        // Constraint barriers
        float kappa = 10000.f;
        for(CountT i = 0; i < num_contacts; ++i) {
            ContactTmpState &contact_tmp_state = contacts_tmp_state[i];
            CountT idx = 3 * i;

            // Constraint 1: positivity at normals
            float s1 = f[idx];
            gradient[idx] += -kappa * (1 / s1);

            // Second constraint - friction cone
            float mu = contact_tmp_state.mu;
            float s2 = (mu * mu * f[idx] * f[idx]) - f[idx + 1] * f[idx + 1] - f[idx + 2] * f[idx + 2];
            gradient[idx] += -kappa * (2 * (mu * mu) * f[idx]) / s2;
            gradient[idx + 1] += kappa * (2 * f[idx + 1]) / s2;
            gradient[idx + 2] += kappa * (2 * f[idx + 2]) / s2;

            // Third constraint - avoid penetration
            float pen_depth = contact_tmp_state.maxDepth / physics_state.h;
            for(CountT j = 0; j < 3 * num_contacts; ++j) {
                gradient[idx + j] += -kappa * A_entry(idx, j) / (v_C[idx]);
            }
        }

        float norm = 0.f;
        for(CountT i = 0; i < 3 * num_contacts; ++i) {
            norm += gradient[i] * gradient[i];
        }
        printf("Norm: %f\n", sqrt(norm));
        return sqrt(norm);
    };

    float norm = grad(g, f_C);
    while (norm > 0.03f)
    {
        for(CountT i = 0; i < 3 * num_contacts; ++i) {
            f_C[i] -= 0.001f * g[i];
        }
        norm = grad(g, f_C);
    }

    // Post-solve f_C. Impulse is J_c^T f_C
    float *contact_force = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * num_dof);
    for(CountT i = 0; i < num_dof; ++i) {
        contact_force[i] = 0.f;
        for(CountT j = 0; j < 3 * num_contacts; ++j) {
            contact_force[i] += j_entry(i, j) * f_C[j];
        }
    }

    // Add to bodies
    for(CountT i = 0; i < num_bodies; ++i) {
        DofObjectTmpState &body_tmp_state = tmp_states[i];
        CountT idx_start = i * 6;

        printf("contact_force: %f %f %f\n",
                contact_force[idx_start],
                contact_force[idx_start+1],
                contact_force[idx_start+2]);

        if(body_tmp_state.invMass == 0) {
            continue;
        }

        body_tmp_state.externalForces += Vector3(contact_force[idx_start],
                                                contact_force[idx_start + 1],
                                                contact_force[idx_start + 2]);
        body_tmp_state.externalMoment += Vector3(contact_force[idx_start + 3],
                                                contact_force[idx_start + 4],
                                                contact_force[idx_start + 5]);
    }
}

static void integrationStep(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectNumDofs &numDofs,
                            DofObjectTmpState &tmp_state,
                            ObjectID &objID)
{
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    // Phase 1: compute the mass matrix data

    Quat rot_quat = {
        position.q[3],
        position.q[4],
        position.q[5],
        position.q[6],
    };

    float invMass = tmp_state.invMass;
    Mat3x3 inv_I_world_frame = tmp_state.invInertia;

    Vector3 trans_forces = tmp_state.externalForces;
    Vector3 rot_moments = tmp_state.externalMoment;

    Vector3 delta_v = invMass * trans_forces;
    Vector3 delta_omega = inv_I_world_frame * rot_moments;
    float h = physics_state.h;
    for (int i = 0; i < 3; ++i) {
        velocity.qv[i] += h * delta_v[i];
    }
    for (int i = 3; i < 6; ++i) {
        velocity.qv[i] += h * delta_omega[i - 3];
    }

    // Step N: Integrate position
    for (int i = 0; i < 3; ++i) {
        position.q[i] += h * velocity.qv[i];
    }

    // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
    Vector3 omega = { velocity.qv[3], velocity.qv[4], velocity.qv[5] };
    Quat new_rot = {rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z};
    new_rot.w += 0.5f * h * (-rot_quat.x * omega.x - rot_quat.y * omega.y - rot_quat.z * omega.z);
    new_rot.x += 0.5f * h * (rot_quat.w * omega.x + rot_quat.z * omega.y - rot_quat.y * omega.z);
    new_rot.y += 0.5f * h * (-rot_quat.z * omega.x + rot_quat.w * omega.y + rot_quat.x * omega.z);
    new_rot.z += 0.5f * h * (rot_quat.y * omega.x - rot_quat.x * omega.y + rot_quat.w * omega.z);
    new_rot = new_rot.normalize();
    position.q[3] = new_rot.w;
    position.q[4] = new_rot.x;
    position.q[5] = new_rot.y;
    position.q[6] = new_rot.z;
}
#endif

// Convert all the generalized coordinates here.
static void convertPostSolve(
        Context &ctx,
        Position &position,
        Rotation &rotation,
        const CVPhysicalComponent &phys)
{
    Entity physical_entity = phys.physicsEntity;

    DofObjectNumDofs num_dofs = ctx.get<DofObjectNumDofs>(physical_entity);
    DofObjectPosition pos = ctx.get<DofObjectPosition>(physical_entity);

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        position.x = pos.q[0];
        position.y = pos.q[1];
        position.z = pos.q[2];

        rotation = Quat {
            pos.q[3],
            pos.q[4],
            pos.q[5],
            pos.q[6],
        };
    } else {
        MADRONA_UNREACHABLE();
    }
}

}

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<CVPhysicalComponent>();

    registry.registerSingleton<CVSingleton>();

    registry.registerComponent<DofObjectPosition>();
    registry.registerComponent<DofObjectVelocity>();
    registry.registerComponent<DofObjectNumDofs>();
    registry.registerComponent<DofObjectTmpState>();
    registry.registerComponent<ContactTmpState>();

    registry.registerArchetype<DofObjectArchetype>();
    registry.registerArchetype<Contact>();
    registry.registerArchetype<Joint>();

    registry.registerBundle<CVRigidBodyState>();
    registry.registerBundleAlias<SolverBundleAlias, CVRigidBodyState>();
}

void makeFreeBodyEntityPhysical(Context &ctx, Entity e,
                                Position position,
                                Rotation rotation,
                                base::ObjectID obj_id) {
    Entity physical_entity = ctx.makeEntity<DofObjectArchetype>();

    auto &pos = ctx.get<DofObjectPosition>(physical_entity);

    pos.q[0] = position.x;
    pos.q[1] = position.y;
    pos.q[2] = position.z;

    pos.q[3] = rotation.w;
    pos.q[4] = rotation.x;
    pos.q[5] = rotation.y;
    pos.q[6] = rotation.z;

    auto &vel = ctx.get<DofObjectVelocity>(physical_entity);

    vel.qv[0] = 0.f;
    vel.qv[1] = 0.f;
    vel.qv[2] = 0.f;
    vel.qv[3] = 0.f;
    vel.qv[4] = 0.f;
    vel.qv[5] = 0.f;
    ctx.get<base::ObjectID>(physical_entity) = obj_id;

    ctx.get<DofObjectNumDofs>(physical_entity).numDofs = 6;

    ctx.get<CVPhysicalComponent>(e) = {
        .physicsEntity = physical_entity,
    };
}

void cleanupPhysicalEntity(Context &ctx, Entity e)
{
    CVPhysicalComponent physical_comp = ctx.get<CVPhysicalComponent>(e);
    ctx.destroyEntity(physical_comp.physicsEntity);
}

TaskGraphNodeID setupCVSolverTasks(TaskGraphBuilder &builder,
                                   TaskGraphNodeID broadphase,
                                   CountT num_substeps)
{
    auto cur_node = broadphase;

    for (CountT i = 0; i < num_substeps; ++i) {
        auto run_narrowphase = narrowphase::setupTasks(builder, {cur_node});

#ifdef MADRONA_GPU_MODE
        // We need to sort the contacts by world.
        run_narrowphase = builder.addToGraph<
            SortArchetypeNode<Contact, WorldID>>(
                {run_narrowphase});
        run_narrowphase = builder.addToGraph<ResetTmpAllocNode>(
                {run_narrowphase});
#endif

#ifdef MADRONA_GPU_MODE
        auto gauss_node = builder.addToGraph<tasks::GaussMinimizationNode>(
                {run_narrowphase});

        gauss_node = builder.addToGraph<ResetTmpAllocNode>(
                {gauss_node});
#else
        auto gauss_node = builder.addToGraph<ParallelForNode<Context,
             tasks::gaussMinimizeFn,
                DofObjectPosition,
                DofObjectVelocity,
                DofObjectNumDofs,
                DofObjectTmpState,
                ObjectID
            >>({run_narrowphase});

        auto contact_node = builder.addToGraph<ParallelForNode<Context,
             tasks::processContacts,
                ContactConstraint,
                ContactTmpState
            >>({gauss_node});

        auto solve_sys = builder.addToGraph<ParallelForNode<Context,
             tasks::solveSystem,
                CVSingleton
            >>({contact_node});

        auto int_node = builder.addToGraph<ParallelForNode<Context,
             tasks::integrationStep,
                 DofObjectPosition,
                 DofObjectVelocity,
                 DofObjectNumDofs,
                 DofObjectTmpState,
                 ObjectID
            >>({solve_sys});
#endif

        cur_node =
            builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
                Position,
                Rotation,
                CVPhysicalComponent
            >>({int_node});

        cur_node = builder.addToGraph<
            ClearTmpNode<Contact>>({cur_node});

        cur_node = builder.addToGraph<ResetTmpAllocNode>({cur_node});
    }

    auto clear_broadphase = builder.addToGraph<
        ClearTmpNode<CandidateTemporary>>({cur_node});
    
    return clear_broadphase;
}



void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id)
{
    *contact_archetype_id = TypeTracker::typeID<Contact>();
    *joint_archetype_id = TypeTracker::typeID<Joint>();
}

void init(Context &ctx)
{
    // Nothing for now
    (void)ctx;
}
    
}
