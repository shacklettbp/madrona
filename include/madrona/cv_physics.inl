namespace madrona::phys::cv {

uint32_t BodyOffsets::getDofTypeDim(DofType type, bool is_pos)
{
    switch (type) {
    case DofType::FreeBody: {
        return is_pos ? 7 : 6;
    }

    case DofType::Hinge: {
        return 1;
    } break;

    case DofType::Ball: {
        return 3;
    }

    case DofType::FixedBody: {
        return is_pos ? 7 : 0;
    }

    case DofType::Slider: {
        return 1;
    }

    default: {
        // Something is wrong if we're here
        MADRONA_UNREACHABLE();
        return 0;
    }
    }
}

float * BodyGroupMemory::q(BodyGroupProperties) { return (float *)qVectorsPtr; }
float * BodyGroupMemory::qv(BodyGroupProperties p) { return q(p) + p.qDim; }
float * BodyGroupMemory::dqv(BodyGroupProperties p) { return qv(p) + p.qvDim; }
float * BodyGroupMemory::f(BodyGroupProperties p) { return dqv(p) + p.qvDim; }
float * BodyGroupMemory::mus(BodyGroupProperties p) { return f(p) + p.qvDim; }
BodyLimitConstraint * BodyGroupMemory::limits(BodyGroupProperties p) { return (BodyLimitConstraint *)(mus(p) + p.numBodies); }
BodyInertial * BodyGroupMemory::inertials(BodyGroupProperties p) { return (BodyInertial *)(limits(p) + p.numEq); }
int32_t * BodyGroupMemory::expandedParent(BodyGroupProperties p) { return (int32_t *)(inertials(p) + p.numBodies); }
BodyObjectData * BodyGroupMemory::objectData(BodyGroupProperties p) { return (BodyObjectData *)(expandedParent(p) + p.qvDim); }
BodyHierarchy * BodyGroupMemory::hierarchies(BodyGroupProperties p) { return (BodyHierarchy *)(objectData(p) + p.numObjData); }
Entity * BodyGroupMemory::entities(BodyGroupProperties p) { return (Entity *)(hierarchies(p) + p.numBodies); }
BodyOffsets * BodyGroupMemory::offsets(BodyGroupProperties p) { return (BodyOffsets *)(entities(p) + p.numBodies); }




BodyTransform * BodyGroupMemory::bodyTransforms(BodyGroupProperties) { return (BodyTransform *)tmpPtr; }
BodyPhi * BodyGroupMemory::bodyPhi(BodyGroupProperties p) { return (BodyPhi *)(bodyTransforms(p) + p.numBodies); }
BodySpatialVectors * BodyGroupMemory::spatialVectors(BodyGroupProperties p) { return (BodySpatialVectors *)(bodyPhi(p) + p.numBodies); }
float * BodyGroupMemory::biasVector(BodyGroupProperties p) { return (float *)(spatialVectors(p) + p.numBodies); }
float * BodyGroupMemory::massMatrix(BodyGroupProperties p) { return (float *)(biasVector(p) + p.qvDim); }
float * BodyGroupMemory::massLTDLMatrix(BodyGroupProperties p) { return (float *)(massMatrix(p) + p.qvDim * p.qvDim); }
float * BodyGroupMemory::phiFull(BodyGroupProperties p) { return (float *)(massLTDLMatrix(p) + p.qvDim * p.qvDim); }
inline float * BodyGroupMemory::scratch(BodyGroupProperties p) { return (float *)(phiFull(p) + p.qvDim * 2 * 6); }

inline uint32_t BodyGroupMemory::qVectorsNumBytes(BodyGroupProperties p)
{
    return p.qDim * sizeof(float) +                 // q
           p.qvDim * sizeof(float) +                // qv
           p.qvDim * sizeof(float) +                // dqv
           p.qvDim * sizeof(float) +               // force
           p.numBodies * sizeof(float) +            // mus
           p.numEq * sizeof(BodyLimitConstraint) +  // equalities
           p.numBodies * sizeof(BodyInertial) +     // inertias
           p.qvDim * sizeof(int32_t) +              // expanded parent
           p.numObjData * sizeof(BodyObjectData) +  // body object data
           p.numBodies * sizeof(BodyHierarchy) +    // body hierarchy
           p.numBodies * sizeof(Entity) +
           p.numBodies * sizeof(BodyOffsets);
}

inline uint32_t BodyGroupMemory::tmpNumBytes(BodyGroupProperties p)
{
    return p.numBodies * sizeof(BodyTransform) +        // com / rotation
           p.numBodies * sizeof(BodyPhi) +              // phi
           p.numBodies * sizeof(BodySpatialVectors) +   // spatial vectors
           p.qvDim * sizeof(float) +                    // bias vector
           p.qvDim * p.qvDim * sizeof(float) +          // mass matrix
           p.qvDim * p.qvDim * sizeof(float) +          // LTDL mass matrix
           p.qvDim * 2 * 6 * sizeof(float) +
           36 * sizeof(float);
}

SpatialVector SpatialVector::fromVec(const float* v)
{
    return { 
        {v[0], v[1], v[2]}, 
        {v[3], v[4], v[5]} 
    };
}

float SpatialVector::operator[](const CountT i) const
{
    return i < 3 ? linear[i] : angular[i - 3];
}

float & SpatialVector::operator[](const CountT i)
{
    return i < 3 ? linear[i] : angular[i - 3];
}

SpatialVector & SpatialVector::operator+=(const SpatialVector &rhs)
{
    linear += rhs.linear;
    angular += rhs.angular;
    return *this;
}

SpatialVector & SpatialVector::operator-=(const SpatialVector &rhs)
{
    linear -= rhs.linear;
    angular -= rhs.angular;
    return *this;
}

SpatialVector SpatialVector::cross(const SpatialVector &rhs) const
{
    return {
        angular.cross(rhs.linear) + linear.cross(rhs.angular),
        angular.cross(rhs.angular)
    };
}

SpatialVector SpatialVector::crossStar(const SpatialVector &rhs) const
{
    return {
        angular.cross(rhs.linear),
        angular.cross(rhs.angular) + linear.cross(rhs.linear)
    };
}

InertiaTensor & InertiaTensor::operator+=(const InertiaTensor &rhs)
{
    mass += rhs.mass;
    mCom += rhs.mCom;

    #pragma unroll
    for (int i = 0; i < 6; i++) {
        spatial_inertia[i] += rhs.spatial_inertia[i];
    }

    return *this;
}

// Multiply with vector [v] of length 6, storing the result in [out]
void InertiaTensor::multiply(const float* v, float* out) const
{
    math::Vector3 v_trans = {v[0], v[1], v[2]};
    math::Vector3 v_rot = {v[3], v[4], v[5]};
    math::Vector3 out_trans = mass * v_trans - mCom.cross(v_rot);
    math::Vector3 out_rot = mCom.cross(v_trans);

    out_rot[0] += spatial_inertia[0] * v_rot[0] + 
                  spatial_inertia[3] * v_rot[1] + 
                  spatial_inertia[4] * v_rot[2];

    out_rot[1] += spatial_inertia[3] * v_rot[0] + 
                  spatial_inertia[1] * v_rot[1] +
                  spatial_inertia[5] * v_rot[2];

    out_rot[2] += spatial_inertia[4] * v_rot[0] + 
                  spatial_inertia[5] * v_rot[1] +
                  spatial_inertia[2] * v_rot[2];

    out[0] = out_trans.x;
    out[1] = out_trans.y;
    out[2] = out_trans.z;
    out[3] = out_rot.x;
    out[4] = out_rot.y;
    out[5] = out_rot.z;
}

SpatialVector InertiaTensor::multiply(const SpatialVector &v) const
{
    SpatialVector out;
    out.linear = mass * v.linear - mCom.cross(v.angular);
    out.angular = mCom.cross(v.linear);

    out.angular[0] += spatial_inertia[0] * v.angular[0] + 
                      spatial_inertia[3] * v.angular[1] +
                      spatial_inertia[4] * v.angular[2];

    out.angular[1] += spatial_inertia[3] * v.angular[0] + 
                      spatial_inertia[1] * v.angular[1] +
                      spatial_inertia[5] * v.angular[2];

    out.angular[2] += spatial_inertia[4] * v.angular[0] + 
                      spatial_inertia[5] * v.angular[1] +
                      spatial_inertia[2] * v.angular[2];

    return out;
}

float HingeLimit::dConstraintViolation(float q)
{
    float c1 = q - lower;
    float c2 = upper - q;

    if (c1 < 0.f) {
        return 1.f;
    } else if (c2 < 0.f) {
        return -1.f;
    } else {
        // Anything
        return 0.f;
    }
}

float HingeLimit::constraintViolation(float q)
{
    float c1 = q - lower;
    float c2 = upper - q;

    if (c1 < 0.f) {
        return c1;
    } else if (c2 < 0.f) {
        return c2;
    } else {
        // Anything
        return 0.f;
    }
}

float SliderLimit::dConstraintViolation(float q)
{
    float c1 = q - lower;
    float c2 = upper - q;

    if (c1 < 0.f) {
        return 1.f;
    } else if (c2 < 0.f) {
        return -1.f;
    } else {
        // Return anything
        return 0.f;
    }
}

float SliderLimit::constraintViolation(float q)
{
    float c1 = q - lower;
    float c2 = upper - q;

    if (c1 < 0.f) {
        return c1;
    } else if (c2 < 0.f) {
        return c2;
    } else {
        // Anything
        return 0.f;
    }
}

}
