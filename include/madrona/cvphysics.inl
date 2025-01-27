namespace madrona::phys::cv {

uint32_t getNumDofs(DofType type)
{
    switch (type) {
    case DofType::FreeBody: {
        return 6;
    }

    case DofType::Hinge: {
        return 1;
    } break;

    case DofType::Ball: {
        return 3;
    }

    case DofType::FixedBody: {
        return 0;
    }

    case DofType::Sliding: {
        return 1;
    }

    case DofType::None: {
        // Something is wrong if we're here
        MADRONA_UNREACHABLE();
    }
    }
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
        // Return anything
        return -1.f;
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
        return -1.f;
    }
}

}
