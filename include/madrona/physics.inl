#pragma once

namespace madrona {
namespace phys {

namespace broadphase {

LeafID BVH::reserveLeaf(Entity e)
{
    int32_t leaf_idx = num_leaves_.fetch_add(1, std::memory_order_relaxed);
    assert(leaf_idx < num_allocated_leaves_);

    leaf_entities_[leaf_idx] = e;

    return LeafID {
        leaf_idx,
    };
}

template <typename Fn>
void BVH::findOverlaps(const math::AABB &aabb, Fn &&fn) const
{
    int32_t stack[128];
    stack[0] = 0;
    CountT stack_size = 1;

    while (stack_size > 0) {
        int32_t node_idx = stack[--stack_size];
        const Node &node = nodes_[node_idx];
        for (int i = 0; i < 4; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };

            madrona::math::AABB child_aabb {
                /* .pMin = */ {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                /* .pMax = */ {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            };

            if (aabb.overlaps(child_aabb)) {
                if (node.isLeaf(i)) {
                    Entity e = leaf_entities_[node.leafIDX(i)];
                    fn(e);
                } else {
                    stack[stack_size++] = node.children[i];
                }
            }
        }
    }
}

float BVH::traceRay(math::Vector3 o, math::Vector3 d, float t_max)
{
    using namespace math;

    Vector3 inv_d = 1.f / d;

    int32_t stack[128];
    stack[0] = 0;
    CountT stack_size = 1;

    while (stack_size > 0) { 
        int32_t node_idx = stack[--stack_size];
        const Node &node = nodes_[node_idx];
        for (int i = 0; i < 4; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };

            madrona::math::AABB child_aabb {
                /* .pMin = */ {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                /* .pMax = */ {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            };

            if (child_aabb.rayIntersects(o, inv_d, 0.f, t_max)) {
                if (node.isLeaf(i)) {
                    // FIXME: this leaf information should probably be copied
                    // inline into the tree to avoid the entity lookup
                    Entity e = leaf_entities_[node.leafIDX(i)];
                    traceRayIntoEntity(o, d, t_max, e);
                } else {
                    stack[stack_size++] = node.children[i];
                }
            }
        }
    }
    
    // FIXME
    return false;
}

bool BVH::traceRayIntoEntity(math::Vector3 o,
                             math::Vector3 d,
                             float t_max,
                             Entity e)
{
    // GPU GEMS II, Fast Ray - Convex Polyhedron Intersection
    
    using namespace math;

    (void)o;
    (void)d;
    (void)t_max;
    (void)e;

    return false;

#if 0
/* fast macro version of V3Dot, usable with Point4 */
#define DOT3( a, b )	( (a)->x*(b)->x + (a)->y*(b)->y + (a)->z*(b)->z )

/* return codes */
#define	MISSED		 0
#define	FRONTFACE	 1
#define	BACKFACE	-1

int	RayCvxPolyhedronInt( org, dir, tmax, phdrn, ph_num, tresult, norm )
Point3	*org, *dir ;	/* origin and direction of ray */
double	tmax ;		/* maximum useful distance along ray */
Point4	*phdrn ;	/* list of planes in convex polyhedron */
int	ph_num ;	/* number of planes in convex polyhedron */
double	*tresult ;	/* returned: distance of intersection along ray */
Point3	*norm ;		/* returned: normal of face hit */
{
Point4	*pln ;			/* plane equation */
double	tnear, tfar, t, vn, vd ;
int	fnorm_num, bnorm_num ;	/* front/back face # hit */

    tnear = -FLT_MAX;
    tfar = tmax ;

    /* Test each plane in polyhedron */
    for ( pln = &phdrn[ph_num-1] ; ph_num-- ; pln-- ) {
	/* Compute intersection point T and sidedness */
	vd = DOT3( dir, pln ) ;
	vn = DOT3( org, pln ) + pln->w ;
	if ( vd == 0.0 ) {
	    /* ray is parallel to plane - check if ray origin is inside plane's
	       half-space */
	    if ( vn > 0.0 )
		/* ray origin is outside half-space */
		return ( MISSED ) ;
	} else {
	    /* ray not parallel - get distance to plane */
	    t = -vn / vd ;
	    if ( vd < 0.0 ) {
		/* front face - T is a near point */
		if ( t > tfar ) return ( MISSED ) ;
		if ( t > tnear ) {
		    /* hit near face, update normal */
		    fnorm_num = ph_num ;
		    tnear = t ;
		}
	    } else {
		/* back face - T is a far point */
		if ( t < tnear ) return ( MISSED ) ;
		if ( t < tfar ) {
		    /* hit far face, update normal */
		    bnorm_num = ph_num ;
		    tfar = t ;
		}
	    }
	}
    }

    /* survived all tests */
    /* Note: if ray originates on polyhedron, may want to change 0.0 to some
     * epsilon to avoid intersecting the originating face.
     */
    if ( tnear >= 0.0 ) {
	/* outside, hitting front face */
	*norm = *(Point3 *)&phdrn[fnorm_num] ;
	*tresult = tnear ;
	return ( FRONTFACE ) ;
    } else {
	if ( tfar < tmax ) {
	    /* inside, hitting back face */
	    *norm = *(Point3 *)&phdrn[bnorm_num] ;
	    *tresult = tfar ;
	    return ( BACKFACE ) ;
	} else {
	    /* inside, but back face beyond tmax */
	    return ( MISSED ) ;
	}
    }
}
#endif
}

void BVH::rebuildOnUpdate()
{
    force_rebuild_ = true;
}

void BVH::clearLeaves()
{
    num_leaves_.store(0, std::memory_order_relaxed);
}

bool BVH::Node::isLeaf(CountT child) const
{
    return children[child] & 0x80000000;
}

int32_t BVH::Node::leafIDX(CountT child) const
{
    return children[child] & ~0x80000000;
}

void BVH::Node::setLeaf(CountT child, int32_t idx)
{
    children[child] = 0x80000000 | idx;
}

void BVH::Node::setInternal(CountT child, int32_t internal_idx)
{
    children[child] = internal_idx;
}

bool BVH::Node::hasChild(CountT child) const
{
    return children[child] != sentinel_;
}

void BVH::Node::clearChild(CountT child)
{
    children[child] = sentinel_;
}

}

}

}
