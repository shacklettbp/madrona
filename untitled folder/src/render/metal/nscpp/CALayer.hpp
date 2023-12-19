#pragma once

#include <Foundation/Foundation.hpp>
#include <QuartzCore/CAPrivate.hpp>

namespace CA {

class Layer : NS::Referencing<Layer> {
public:
    CGFloat contentsScale();
    void setContentsScale(CGFloat scale);
};

namespace Private {
namespace Selector {

_CA_PRIVATE_DEF_SEL(contentsScale,
                    "contentsScale");

_CA_PRIVATE_DEF_SEL(setContentsScale_,
                    "setContentsScale:");

}
}

}

_NS_INLINE CGFloat CA::Layer::contentsScale()
{
	return Object::sendMessage<CGFloat>(this, _CA_PRIVATE_SEL(contentsScale));
}

_NS_INLINE void CA::Layer::setContentsScale(CGFloat scale)
{
	return Object::sendMessage<void>(
        this, _CA_PRIVATE_SEL(setContentsScale_), scale);
}
