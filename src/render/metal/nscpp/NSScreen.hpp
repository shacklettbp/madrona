#pragma once 

#include "AppKitPrivate.hpp"

#include <Foundation/NSObject.hpp>

namespace NS {

class Screen : public Referencing<Screen> {
public:
    static Screen * main();

    CGFloat backingScaleFactor();
};

_NS_INLINE Screen * Screen::main()
{
    return Object::sendMessage<Screen *>(_APPKIT_PRIVATE_CLS(NSScreen),
                                         _APPKIT_PRIVATE_SEL(mainScreen));
}

_NS_INLINE CGFloat Screen::backingScaleFactor()
{
    return Object::sendMessage<CGFloat>(
        this, _APPKIT_PRIVATE_SEL(backingScaleFactor));
}

}
