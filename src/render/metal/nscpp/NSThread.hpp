#include "AppKitPrivate.hpp"
#include <Foundation/NSObject.hpp>

namespace NS {

class Thread : public Referencing<Thread> {
public:
    static void detachNewThreadSelector(SEL sel, NS::Object *tgt, void *obj);
};

_NS_INLINE void Thread::detachNewThreadSelector(
    SEL sel, NS::Object *tgt, void *obj)
{
    Object::sendMessage<void>(_APPKIT_PRIVATE_CLS(NSThread),
        _APPKIT_PRIVATE_SEL(detachNewThreadSelector_toTarget_withObject_),
        sel, tgt, obj);
}

}
