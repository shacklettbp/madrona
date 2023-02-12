#pragma once
#include <CoreGraphics/CGEventTypes.h>
#include "AppKitPrivate.hpp"
#include <Foundation/NSTypes.hpp>
#include <Foundation/NSObject.hpp>

namespace NS {

// various types of events
// Copied from NSEvent.h
_NS_ENUM(UInteger, EventType) {
    EventTypeLeftMouseDown             = 1,
    EventTypeLeftMouseUp               = 2,
    EventTypeRightMouseDown            = 3,
    EventTypeRightMouseUp              = 4,
    EventTypeMouseMoved                = 5,
    EventTypeLeftMouseDragged          = 6,
    EventTypeRightMouseDragged         = 7,
    EventTypeMouseEntered              = 8,
    EventTypeMouseExited               = 9,
    EventTypeKeyDown                   = 10,
    EventTypeKeyUp                     = 11,
    EventTypeFlagsChanged              = 12,
    EventTypeAppKitDefined             = 13,
    EventTypeSystemDefined             = 14,
    EventTypeApplicationDefined        = 15,
    EventTypePeriodic                  = 16,
    EventTypeCursorUpdate              = 17,
    EventTypeScrollWheel               = 22,
    EventTypeTabletPoint               = 23,
    EventTypeTabletProximity           = 24,
    EventTypeOtherMouseDown            = 25,
    EventTypeOtherMouseUp              = 26,
    EventTypeOtherMouseDragged         = 27,
    /* The following event types are available on some hardware on 10.5.2 and later */
    EventTypeGesture                   = 29,
    EventTypeMagnify                   = 30,
    EventTypeSwipe                     = 31,
    EventTypeRotate                    = 18,
    EventTypeBeginGesture              = 19,
    EventTypeEndGesture                = 20,
    EventTypeSmartMagnify              = 32,
    EventTypeQuickLook                 = 33,
    EventTypePressure                  = 34,
    EventTypeDirectTouch               = 37,
    EventTypeChangeMode                = 38,
};

// For APIs introduced in Mac OS X 10.6 and later, this type is used with NS*Mask constants to indicate the events of interest.
// masks for the types of events
_NS_OPTIONS(unsigned long long, EventMask) {
    EventMaskLeftMouseDown         = 1ULL << EventTypeLeftMouseDown,
    EventMaskLeftMouseUp           = 1ULL << EventTypeLeftMouseUp,
    EventMaskRightMouseDown        = 1ULL << EventTypeRightMouseDown,
    EventMaskRightMouseUp          = 1ULL << EventTypeRightMouseUp,
    EventMaskMouseMoved            = 1ULL << EventTypeMouseMoved,
    EventMaskLeftMouseDragged      = 1ULL << EventTypeLeftMouseDragged,
    EventMaskRightMouseDragged     = 1ULL << EventTypeRightMouseDragged,
    EventMaskMouseEntered          = 1ULL << EventTypeMouseEntered,
    EventMaskMouseExited           = 1ULL << EventTypeMouseExited,
    EventMaskKeyDown               = 1ULL << EventTypeKeyDown,
    EventMaskKeyUp                 = 1ULL << EventTypeKeyUp,
    EventMaskFlagsChanged          = 1ULL << EventTypeFlagsChanged,
    EventMaskAppKitDefined         = 1ULL << EventTypeAppKitDefined,
    EventMaskSystemDefined         = 1ULL << EventTypeSystemDefined,
    EventMaskApplicationDefined    = 1ULL << EventTypeApplicationDefined,
    EventMaskPeriodic              = 1ULL << EventTypePeriodic,
    EventMaskCursorUpdate          = 1ULL << EventTypeCursorUpdate,
    EventMaskScrollWheel           = 1ULL << EventTypeScrollWheel,
    EventMaskTabletPoint           = 1ULL << EventTypeTabletPoint,
    EventMaskTabletProximity       = 1ULL << EventTypeTabletProximity,
    EventMaskOtherMouseDown        = 1ULL << EventTypeOtherMouseDown,
    EventMaskOtherMouseUp          = 1ULL << EventTypeOtherMouseUp,
    EventMaskOtherMouseDragged     = 1ULL << EventTypeOtherMouseDragged,
    /* The following event masks are available on some hardware on 10.5.2 and later */
    EventMaskGesture               = 1ULL << EventTypeGesture,
    EventMaskMagnify               = 1ULL << EventTypeMagnify,
    EventMaskSwipe                 = 1ULL << EventTypeSwipe,
    EventMaskRotate                = 1ULL << EventTypeRotate,
    EventMaskBeginGesture          = 1ULL << EventTypeBeginGesture,
    EventMaskEndGesture            = 1ULL << EventTypeEndGesture,
    
    /* Note: You can only use these event masks on 64 bit. In other words, you cannot setup a local, nor global, event monitor for these event types on 32 bit. Also, you cannot search the event queue for them (nextEventMatchingMask:...) on 32 bit.
     */
    EventMaskSmartMagnify          = 1ULL << EventTypeSmartMagnify,
    EventMaskPressure              = 1ULL << EventTypePressure,
    EventMaskDirectTouch           = 1ULL << EventTypeDirectTouch,

    EventMaskChangeMode            = 1ULL << EventTypeChangeMode,
    
    EventMaskAny                   = UIntegerMax,
};

/* Device-independent bits found in event modifier flags */
_NS_OPTIONS(UInteger, EventModifierFlags) {
    EventModifierFlagCapsLock           = 1 << 16, // Set if Caps Lock key is pressed.
    EventModifierFlagShift              = 1 << 17, // Set if Shift key is pressed.
    EventModifierFlagControl            = 1 << 18, // Set if Control key is pressed.
    EventModifierFlagOption             = 1 << 19, // Set if Option or Alternate key is pressed.
    EventModifierFlagCommand            = 1 << 20, // Set if Command key is pressed.
    EventModifierFlagNumericPad         = 1 << 21, // Set if any key in the numeric keypad is pressed.
    EventModifierFlagHelp               = 1 << 22, // Set if the Help key is pressed.
    EventModifierFlagFunction           = 1 << 23, // Set if any function key is pressed.
    
    // Used to retrieve only the device-independent modifier flags, allowing applications to mask off the device-dependent modifier flags, including event coalescing information.
    EventModifierFlagDeviceIndependentFlagsMask    = 0xffff0000UL
};

#define NX_SUBTYPE_DEFAULT					0
#define NX_SUBTYPE_TABLET_POINT				1
#define NX_SUBTYPE_TABLET_PROXIMITY			2
#define NX_SUBTYPE_MOUSE_TOUCH              3

_NS_ENUM(short, NSEventSubtype) {
    /* event subtypes for NSEventTypeAppKitDefined events */
    EventSubtypeWindowExposed            = 0,
    EventSubtypeApplicationActivated     = 1,
    EventSubtypeApplicationDeactivated   = 2,
    EventSubtypeWindowMoved              = 4,
    EventSubtypeScreenChanged            = 8,
    
    /* event subtypes for NSEventTypeSystemDefined events */
    EventSubtypePowerOff             = 1,
    
    /* event subtypes for mouse events */
    EventSubtypeMouseEvent        = NX_SUBTYPE_DEFAULT,
    EventSubtypeTabletPoint       = NX_SUBTYPE_TABLET_POINT,
    EventSubtypeTabletProximity   = NX_SUBTYPE_TABLET_PROXIMITY,
    EventSubtypeTouch             = NX_SUBTYPE_MOUSE_TOUCH
};

namespace Private {
namespace Class {
_APPKIT_PRIVATE_DEF_CLS(NSEvent);
}

namespace Selector {
_APPKIT_PRIVATE_DEF_SEL(otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data_data2_, "otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:");
}
}

class Event : public Referencing<Event> {
public:
    static _NS_INLINE Event * otherEventWithType(EventType type,
                                                 CGPoint location,
                                                 EventModifierFlags flags,
                                                 TimeInterval timestamp,
                                                 Integer wNum,
                                                 void *unused,
                                                 int16_t subtype,
                                                 Integer d1,
                                                 Integer d2);
};

_NS_INLINE Event * Event::otherEventWithType(EventType type,
                                             CGPoint location,
                                             EventModifierFlags flags,
                                             TimeInterval timestamp,
                                             Integer wNum,
                                             void *unused,
                                             int16_t subtype,
                                             Integer d1,
                                             Integer d2)
{
	return Object::sendMessage<Event *>(_APPKIT_PRIVATE_CLS(NSEvent),
                                        _APPKIT_PRIVATE_SEL(otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data_data2_),
                                        type,
                                        location,
                                        flags,
                                        timestamp,
                                        wNum,
                                        unused,
                                        subtype,
                                        d1,
                                        d2);
}

}
