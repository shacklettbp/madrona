/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <type_traits>

namespace madrona::utils {

template <typename> struct PackDelegator;
template <template <typename...> typename T, typename ...Args>
struct PackDelegator<T<Args...>> {
    template <typename Fn>
    static auto call(Fn &&fn) -> decltype(fn.template operator()<Args...>())
    {
        return fn.template operator()<Args...>();
    }
};

// Extract the type of the first argument. Not fully fleshed out but works for
// the needed cases currently.
template <typename Fn> struct FirstArgTypeExtractor;

template <typename ReturnT, typename FirstT, typename... ArgsT>
struct FirstArgTypeExtractor<ReturnT(FirstT, ArgsT...)> {
    using type = FirstT;
};

template <typename ReturnT, typename ClassT, typename FirstT,
          typename... ArgsT>
struct FirstArgTypeExtractor<ReturnT (ClassT::*)(FirstT, ArgsT...)> {
    using type = FirstT;
};

template <typename ReturnT, typename ClassT, typename FirstT,
          typename... ArgsT>
struct FirstArgTypeExtractor<ReturnT (ClassT::*)(FirstT, ArgsT...) const> {
    using type = FirstT;
};

template <typename T>
struct ExtractClassFromMemberPtr;

template <typename ClassT, typename T>
struct ExtractClassFromMemberPtr<T ClassT::*> {
    using type = ClassT;
};

}
