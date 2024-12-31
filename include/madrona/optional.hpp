/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <type_traits>
#include <memory>
#include <utility>

namespace madrona {

template <typename T>
class Optional {
public:
    static constexpr Optional<T> none()
    {
        return Optional<T>();
    }

    template <typename... Args>
    static constexpr Optional<T> make(Args && ...args)
    {
        return Optional<T>(std::in_place, std::forward<Args>(args)...);
    }

    static constexpr void noneAt(Optional<T> *ptr)
    {
        new (ptr) Optional<T>();
    }

    template <typename... Args>
    static constexpr void makeAt(Optional<T> *ptr, Args && ...args)
    {
        new (ptr) Optional<T>(std::in_place, std::forward<Args>(args)...);
    }

    template <typename... Args>
    T & emplace(Args && ...args)
    {
        destruct();

        this->construct(std::forward<Args>(args)...);
        initialized_ = true;

        return value_;
    }

    constexpr Optional(const Optional<T> &o)
        requires (std::is_trivially_copy_constructible_v<T>) = default;

    constexpr Optional(const Optional<T> &o)
        requires (std::is_copy_constructible_v<T> &&
                  !std::is_trivially_copy_constructible_v<T>)
    {
        this->initialized_ = o.initialized_;

        if (this->initialized_) {
            std::construct_at(&this->value_, o.value_);
        } 
    }

    constexpr Optional(Optional<T> &&o)
        requires (std::is_trivially_move_constructible_v<T>) = default;

    constexpr Optional(Optional<T> &&o)
        requires (std::is_move_constructible_v<T> &&
                  !std::is_trivially_move_constructible_v<T>)
    {
        this->initialized_ = o.initialized_;

        if (this->initialized_) {
            std::construct_at(&this->value_, std::forward<T>(o.value_));
        } 
    }

    template <typename U = T>
    constexpr explicit(!std::is_convertible_v<U&&, T>) Optional(U &&o)
        requires(std::is_constructible_v<T, U&&> &&
                 !std::is_same_v<std::remove_cvref_t<U>, Optional<T>>)
        : value_(std::forward<U>(o)),
          initialized_(true)
    {}

    constexpr ~Optional()
        requires (std::is_trivially_destructible_v<T>) = default;

    constexpr ~Optional()
        requires (!std::is_trivially_destructible_v<T>)
    {
        destruct();
    }

    constexpr Optional<T> & operator=(const Optional<T> &o)
        requires (std::is_trivially_copy_assignable_v<T> &&
                  std::is_trivially_copy_constructible_v<T> &&
                  std::is_trivially_destructible_v<T>) = default;

    constexpr Optional<T> & operator=(const Optional<T> &o)
        requires (std::is_copy_assignable_v<T> &&
                  std::is_copy_constructible_v<T> && !(
                      std::is_trivially_copy_assignable_v<T> &&
                      std::is_trivially_copy_constructible_v<T> &&
                      std::is_trivially_destructible_v<T>))
    {
        if (this->initialized_) {
            if (o.initialized_) {
                this->value_ = o.value_;
            } else {
                this->destruct();
                this->initialized_ = false;
            }
        } else if (o.initialized_) {
            this->construct(o.value_);
            this->initialized_ = true;
        }
        
        return *this;
    }

    constexpr Optional<T> & operator=(Optional<T> &&o)
        requires (std::is_trivially_move_assignable_v<T> &&
                  std::is_trivially_move_constructible_v<T> &&
                  std::is_trivially_destructible_v<T>) = default;

    constexpr Optional<T> & operator=(Optional<T> &&o)
        requires (std::is_move_assignable_v<T> &&
                  std::is_move_constructible_v<T> && !(
                      std::is_trivially_move_assignable_v<T> &&
                      std::is_trivially_move_constructible_v<T> &&
                      std::is_trivially_destructible_v<T>))
    {
        if (this->initialized_) {
            if (o.initialized_) {
                this->value_ = std::forward<T>(o.value_);
            } else {
                this->destruct();
                this->initialized_ = false;
            }
        } else if (o.initialized_) {
            this->construct(std::forward<T>(o.value_));
            this->initialized_ = true;
        }
        
        return *this;
    }

    template <typename U = T>
    constexpr Optional<T> & operator=(U &&o)
        requires(!std::is_same_v<std::remove_cvref_t<U>, Optional<T>> &&
                 std::is_constructible_v<T, U> &&
                 std::is_assignable_v<T&, U> &&
                 (!std::is_scalar_v<T> ||
                  !std::is_same_v<std::decay_t<U>, T>))
    {
        if (this->initialized_) {
            this->value_ = std::forward<U>(o);
        } else {
            this->construct(std::forward<U>(o));
            this->initialized_ = true;
        }
        
        return *this;
    }

    void reset()
    {
        destruct();
        initialized_ = false;
    }

    constexpr bool has_value() const { return initialized_; }

    constexpr const T & operator*() const & { return value_; }
    constexpr T & operator*() & { return value_; }
    constexpr const T && operator*() const &&
    { 
        return std::forward<T>(value_);
    }

    constexpr T && operator*() && {
        return std::forward<T>(value_);
    }

    constexpr const T * operator->() const { return &value_; }
    constexpr T * operator->() { return &value_; }

private:
    Optional() 
        : empty_(),
          initialized_(false)
    {
        static_assert(std::is_trivially_copy_constructible_v<Optional<T>> ==
            std::is_trivially_copy_constructible_v<T>);

        static_assert(std::is_trivially_move_constructible_v<Optional<T>> ==
            std::is_trivially_move_constructible_v<T>);

        static_assert(std::is_trivially_destructible_v<Optional<T>> ==
            std::is_trivially_destructible_v<T>);
    }

    template <typename... Args>
    constexpr Optional(std::in_place_t, Args && ...args)
        : value_(std::forward<Args>(args)...),
          initialized_(true)
    {
        static_assert(std::is_trivially_copy_constructible_v<Optional<T>> ==
            std::is_trivially_copy_constructible_v<T>);

        static_assert(std::is_trivially_move_constructible_v<Optional<T>> ==
            std::is_trivially_move_constructible_v<T>);

        static_assert(std::is_trivially_destructible_v<Optional<T>> ==
            std::is_trivially_destructible_v<T>);
    }

    void destruct()
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            if (initialized_) {
                value_.~T();
            }
        }
    }

    template <typename... Args>
    constexpr void construct(Args && ...args)
    {
#ifdef MADRONA_GPU_MODE
        new (&value_) T(std::forward<Args>(args)...);
#else
        std::construct_at(&value_, std::forward<Args>(args)...);
#endif
    }

    struct Empty {};
    union {
        Empty empty_;
        T value_;
    };
    bool initialized_;
};

}
