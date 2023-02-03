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

        std::construct_at(&value_, std::forward<Args>(args)...);
        initialized_ = true;

        return value_;
    }

    constexpr Optional(const Optional<T> &o)
        requires (std::is_trivially_copy_constructible_v<T>) = default;

    constexpr Optional(const Optional<T> &o)
        requires (std::is_copy_constructible_v<T> &&
                  !std::is_trivially_copy_constructible_v<T>)
    {
        this->initialized = o.initialized;

        if (this->initialized) {
            std::construct_at(&this->value, o.value);
        } 
    }

    constexpr Optional(Optional<T> &&o)
        requires (std::is_trivially_move_constructible_v<T>) = default;

    constexpr Optional(Optional<T> &&o)
        requires (std::is_move_constructible_v<T> &&
                  !std::is_trivially_move_constructible_v<T>)
    {
        this->initialized = o.initialized;

        if (this->initialized) {
            std::construct_at(&this->value, std::forward<T>(o.value));
        } 
    }

    constexpr Optional(T &&o)
        : value_(std::move(o)),
          initialized_(true)
    {}

    constexpr ~Optional()
        requires (!std::is_trivially_destructible_v<T>)
    {
        destruct();
    }

    constexpr ~Optional() = default;

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
            std::construct_at(&this->value_, o.value_);
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
        } else if (o.initialized) {
            std::construct_at(&this->value_, std::forward<T>(o.value_));
            this->initialized_ = true;
        }
        
        return *this;
    }

    constexpr Optional<T> & operator=(T &&o)
    {
        if (this->initialized_) {
            this->value_ = std::forward<T>(o);
        } else {
            std::construct_at(&this->value_, std::forward<T>(o));
            this->initialized_ = true;
        }
        
        return *this;
    }

    constexpr bool has_value() const { return initialized_; }

    constexpr T & operator*() { return value_; }
    constexpr const T & operator*() const { return value_; }

    constexpr T * operator->() { return &value_; }
    constexpr const T * operator->() const { return &value_; }

private:
    Optional() 
        : empty_(),
          initialized_(false)
    {}

    template <typename... Args>
    constexpr Optional(std::in_place_t, Args && ...args)
        : value_(std::forward<Args>(args)...),
          initialized_(true)
    {}

    void destruct()
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            if (initialized_) {
                value_.~T();
            }
        }
    }

    struct Empty {};
    union {
        Empty empty_;
        T value_;
    };
    bool initialized_;

    static_assert(std::is_trivially_copy_constructible_v<Optional<T>> ==
        std::is_trivially_copy_constructible_v<T>);

    static_assert(std::is_trivially_move_constructible_v<Optional<T>> ==
        std::is_trivially_move_constructible_v<T>);

    static_assert(std::is_trivially_destructible_v<Optional<T>> ==
        std::is_trivially_destructible_v<T>);
};

}
