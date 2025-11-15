#pragma once
#include <vector>
#include <new>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "util/CompilerHints.hpp"

// Aligned allocator
template <typename T, size_t Alignment = 64>
struct AlignedAllocator
{
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;
    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(size_t n)
    {
        if (n == 0) return nullptr;
        if (n > size_t(-1) / sizeof(T))
            throw std::bad_alloc();

        void* p = ::operator new(n * sizeof(T), std::align_val_t(Alignment));
        return static_cast<T*>(p);
    }

    void deallocate(T* p, size_t) noexcept
    {
        ::operator delete(p, std::align_val_t(Alignment));
    }

    template <class U, class... Args>
    void construct(U* p, Args&&... args)
    {
        ::new ((void*)p) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) noexcept
    {
        p->~U();
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

// reserve-only tag
struct reserve_only_tag { explicit constexpr reserve_only_tag() noexcept = default; };
inline constexpr reserve_only_tag reserve_only{};

template <typename T, size_t Alignment = 64>
class AlignedVec : public std::vector<T, AlignedAllocator<T, Alignment>>
{
public:
    using base_t = std::vector<T, AlignedAllocator<T, Alignment>>;
    using typename base_t::value_type;
    using typename base_t::allocator_type;
    using typename base_t::size_type;
    using typename base_t::iterator;
    using typename base_t::const_iterator;

    // inherit base constructors
    using base_t::base_t;

    // default ctor
    AlignedVec() noexcept = default;

    // reserve-only (creates empty vector)
    AlignedVec(reserve_only_tag, size_type reserve_capacity)
    {
        this->reserve(reserve_capacity);
    }

    // reserve-only + pre-size + value initialization
    AlignedVec(reserve_only_tag, size_type reserve_capacity, size_type initial_size, const value_type& value = value_type())
    {
        this->reserve(reserve_capacity);
        this->assign(initial_size, value); // now size() == initial_size
    }

    // copy/move defaulted
    AlignedVec(const AlignedVec&) = default;
    AlignedVec(AlignedVec&&) noexcept = default;
    AlignedVec& operator=(const AlignedVec&) = default;
    AlignedVec& operator=(AlignedVec&&) noexcept = default;

    // destructor defaulted
    ~AlignedVec() noexcept = default;

    // returns the last element by value
    value_type pop_back_value()
    {
        assert(!this->empty() && "pop_back_value() called on empty vector");

        if constexpr (std::is_trivially_copyable_v<value_type> && std::is_trivially_default_constructible_v<value_type>)
        {
            // for trivial types, copy is cheap and zeroing semantics are well defined for POD-like types
            value_type tmp = this->back();
            this->pop_back();
            return tmp;
        }
        else
        {
            value_type tmp = std::move(this->back());
            this->pop_back();
            return tmp;
        }
    }

    // reset elements to zero/default without changing size
    void reset()
    {
        if (this->size() == 0) return;

        if constexpr (std::is_trivially_copyable_v<value_type> && std::is_trivially_default_constructible_v<value_type>)
        {
            std::memset(static_cast<void*>(this->data()), 0, this->size() * sizeof(value_type));
        }
        else
        {
            std::fill_n(this->data(), this->size(), value_type{});
        }
    }
};