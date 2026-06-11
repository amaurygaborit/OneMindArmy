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

namespace Core
{
    // ========================================================================
    // ALIGNED ALLOCATOR
    // Overrides standard allocator routines to enforce strict memory alignment 
    // (default 64-byte / Cache Line). 
    //
    // Design Intent:
    // Prevents false sharing across threads in the ThreadPool and perfectly 
    // aligns continuous arrays (like tensors) for zero-copy DMA transfers 
    // over the PCIe bus to the GPU.
    // ========================================================================
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

    // Tag dispatch for optimized constructor pathways
    struct reserve_only_tag { explicit constexpr reserve_only_tag() noexcept = default; };
    inline constexpr reserve_only_tag reserve_only{};

    // ========================================================================
    // ALIGNED VECTOR
    // A drop-in replacement for std::vector that guarantees underlying 
    // buffer alignment and provides performance-focused extensions.
    // ========================================================================
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

        using base_t::base_t;

        AlignedVec() noexcept = default;

        // Bypasses the default O(N) zero-initialization of std::vector when 
        // allocating large pre-sized buffers (like MCTS node arrays).
        AlignedVec(reserve_only_tag, size_type reserve_capacity)
        {
            this->reserve(reserve_capacity);
        }

        AlignedVec(reserve_only_tag, size_type reserve_capacity, size_type initial_size, const value_type& value = value_type())
        {
            this->reserve(reserve_capacity);
            this->assign(initial_size, value);
        }

        AlignedVec(const AlignedVec&) = default;
        AlignedVec(AlignedVec&&) noexcept = default;
        AlignedVec& operator=(const AlignedVec&) = default;
        AlignedVec& operator=(AlignedVec&&) noexcept = default;
        ~AlignedVec() noexcept = default;

        value_type pop_back_value()
        {
            assert(!this->empty() && "pop_back_value() called on empty vector");

            // Avoids costly move constructors for trivial POD types
            if constexpr (std::is_trivially_copyable_v<value_type> && std::is_trivially_default_constructible_v<value_type>)
            {
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

        // Hardware-accelerated buffer wipe. Much faster than loop assignment 
        // or calling .clear() + .resize() for recyclable arrays.
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
}