#pragma once
#include <atomic>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_InterlockedExchangeAdd8)
#pragma intrinsic(_InterlockedExchangeAdd16)
#pragma intrinsic(_InterlockedExchangeAdd)
#pragma intrinsic(_InterlockedExchangeAdd64)
#pragma intrinsic(_InterlockedExchange8)
#pragma intrinsic(_InterlockedExchange16)
#pragma intrinsic(_InterlockedExchange)
#pragma intrinsic(_InterlockedExchange64)
#pragma intrinsic(_InterlockedCompareExchange8)
#pragma intrinsic(_InterlockedCompareExchange16)
#pragma intrinsic(_InterlockedCompareExchange)
#pragma intrinsic(_InterlockedCompareExchange64)
#endif

namespace AtomicOps
{
    // Load operations
    template<typename T>
    inline T load(const T* ptr, std::memory_order order = std::memory_order_acquire)
    {
#if defined(_MSC_VER)
        if constexpr (sizeof(T) == 1)
        {
            return static_cast<T>(_InterlockedExchangeAdd8(
                reinterpret_cast<volatile char*>(const_cast<T*>(ptr)), 0));
        }
        else if constexpr (sizeof(T) == 2)
        {
            return static_cast<T>(_InterlockedExchangeAdd16(
                reinterpret_cast<volatile short*>(const_cast<T*>(ptr)), 0));
        }
        else if constexpr (sizeof(T) == 4)
        {
            return static_cast<T>(_InterlockedExchangeAdd(
                reinterpret_cast<volatile long*>(const_cast<T*>(ptr)), 0));
        }
        else if constexpr (sizeof(T) == 8)
        {
            return static_cast<T>(_InterlockedExchangeAdd64(
                reinterpret_cast<volatile long long*>(const_cast<T*>(ptr)), 0));
        }
#else
        return __atomic_load_n(ptr, static_cast<int>(order));
#endif
    }

    // Store operations
    template<typename T>
    inline void store(T* ptr, T value, std::memory_order order = std::memory_order_release)
    {
#if defined(_MSC_VER)
        if constexpr (sizeof(T) == 1)
        {
            _InterlockedExchange8(reinterpret_cast<volatile char*>(ptr), static_cast<char>(value));
        }
        else if constexpr (sizeof(T) == 2)
        {
            _InterlockedExchange16(reinterpret_cast<volatile short*>(ptr), static_cast<short>(value));
        }
        else if constexpr (sizeof(T) == 4)
        {
            _InterlockedExchange(reinterpret_cast<volatile long*>(ptr), static_cast<long>(value));
        }
        else if constexpr (sizeof(T) == 8)
        {
            _InterlockedExchange64(reinterpret_cast<volatile long long*>(ptr), static_cast<long long>(value));
        }
#else
        __atomic_store_n(ptr, value, static_cast<int>(order));
#endif
    }

    // Fetch-add operations
    template<typename T>
    inline T fetch_add(T* ptr, T value, std::memory_order order = std::memory_order_acq_rel)
    {
#if defined(_MSC_VER)
        if constexpr (sizeof(T) == 1)
        {
            return static_cast<T>(_InterlockedExchangeAdd8(
                reinterpret_cast<volatile char*>(ptr), static_cast<char>(value)));
        }
        else if constexpr (sizeof(T) == 2)
        {
            return static_cast<T>(_InterlockedExchangeAdd16(
                reinterpret_cast<volatile short*>(ptr), static_cast<short>(value)));
        }
        else if constexpr (sizeof(T) == 4)
        {
            return static_cast<T>(_InterlockedExchangeAdd(
                reinterpret_cast<volatile long*>(ptr), static_cast<long>(value)));
        }
        else if constexpr (sizeof(T) == 8)
        {
            return static_cast<T>(_InterlockedExchangeAdd64(
                reinterpret_cast<volatile long long*>(ptr), static_cast<long long>(value)));
        }
#else
        return __atomic_fetch_add(ptr, value, static_cast<int>(order));
#endif
    }

    // Compare-exchange operations
    template<typename T>
    inline bool compare_exchange(T* ptr, T* expected, T desired,
        std::memory_order success = std::memory_order_acq_rel,
        std::memory_order failure = std::memory_order_acquire)
    {
#if defined(_MSC_VER)
        if constexpr (sizeof(T) == 1)
        {
            char old = static_cast<char>(*expected);
            char result = _InterlockedCompareExchange8(
                reinterpret_cast<volatile char*>(ptr),
                static_cast<char>(desired), old);
            if (result == old) return true;
            *expected = static_cast<T>(result);
            return false;
        }
        else if constexpr (sizeof(T) == 2)
        {
            short old = static_cast<short>(*expected);
            short result = _InterlockedCompareExchange16(
                reinterpret_cast<volatile short*>(ptr),
                static_cast<short>(desired), old);
            if (result == old) return true;
            *expected = static_cast<T>(result);
            return false;
        }
        else if constexpr (sizeof(T) == 4)
        {
            long old = static_cast<long>(*expected);
            long result = _InterlockedCompareExchange(
                reinterpret_cast<volatile long*>(ptr),
                static_cast<long>(desired), old);
            if (result == old) return true;
            *expected = static_cast<T>(result);
            return false;
        }
        else if constexpr (sizeof(T) == 8)
        {
            long long old = static_cast<long long>(*expected);
            long long result = _InterlockedCompareExchange64(
                reinterpret_cast<volatile long long*>(ptr),
                static_cast<long long>(desired), old);
            if (result == old) return true;
            *expected = static_cast<T>(result);
            return false;
        }
#else
        return __atomic_compare_exchange_n(ptr, expected, desired, false,
            static_cast<int>(success), static_cast<int>(failure));
#endif
    }

    // Float specializations using memcpy for type-punning
    inline float load(const float* ptr, std::memory_order order = std::memory_order_acquire)
    {
        uint32_t temp;
#if defined(_MSC_VER)
        temp = static_cast<uint32_t>(_InterlockedExchangeAdd(
            reinterpret_cast<volatile long*>(const_cast<float*>(ptr)), 0));
#else
        __atomic_load(reinterpret_cast<const uint32_t*>(ptr), &temp, static_cast<int>(order));
#endif
        float result;
        std::memcpy(&result, &temp, sizeof(float));
        return result;
    }

    inline void store(float* ptr, float value, std::memory_order order = std::memory_order_release)
    {
        uint32_t temp;
        std::memcpy(&temp, &value, sizeof(float));
#if defined(_MSC_VER)
        _InterlockedExchange(reinterpret_cast<volatile long*>(ptr), static_cast<long>(temp));
#else
        __atomic_store(reinterpret_cast<uint32_t*>(ptr), &temp, static_cast<int>(order));
#endif
    }

    inline bool compare_exchange(float* ptr, float* expected, float desired,
        std::memory_order success = std::memory_order_acq_rel,
        std::memory_order failure = std::memory_order_acquire)
    {
        uint32_t exp_int, des_int;
        std::memcpy(&exp_int, expected, sizeof(float));
        std::memcpy(&des_int, &desired, sizeof(float));

#if defined(_MSC_VER)
        long old = static_cast<long>(exp_int);
        long result = _InterlockedCompareExchange(reinterpret_cast<volatile long*>(ptr),
            static_cast<long>(des_int), old);
        if (result == old) return true;
        uint32_t result_uint = static_cast<uint32_t>(result);
        std::memcpy(expected, &result_uint, sizeof(float));
        return false;
#else
        bool success_flag = __atomic_compare_exchange_n(
            reinterpret_cast<uint32_t*>(ptr), &exp_int, des_int, false,
            static_cast<int>(success), static_cast<int>(failure));
        if (!success_flag)
        {
            std::memcpy(expected, &exp_int, sizeof(float));
        }
        return success_flag;
#endif
    }
}