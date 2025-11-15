#pragma once

#if defined(_MSC_VER) && _MSVC_LANG >= 201703L  // Visual Studio avec C++17
#define ALWAYS_INLINE inline __forceinline
#elif defined(__GNUC__) || defined(__clang__)  // GCC ou Clang
#define ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define ALWAYS_INLINE inline  // Fallback pour autres compilateurs
#endif

#ifdef _MSC_VER
// Visual Studio
#if _MSC_VER >= 1926 && _MSVC_LANG >= 202002L
#define LIKELY(x) [[likely]] x
#define UNLIKELY(x) [[unlikely]] x
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif
#elif defined(__GNUC__) || defined(__clang__)
// GCC/Clang
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif
