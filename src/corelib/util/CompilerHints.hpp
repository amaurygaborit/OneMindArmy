#pragma once

// ============================================================================
// COMPILER HINTS (Global Macros)
// ============================================================================
// Note: Les macros sont globales, pas de namespace possible.

// --- INLINING FORCÉ ---

#if defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

// --- BRANCH PREDICTION (Optimization) ---
// Note: On n'utilise pas [[likely]] ici car il ne s'applique pas aux expressions.
// Pour MSVC, on laisse l'expression telle quelle (le compilateur est assez malin).

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
    // Fallback neutre (MSVC ou autres)
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif