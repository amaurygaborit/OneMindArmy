#pragma once

// ============================================================================
// COMPILER HINTS
// Global macros to forcibly strip overhead in hot-paths (e.g., MCTS node evaluations).
// Because macros ignore C++ namespaces, they must be globally accessible.
// ============================================================================

// --- FORCED INLINING ---
// Overrides the compiler's cost-benefit analysis. Essential for wrapping small 
// tensor conversion utilities without triggering function call stack overhead.
#if defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

// --- BRANCH PREDICTION (CPU Pipeline Optimization) ---
// Instructs the CPU's branch predictor. Crucial for handling deep nested loops
// where bounds checks (e.g., "is game over?") evaluate to false 99% of the time.
// Note: Cannot use C++20 [[likely]] here as it only applies to statements, 
// not inline expressions. MSVC generally handles this optimally without hints.
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
    // Neutral fallback for MSVC/others
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif