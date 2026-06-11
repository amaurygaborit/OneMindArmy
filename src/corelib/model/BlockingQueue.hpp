#pragma once

#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <chrono>

#include "../util/AlignedVec.hpp"

namespace Core
{
    // ========================================================================
    // THREAD-SAFE RING BUFFER
    // High-performance blocking queue for producer-consumer pipelines.
    // 
    // Design Intent:
    // Optimizes memory alignment and modulo arithmetic by forcing the capacity 
    // to the nearest power of 2. Uses atomic counters and condition variables 
    // to handle asynchronous burst traffic between CPU threads and GPU inference.
    // ========================================================================
    template<typename T>
    class BlockingQueue
    {
        // CRITICAL: Declaration order dictates initialization order.
        // m_capacity and m_mask MUST be declared before m_buffer to guarantee 
        // correct sizing during constructor member initialization.
        size_t        m_capacity;
        size_t        m_mask;
        AlignedVec<T> m_buffer;

        size_t        m_head = 0;
        size_t        m_tail = 0;
        std::atomic<size_t> m_count{ 0 };

        std::mutex              m_mutex;
        std::condition_variable m_cv_push;
        std::condition_variable m_cv_pop;
        bool m_closed = false;
        bool m_fastDrain = false;

        // Elevates capacity to the next power of 2 to replace slow modulo '%' 
        // operations with fast bitwise '&' during pointer wrap-around.
        static constexpr size_t nextPowerOf2(size_t n) {
            size_t count = 0;
            if (n && !(n & (n - 1))) return n;
            while (n != 0) { n >>= 1; count += 1; }
            return 1ULL << count;
        }

    public:
        explicit BlockingQueue(size_t capacity)
            : m_capacity(nextPowerOf2(capacity))
            , m_mask(m_capacity - 1)
            , m_buffer(reserve_only, m_capacity, m_capacity)
        {
        }

        [[nodiscard]] size_t size() const {
            return m_count.load(std::memory_order_relaxed);
        }

        void close(bool fastDrain = false)
        {
            {
                std::lock_guard lock(m_mutex);
                m_closed = true;
                m_fastDrain = fastDrain;
                if (m_fastDrain) m_count = 0;
            }
            // Awaken all pending threads so they can exit gracefully.
            m_cv_push.notify_all();
            m_cv_pop.notify_all();
        }

        bool push(const T& item)
        {
            std::unique_lock lock(m_mutex);
            m_cv_push.wait(lock, [this] { return m_count < m_capacity || m_closed; });
            if (m_closed) return false;

            m_buffer[m_tail] = item;
            m_tail = (m_tail + 1) & m_mask;

            // Release semantic ensures the written item is visible before the count updates.
            m_count.fetch_add(1, std::memory_order_release);

            lock.unlock();
            m_cv_pop.notify_one();
            return true;
        }

        // Extracts multiple items simultaneously to minimize lock contention.
        // Uses a timeout to prevent pipeline stalling if producers are slow.
        size_t pop_batch(AlignedVec<T>& out, size_t maxItems,
            std::chrono::microseconds timeout)
        {
            std::unique_lock lock(m_mutex);

            // Wait for AT LEAST one item. Waiting for maxItems would deadlock 
            // the pipeline during low-traffic phases.
            m_cv_pop.wait_for(lock, timeout, [this] {
                return m_count.load(std::memory_order_relaxed) > 0 || m_closed;
                });

            size_t currentCount = m_count.load(std::memory_order_relaxed);
            if (m_closed && (currentCount == 0 || m_fastDrain)) {
                return 0;
            }

            size_t n = std::min(currentCount, maxItems);
            if (n == 0) return 0;

            for (size_t i = 0; i < n; ++i) {
                out.push_back(m_buffer[m_head]);
                m_head = (m_head + 1) & m_mask;
            }

            // Atomically decrement BEFORE releasing the lock to maintain consistency.
            m_count.fetch_sub(n, std::memory_order_release);

            lock.unlock();

            // Notify producers that a batch of slots just opened up.
            m_cv_push.notify_all();

            return n;
        }

        bool pop(T& out)
        {
            std::unique_lock lock(m_mutex);
            m_cv_pop.wait(lock, [this] { return m_count > 0 || m_closed; });
            if (m_closed && (m_count == 0 || m_fastDrain)) return false;

            out = m_buffer[m_head];
            m_head = (m_head + 1) & m_mask;
            m_count.fetch_sub(1, std::memory_order_release);

            lock.unlock();
            m_cv_push.notify_one();
            return true;
        }
    };
}