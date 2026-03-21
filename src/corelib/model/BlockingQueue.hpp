#pragma once

#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <chrono>

#include "../util/AlignedVec.hpp"

namespace Core
{
    // ========================================================================
    // BLOCKING QUEUE (ring buffer, thread-safe)
    // ========================================================================
    template<typename T>
    class BlockingQueue
    {
        AlignedVec<T> m_buffer;
        size_t        m_capacity;
        size_t        m_head = 0;
        size_t        m_tail = 0;
        size_t        m_count = 0;

        std::mutex              m_mutex;
        std::condition_variable m_cv_push;
        std::condition_variable m_cv_pop;
        bool m_closed = false;
        bool m_fastDrain = false;

    public:
        explicit BlockingQueue(size_t capacity)
            : m_buffer(reserve_only, capacity, capacity)
            , m_capacity(capacity)
        {
        }

        void close(bool fastDrain = false)
        {
            {
                std::lock_guard lock(m_mutex);
                m_closed = true;
                m_fastDrain = fastDrain;
                if (m_fastDrain) m_count = 0;
            }
            m_cv_push.notify_all();
            m_cv_pop.notify_all();
        }

        bool push(const T& item)
        {
            std::unique_lock lock(m_mutex);
            m_cv_push.wait(lock, [this] { return m_count < m_capacity || m_closed; });
            if (m_closed) return false;
            m_buffer[m_tail] = item;
            m_tail = (m_tail + 1) % m_capacity;
            ++m_count;
            lock.unlock();
            m_cv_pop.notify_one();
            return true;
        }

        size_t pop_batch(AlignedVec<T>& out, size_t maxItems,
            std::chrono::microseconds timeout)
        {
            std::unique_lock lock(m_mutex);
            m_cv_pop.wait_for(lock, timeout,
                [this, maxItems] { return m_count >= maxItems || m_closed; });
            if (m_closed && (m_count == 0 || m_fastDrain)) return 0;
            size_t n = std::min(m_count, maxItems);
            if (n == 0) return 0;
            for (size_t i = 0; i < n; ++i) {
                out.push_back(m_buffer[m_head]);
                m_head = (m_head + 1) % m_capacity;
            }
            m_count -= n;
            lock.unlock();
            m_cv_push.notify_all();
            return n;
        }

        bool pop(T& out)
        {
            std::unique_lock lock(m_mutex);
            m_cv_pop.wait(lock, [this] { return m_count > 0 || m_closed; });
            if (m_closed && (m_count == 0 || m_fastDrain)) return false;
            out = m_buffer[m_head];
            m_head = (m_head + 1) % m_capacity;
            --m_count;
            lock.unlock();
            m_cv_push.notify_one();
            return true;
        }
    };
}