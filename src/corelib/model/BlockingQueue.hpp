#pragma once

#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <chrono>

#include "../util/AlignedVec.hpp"

namespace Core
{
    // ========================================================================
    // BLOCKING QUEUE (ring buffer, thread-safe, power-of-2 optimized)
    // ========================================================================
    template<typename T>
    class BlockingQueue
    {
        // CORRECTION CRITIQUE : L'ordre de déclaration dicte l'ordre d'initialisation.
        // m_capacity et m_mask DOIVENT être déclarés avant m_buffer.
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

        // Force la capacité à la puissance de 2 supérieure pour optimiser le wrap-around
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
            m_count.fetch_add(1, std::memory_order_release);

            lock.unlock();
            m_cv_pop.notify_one();
            return true;
        }

        size_t pop_batch(AlignedVec<T>& out, size_t maxItems,
            std::chrono::microseconds timeout)
        {
            std::unique_lock lock(m_mutex);

            // 1. On attend qu'il y ait AU MOINS un élément ou que la queue ferme.
            // Attendre maxItems bloquerait inutilement le pipeline si la production est lente.
            m_cv_pop.wait_for(lock, timeout, [this] {
                return m_count.load(std::memory_order_relaxed) > 0 || m_closed;
                });

            // 2. Vérification de l'état de fermeture
            size_t currentCount = m_count.load(std::memory_order_relaxed);
            if (m_closed && (currentCount == 0 || m_fastDrain)) {
                return 0;
            }

            // 3. On prend ce qu'il y a de disponible, dans la limite de maxItems
            size_t n = std::min(currentCount, maxItems);
            if (n == 0) return 0;

            // 4. Extraction des données (m_head est protégé par le mutex)
            for (size_t i = 0; i < n; ++i) {
                out.push_back(m_buffer[m_head]);
                m_head = (m_head + 1) & m_mask;
            }

            // 5. Mise à jour atomique du compteur AVANT de libérer le verrou
            m_count.fetch_sub(n, std::memory_order_release);

            lock.unlock();

            // 6. Notification des producteurs (un batch de places s'est libéré)
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