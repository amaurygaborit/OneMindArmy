#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <concepts>
#include <cassert>
#include <bit>
#include <span>
#include <iostream>
#include <iomanip>

#include "../util/AlignedVec.hpp"

namespace Core
{
    // ========================================================================
    // VALID GAME TRAITS CONCEPT
    // Compile-time structural contract. 
    // 
    // Design Intent:
    // By enforcing these bounds at compile-time, the framework guarantees 
    // zero dynamic memory allocation (no heap fragmentation) during the hot loop 
    // of MCTS tree traversal. It also ensures the C++ memory layout aligns 
    // perfectly with the Python ONNX neural network architecture.
    // ========================================================================
    template<typename GT>
    concept ValidGameTraits =
        requires
    {
        { GT::kNumElemTypes }      -> std::convertible_to<uint32_t>;
        { GT::kNumMetaTypes }      -> std::convertible_to<uint32_t>;
        { GT::kMaxElems }          -> std::convertible_to<uint32_t>;
        { GT::kMaxMetas }          -> std::convertible_to<uint32_t>;

        { GT::kNumPlayers }        -> std::convertible_to<uint32_t>;
        { GT::kNumPos }            -> std::convertible_to<uint32_t>;
        { GT::kMaxValidActions }   -> std::convertible_to<uint32_t>;
        { GT::kActionSpace }       -> std::convertible_to<uint32_t>;
    }
    && (GT::kNumElemTypes > 0)
        && (GT::kMaxElems > 0)
        && (GT::kNumPlayers > 0)
        && (GT::kNumPos > 0)
        && (GT::kMaxValidActions > 0)
        && (GT::kActionSpace > 0);

    template<ValidGameTraits GT> class GenericZobrist;
    template<ValidGameTraits GT> class PovUtils;

    // ========================================================================
    // MEMORY OPTIMIZATION TRAIT
    // Squeezes entity identifiers down to the absolute minimum byte size required.
    // Drastically reduces the memory footprint of MCTS nodes, improving cache 
    // locality during massive parallel searches.
    // ========================================================================
    template<size_t MaxValue>
    struct SelectMinimalUInt
    {
        using type = std::conditional_t<(MaxValue <= UINT8_MAX), uint8_t,
            std::conditional_t<(MaxValue <= UINT16_MAX), uint16_t,
            std::conditional_t<(MaxValue <= UINT32_MAX), uint32_t,
            uint64_t>>>;
    };

    template<size_t MaxValue>
    using SelectMinimalUIntT = typename SelectMinimalUInt<MaxValue>::type;


    // ========================================================================
    // ADAPTIVE SPATIAL BITSET
    // Universal representation of board geometry. 
    // 
    // Design Intent:
    // Seamlessly defaults to raw hardware primitives (uint8/16/32/64) for small 
    // boards to maximize performance, but safely degrades to std::array for 
    // arbitrarily large state spaces.
    // ========================================================================
    template<size_t NumBits>
    struct BitsetProps
    {
        static constexpr bool IsPrimitive = (NumBits <= 64);

        using PrimitiveType = std::conditional_t<(NumBits <= 8), uint8_t,
            std::conditional_t<(NumBits <= 16), uint16_t,
            std::conditional_t<(NumBits <= 32), uint32_t,
            uint64_t>>>;

        static constexpr size_t kNumWords = (NumBits + 63) / 64;
        using ArrayType = std::array<uint64_t, kNumWords>;
        using StorageType = std::conditional_t<IsPrimitive, PrimitiveType, ArrayType>;
    };

    template<size_t NumBits>
    struct BitsetT
    {
        using Props = BitsetProps<NumBits>;
        using Storage = typename Props::StorageType;

        Storage bits{};

        constexpr void clear() noexcept
        {
            if constexpr (Props::IsPrimitive) bits = 0;
            else                              bits.fill(0);
        }

        constexpr void set(size_t pos) noexcept
        {
            assert(pos < NumBits && "[Bitset] set() out of bounds!");
            if constexpr (Props::IsPrimitive) bits |= (static_cast<Storage>(1) << pos);
            else                              bits[pos / 64] |= (1ULL << (pos % 64));
        }

        constexpr void unset(size_t pos) noexcept
        {
            assert(pos < NumBits && "[Bitset] unset() out of bounds!");
            if constexpr (Props::IsPrimitive) bits &= ~(static_cast<Storage>(1) << pos);
            else                              bits[pos / 64] &= ~(1ULL << (pos % 64));
        }

        [[nodiscard]] constexpr bool test(size_t pos) const noexcept
        {
            assert(pos < NumBits && "[Bitset] test() out of bounds!");
            if constexpr (Props::IsPrimitive) return (bits & (static_cast<Storage>(1) << pos)) != 0;
            else                              return (bits[pos / 64] & (1ULL << (pos % 64))) != 0;
        }

        constexpr void setRange(size_t start, size_t end) noexcept
        {
            for (size_t i = start; i <= end; ++i) set(i);
        }

        constexpr void unsetRange(size_t start, size_t end) noexcept
        {
            for (size_t i = start; i <= end; ++i) unset(i);
        }

        [[nodiscard]] constexpr int popcount() const noexcept
        {
            if constexpr (Props::IsPrimitive) return std::popcount(bits);
            else
            {
                int n = 0;
                for (auto w : bits) n += std::popcount(w);
                return n;
            }
        }

        // Resolves superposition. Returns -1 if the bitset represents a probability 
        // cloud (multiple bits set) or is completely empty.
        [[nodiscard]] constexpr int singleBitIndex() const noexcept
        {
            if constexpr (Props::IsPrimitive)
            {
                return (std::popcount(bits) == 1) ? std::countr_zero(bits) : -1;
            }
            else
            {
                int count = 0, idx = -1;
                for (size_t i = 0; i < Props::kNumWords; ++i)
                {
                    if (bits[i] == 0) continue;
                    count += std::popcount(bits[i]);
                    if (count > 1) return -1;
                    idx = static_cast<int>(i * 64) + std::countr_zero(bits[i]);
                }
                return (count == 1) ? idx : -1;
            }
        }
    };

    // ========================================================================
    // COMPILE-TIME CONSTANT REGISTRY
    // Centralizes all derived logic constraints. Prevents magic numbers 
    // from proliferating through the engine and tensor encoding logic.
    // ========================================================================
    template<ValidGameTraits GT>
    struct GameDefs
    {
        static constexpr uint32_t kNumElemTypes = GT::kNumElemTypes;
        static constexpr uint32_t kNumMetaTypes = GT::kNumMetaTypes;
        static constexpr uint32_t kNumFactTypes = kNumElemTypes + kNumMetaTypes;

        static constexpr uint32_t kMaxElems = GT::kMaxElems;
        static constexpr uint32_t kMaxMetas = GT::kMaxMetas;
        static constexpr uint32_t kMaxFacts = kMaxElems + kMaxMetas;

        static constexpr uint32_t kNumPlayers = GT::kNumPlayers;
        static constexpr uint32_t kNumPos = GT::kNumPos;
        static constexpr uint32_t kMaxValidActions = GT::kMaxValidActions;

        static constexpr uint32_t kMaxHistory = GT::kMaxHistory;
        static constexpr uint32_t kTokenDim = 4 + kNumPos;
        static constexpr uint32_t kNNSequenceLength = kMaxFacts + kMaxHistory;
        static constexpr uint32_t kNNInputSize = kNNSequenceLength * kTokenDim;

        static constexpr uint32_t kActionSpace = GT::kActionSpace;

        // Sentinel bounds indicating inactive or unowned states.
        static constexpr uint32_t kPadFact = kNumFactTypes;
        static constexpr uint32_t kNoOwner = kNumPlayers;
        static constexpr uint32_t kNoPos = kNumPos;

        // Tightly packed ID capacities (Includes +1 padding for sentinels)
        static constexpr uint32_t kFactCapacity = kNumFactTypes + 1;
        static constexpr uint32_t kOwnerCapacity = kNumPlayers + 1;
        static constexpr uint32_t kPosCapacity = kNumPos + 1;

        using FId = SelectMinimalUIntT<kFactCapacity>;
        using OId = SelectMinimalUIntT<kOwnerCapacity>;
        using PId = SelectMinimalUIntT<kPosCapacity>;

        enum class FactType : uint8_t
        {
            ELEMENT = 0,    // Physical board pieces/cards
            META = 1,       // Auxiliary game rules (turn counters, castling rights)
            ACTION = 2,     // Graph edge representing a state transition
            PAD = 3         // Null token for transformer padding
        };

        using LocType = BitsetT<kNumPos>;
    };

    // ========================================================================
    // BASE ENTITY
    // Common data carrier for all semantic objects in the framework.
    // ========================================================================
    template<ValidGameTraits GT>
    class Atom
    {
    public:
        using Defs = GameDefs<GT>;
        using FId = typename Defs::FId;
        using OId = typename Defs::OId;
        using FactType = typename Defs::FactType;

    protected:
        float    m_value = 0.0f;
        FId      m_factId = Defs::kPadFact;
        OId      m_ownerId = Defs::kNoOwner;
        FactType m_type = FactType::PAD;

    public:
        [[nodiscard]] constexpr FId      factId()  const noexcept { return m_factId; }
        [[nodiscard]] constexpr OId      ownerId() const noexcept { return m_ownerId; }
        [[nodiscard]] constexpr float    value()   const noexcept { return m_value; }
        [[nodiscard]] constexpr FactType type()    const noexcept { return m_type; }

        [[nodiscard]] constexpr bool exists() const noexcept { return m_value > 1e-5f; }
    };

    // ========================================================================
    // PERSISTENT BOARD ENTITY
    // Maintains spatial state (using BitsetT to support probability clouds 
    // in imperfect-information environments).
    // ========================================================================
    template<ValidGameTraits GT>
    class Fact : public Atom<GT>
    {
    public:
        using Defs = GameDefs<GT>;
        using FId = typename Defs::FId;
        using OId = typename Defs::OId;
        using PId = typename Defs::PId;
        using LocType = typename Defs::LocType;
        using FactType = typename Defs::FactType;

    private:
        LocType m_location{};

    public:
        constexpr Fact() noexcept { m_location.clear(); }

        void initType(FactType t) noexcept { this->m_type = t; }

        void reset() noexcept
        {
            this->m_value = 0.0f;
            this->m_factId = Defs::kPadFact;
            this->m_ownerId = Defs::kNoOwner;
            m_location.clear();
        }

        void configureElem(uint32_t elemId, uint32_t oId, float val = 1.0f) noexcept
        {
            assert(elemId < Defs::kNumElemTypes);
            this->m_factId = static_cast<FId>(elemId);
            this->m_ownerId = static_cast<OId>(oId);
            this->m_value = val;
        }

        void configureMeta(uint32_t metaId, uint32_t oId, float val = 1.0f) noexcept
        {
            assert(metaId < Defs::kNumMetaTypes);
            this->m_factId = static_cast<FId>(Defs::kNumElemTypes + metaId);
            this->m_ownerId = static_cast<OId>(oId);
            this->m_value = val;
        }

        void configureFact(uint32_t factId, uint32_t oId, float val = 1.0f) noexcept
        {
            assert(factId < Defs::kNumFactTypes);
            this->m_factId = static_cast<FId>(factId);
            this->m_ownerId = static_cast<OId>(oId);
            this->m_value = val;
        }

        void setFactId(uint32_t id) noexcept { this->m_factId = static_cast<FId>(id); }
        void setOwner(uint32_t oId) noexcept { this->m_ownerId = static_cast<OId>(oId); }

        void setValue(float val) noexcept { this->m_value = val; }
        void addValue(float delta) noexcept { this->m_value += delta; }

        // Dirac delta location assignment (Perfect info)
        void setPos(uint32_t loc) noexcept
        {
            assert(loc < Defs::kNumPos);
            this->m_value = 1.0f;
            m_location.clear();
            m_location.set(loc);
        }

        // Probability cloud expansion (Imperfect info)
        void addPossiblePos(uint32_t loc) noexcept
        {
            assert(loc < Defs::kNumPos);
            this->m_value = 1.0f;
            m_location.set(loc);
        }

        void removePossiblePos(uint32_t loc) noexcept
        {
            assert(loc < Defs::kNumPos);
            m_location.unset(loc);
        }

        // Zeroes all spatial and semantic markers, rendering it a ghost token.
        void kill() noexcept
        {
            this->m_value = 0.0f;
            m_location.clear();
        }

        // Collapses the state vector if deterministic, returns kNoPos otherwise.
        [[nodiscard]] uint32_t pos() const noexcept
        {
            if (!this->exists()) return Defs::kNoPos;
            int id = m_location.singleBitIndex();
            return (id >= 0) ? static_cast<uint32_t>(id) : Defs::kNoPos;
        }

        [[nodiscard]] bool isPossiblePos(uint32_t loc) const noexcept
        {
            if (!this->exists()) return false;
            return m_location.test(loc);
        }

        [[nodiscard]] const LocType& rawLocation() const noexcept { return m_location; }

        void copyFrom(const Fact& other) noexcept
        {
            *this = other;
        }
    };

    // ========================================================================
    // STATE TRANSITION EDGE
    // Represents a defined transition between two states. Unlike Facts, actions 
    // are deterministic graph edges and only require simple source/dest IDs.
    // ========================================================================
    template<ValidGameTraits GT>
    class Action : public Atom<GT>
    {
    public:
        using Defs = GameDefs<GT>;
        using FId = typename Defs::FId;
        using OId = typename Defs::OId;
        using PId = typename Defs::PId;
        using FactType = typename Defs::FactType;

    private:
        PId m_source = static_cast<PId>(Defs::kNoPos);
        PId m_dest = static_cast<PId>(Defs::kNoPos);

    public:
        constexpr Action() noexcept = default;

        void reset() noexcept
        {
            this->m_type = FactType::PAD;
            this->m_factId = Defs::kPadFact;
            this->m_ownerId = Defs::kNoOwner;
            this->m_value = 0.0f;
            m_source = static_cast<PId>(Defs::kNoPos);
            m_dest = static_cast<PId>(Defs::kNoPos);
        }

        void configure(uint32_t fId, uint32_t oId, uint32_t src, uint32_t dst, float val = 1.0f) noexcept
        {
            this->m_type = FactType::ACTION;
            this->m_factId = static_cast<FId>(fId);
            this->m_ownerId = static_cast<OId>(oId);
            this->m_value = val;
            m_source = static_cast<PId>(src);
            m_dest = static_cast<PId>(dst);
        }

        void setPos(uint32_t src, uint32_t dst, float val = 1.0f) noexcept
        {
            this->m_type = FactType::ACTION;
            this->m_value = val;
            m_source = static_cast<PId>(src);
            m_dest = static_cast<PId>(dst);
        }

        [[nodiscard]] constexpr uint32_t source() const noexcept { return static_cast<uint32_t>(m_source); }
        [[nodiscard]] constexpr uint32_t dest()   const noexcept { return static_cast<uint32_t>(m_dest); }
        [[nodiscard]] constexpr bool     isValid() const noexcept { return this->m_type == FactType::ACTION; }

        [[nodiscard]] constexpr bool operator==(const Action& other) const noexcept
        {
            return this->factId() == other.factId() &&
                this->ownerId() == other.ownerId() &&
                this->m_source == other.m_source &&
                this->m_dest == other.m_dest;
        }
    };

    // ========================================================================
    // RAII ZOBRIST PROXY
    // Grants mutable access to a Fact while strictly guaranteeing that the 
    // underlying State's Zobrist hash stays perfectly synchronized.
    // XORs the fact out of the hash on creation, and XORs it back on destruction.
    // ========================================================================
    template<ValidGameTraits GT>
    class FactMutator
    {
    private:
        uint64_t& m_stateHash;
        Core::Fact<GT>& m_fact;

    public:
        FactMutator(uint64_t& hashRef, Core::Fact<GT>& f) noexcept
            : m_stateHash(hashRef), m_fact(f)
        {
            m_stateHash ^= Core::GenericZobrist<GT>::getKey(m_fact);
        }

        ~FactMutator() noexcept
        {
            m_stateHash ^= Core::GenericZobrist<GT>::getKey(m_fact);
        }

        FactMutator(const FactMutator&) = delete;
        FactMutator& operator=(const FactMutator&) = delete;

        Core::Fact<GT>* operator->() noexcept { return &m_fact; }
    };

    // ========================================================================
    // ABSOLUTE GAME STATE
    // The definitive source of truth for the engine and the neural network.
    // 
    // Design Intent:
    // Maintains a rigidly partitioned memory layout (Elements vs Metas) to 
    // ensure deterministic, position-anchored encoding for the Transformer model.
    // ========================================================================
    template<ValidGameTraits GT>
    class State
    {
    public:
        using Defs = GameDefs<GT>;
        using FactType = typename Defs::FactType;

    private:
        std::array<Core::Fact<GT>, Defs::kMaxFacts> m_facts;
        uint64_t m_hash = 0;

        friend class PovUtils<GT>;

        [[nodiscard]] Core::Fact<GT>& modifyFactNoHash(uint32_t factIdx) noexcept
        {
            assert(factIdx < Defs::kMaxFacts);
            return m_facts[factIdx];
        }

    public:
        constexpr State() noexcept
        {
            for (uint32_t idx = 0; idx < Defs::kMaxFacts; ++idx) {
                m_facts[idx].initType(idx < Defs::kMaxElems ? FactType::ELEMENT : FactType::META);
            }
        }

        void clear() noexcept
        {
            for (auto& f : m_facts) f.reset();
        }

        [[nodiscard]] constexpr uint64_t hash() const noexcept { return m_hash; }
        [[nodiscard]] std::span<const Core::Fact<GT>> all() const noexcept { return m_facts; }
        [[nodiscard]] std::span<const Core::Fact<GT>> elems() const noexcept { return std::span<const Core::Fact<GT>>{ m_facts }.first(Defs::kMaxElems); }
        [[nodiscard]] std::span<const Core::Fact<GT>> metas() const noexcept { return std::span<const Core::Fact<GT>>{ m_facts }.subspan(Defs::kMaxElems, Defs::kMaxMetas); }

        [[nodiscard]] const Core::Fact<GT>& getElem(uint32_t elemIdx) const noexcept
        {
            assert(elemIdx < Defs::kMaxElems);
            return m_facts[elemIdx];
        }

        [[nodiscard]] const Core::Fact<GT>& getMeta(uint32_t metaIdx) const noexcept
        {
            assert(metaIdx < Defs::kMaxMetas);
            return m_facts[Defs::kMaxElems + metaIdx];
        }

        [[nodiscard]] const Core::Fact<GT>& getFact(uint32_t factIdx) const noexcept
        {
            assert(factIdx < Defs::kMaxFacts);
            return m_facts[factIdx];
        }

        // Returns a FactMutator proxy to ensure Zobrist synchronization
        [[nodiscard]] FactMutator<GT> modifyElem(uint32_t elemIdx) noexcept
        {
            assert(elemIdx < Defs::kMaxElems);
            return FactMutator<GT>{m_hash, m_facts[elemIdx]};
        }

        [[nodiscard]] FactMutator<GT> modifyMeta(uint32_t metaIdx) noexcept
        {
            assert(metaIdx < Defs::kMaxMetas);
            return FactMutator<GT>{m_hash, m_facts[Defs::kMaxElems + metaIdx]};
        }

        [[nodiscard]] FactMutator<GT> modifyFact(uint32_t factIdx) noexcept
        {
            assert(factIdx < Defs::kMaxFacts);
            return FactMutator<GT>{m_hash, m_facts[factIdx]};
        }

        // Restores hash synchronization after bulk raw memory operations (e.g., FEN deserialization).
        void recomputeHash() noexcept
        {
            m_hash = 0;
            for (const auto& f : m_facts) {
                m_hash ^= Core::GenericZobrist<GT>::getKey(f);
            }
        }
    };

    // ========================================================================
    // OUTCOME PAYLOAD
    // Maintains structural compatibility with TensorRT array outputs while 
    // carrying engine-specific termination reason codes.
    // ========================================================================
    template<uint32_t NumPlayers>
    struct GameResult
    {
        std::array<float, NumPlayers * 3> wdl{};
        uint32_t reason = 0;

        constexpr void fill(float val) noexcept {
            wdl.fill(val);
            reason = 0;
        }
    };

    // ========================================================================
    // HEAPLESS VECTOR 
    // Array wrapper that mimics std::vector semantics without dynamic allocation.
    // Eliminates heap fragmentation during high-throughput MCTS rollouts.
    // ========================================================================
    template<typename T, size_t Capacity>
    class alignas(64) StaticVec
    {
    private:
        std::array<T, Capacity> m_data;
        size_t m_size = 0;

    public:
        void clear() noexcept { m_size = 0; }
        void push_back(const T& val) noexcept {
            assert(m_size < Capacity && "StaticVec capacity exceeded!");
            m_data[m_size++] = val;
        }

        [[nodiscard]] size_t size() const noexcept { return m_size; }
        [[nodiscard]] bool empty() const noexcept { return m_size == 0; }

        auto begin() noexcept { return m_data.begin(); }
        auto end() noexcept { return m_data.begin() + m_size; }

        operator std::span<T>() noexcept { return std::span<T>(m_data.data(), m_size); }
        operator std::span<const T>() const noexcept { return std::span<const T>(m_data.data(), m_size); }

        const T& operator[](size_t i) const noexcept { return m_data[i]; }
        T& operator[](size_t i) noexcept { return m_data[i]; }
    };

    // ========================================================================
    // DEBUG & LOGGING
    // ========================================================================
    template<ValidGameTraits GT>
    inline std::ostream& operator<<(std::ostream& os, const Core::Fact<GT>& f)
    {
        using Defs = GameDefs<GT>;
        using FactType = typename Defs::FactType;

        os << "[Fact] "
            << (f.type() == FactType::ELEMENT ? "ELEM" : "META")
            << " | ID: " << std::setw(2) << f.factId();

        os << " | Owner: ";
        if (f.ownerId() == Defs::kNoOwner) os << "NONE";
        else                               os << std::setw(4) << f.ownerId();

        os << " | Pos: ";
        uint32_t position = f.pos();
        if (position == Defs::kNoPos) os << "SUPER/DEAD";
        else                          os << std::setw(10) << position;

        os << " | Val: " << std::fixed << std::setprecision(2) << f.value()
            << (f.exists() ? " (ALIVE)" : " (DEAD) ");

        return os;
    }
}

#include "../util/Zobrist.hpp"

// ============================================================================
// DEPENDENCY INJECTION MACRO
// Erases heavy template boilerplate across the codebase by flooding the local 
// scope with perfectly typed aliases mapped to the active GameTraits configuration.
// ============================================================================
#define USING_GAME_TYPES(GT)                                                   \
    using Defs          = Core::GameDefs<GT>;                                  \
    using Fact          = Core::Fact<GT>;                                      \
    using State         = Core::State<GT>;                                     \
    using Action        = Core::Action<GT>;                                    \
    using ActionList    = Core::StaticVec<Core::Action<GT>, Defs::kMaxValidActions>; \
    using GameResult    = Core::GameResult<Defs::kNumPlayers>;                 \
    template<typename T>                                                       \
    using Vec           = Core::AlignedVec<T>;                                 \
                                                                               \
    using ZobristHasher = Core::GenericZobrist<GT>;                            \
    using PovUtils      = Core::PovUtils<GT>;                                  \
                                                                               \
    using FId           = typename Defs::FId;                                  \
    using OId           = typename Defs::OId;                                  \
    using PId           = typename Defs::PId;                                  \
    using LocType       = typename Defs::LocType;                              \
    using FactType      = typename Defs::FactType;