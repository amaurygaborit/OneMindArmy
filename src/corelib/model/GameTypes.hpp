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

// ============================================================================
// GameTypes.hpp — Core Data Model for the OneMindArmy Framework
//
// This file defines the fundamental building blocks shared by every game
// implementation within the framework:
//
//   BitsetT<N>   — Unified bitset (primitive or array-backed, auto-selected).
//                  Serves as the universal spatial representation (Dirac delta 
//                  for perfect info, probability cloud for imperfect info).
//   GameDefs<GT> — Derived constants and index types computed from a GameConfig.
//   Atom<GT>     — Lightweight base entity (index, owner, value, type).
//   Fact<GT>     — Board element or metadata slot (extends Atom with BitsetT).
//   Action<GT>   — A game move (extends Atom with source + destination).
//   State<GT>    — Full game state (arrays of Facts, element + meta sections).
//
// To implement a new game, define a plain struct ("GameConfig") satisfying
// the ValidGameTraits concept and provide the required Engine / Requester /
// Renderer / Handler type aliases.
//
// The macro USING_GAME_TYPES(GT) injects all derived aliases into any class
// that needs them, keeping local code concise and consistent.
// ============================================================================

namespace Core
{
    // ========================================================================
    // VALID GAME TRAITS CONCEPT
    //
    // Any struct passed as the GT template parameter must satisfy this concept.
    // It acts as a compile-time contract.
    //
    // Required interface (all static constexpr):
    //   uint32_t kNumElemTypes      — total physical element types
    //   uint32_t kNumMetaTypes      — number of auxiliary metadata types
    //   uint32_t kMaxElems          — maximum physical elements on board
    //   uint32_t kMaxMetas          — maximum metadata slots
    //   uint32_t kNumPlayers        — number of competing agents (>= 1)
    //   uint32_t kNumPos            — number of distinct board positions
    //   uint32_t kMaxValidActions   — upper bound on legal moves per turn
    //   uint32_t kActionSpace       — flat policy head output dimension
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

    // ========================================================================
    // FORWARD DECLARATION (Required for FactMutator)
    // ========================================================================
    template<ValidGameTraits GT> class GenericZobrist;
    template<ValidGameTraits GT> class PovUtils;

    // ========================================================================
    // UTILITY — Minimal unsigned integer type selection
    // Automatically picks the smallest uint type that can hold [0, maxValue].
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
    // BITSET — Unified bitset with transparent storage selection
    //
    // For N <= 64  : backed by a single primitive uint (8 / 16 / 32 / 64 bit).
    // For N >  64  : backed by a std::array<uint64_t, ceil(N/64)>.
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

        Storage bits{};  // Zero-initialised by default

        // --- Bulk operations ---

        constexpr void clear() noexcept
        {
            if constexpr (Props::IsPrimitive) bits = 0;
            else                              bits.fill(0);
        }

        // --- Single-bit operations ---

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

        // --- Range operations [start, end] inclusive ---

        constexpr void setRange(size_t start, size_t end) noexcept
        {
            for (size_t i = start; i <= end; ++i) set(i);
        }

        constexpr void unsetRange(size_t start, size_t end) noexcept
        {
            for (size_t i = start; i <= end; ++i) unset(i);
        }

        // --- Introspection ---

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

        // Returns the index of the single set bit, or -1 if zero or multiple bits are set.
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
    // GAME DEFS — Derived constants and index types
    //
    // GameDefs<GT> is the single source of truth for everything computed from
    // a game config. Import it once via USING_GAME_TYPES(GT).
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

        // ====================================================================
        // SENTINEL VALUES (One-past-the-end indices)
        // Used to denote dead, unowned, or off-board entities.
        // ====================================================================
        static constexpr uint32_t kPadFact = kNumFactTypes;
        static constexpr uint32_t kNoOwner = kNumPlayers;
        static constexpr uint32_t kNoPos = kNumPos;

        // ====================================================================
        // COMPACT INDEX TYPES
        // ====================================================================
        static constexpr uint32_t kFactCapacity = kNumFactTypes + 1;
        static constexpr uint32_t kOwnerCapacity = kNumPlayers + 1;
        static constexpr uint32_t kPosCapacity = kNumPos + 1; // +1 to support kNoPos sentinels

        using FId = SelectMinimalUIntT<kFactCapacity>;
        using OId = SelectMinimalUIntT<kOwnerCapacity>;
        using PId = SelectMinimalUIntT<kPosCapacity>;

        enum class FactType : uint8_t
        {
            ELEMENT = 0,    // Physical board element (piece, card, token, …)
            META = 1,       // Auxiliary game rule / score slot (turn, castling, …)
            ACTION = 2,     // A legal move (used in Action arrays)
            PAD = 3         // Empty / unused slot
        };

        // ====================================================================
        // LOCATION STORAGE (Universal Spatial Representation)
        // ====================================================================
        using LocType = BitsetT<kNumPos>;
    };

    // ========================================================================
    // ATOM — Base entity
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

        // Returns true when the atom carries a meaningful value (not dead / empty).
        [[nodiscard]] constexpr bool exists() const noexcept { return m_value > 1e-5f; }
    };

    // ========================================================================
    // FACT — Board element or metadata slot
    //
    // A Fact represents a trackable entity on the board or an auxiliary metadata slot.
    // Its location is always stored as a BitsetT, bridging the gap between 
    // perfect information (Dirac delta) and imperfect information (probability cloud).
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
        LocType m_location{};   // Universal spatial mask

    public:
        constexpr Fact() noexcept { m_location.clear(); }

        // --- Lifecycle ---
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

        // --- Identity Modifiers ---
        void setFactId(uint32_t id) noexcept { this->m_factId = static_cast<FId>(id); }
        void setOwner(uint32_t oId) noexcept { this->m_ownerId = static_cast<OId>(oId); }

        // --- Value Modifiers ---
        void setValue(float val) noexcept { this->m_value = val; }
        void addValue(float delta) noexcept { this->m_value += delta; }

        // --- Spatial Modifiers ---

        // Sets the entity to exactly one position (Perfect Info Dirac).
        void setPos(uint32_t loc) noexcept
        {
            assert(loc < Defs::kNumPos);
            this->m_value = 1.0f;
            m_location.clear();
            m_location.set(loc);
        }

        // Adds a possible location to the belief state (Imperfect Info).
        void addPossiblePos(uint32_t loc) noexcept
        {
            assert(loc < Defs::kNumPos);
            this->m_value = 1.0f;
            m_location.set(loc);
        }

        // Removes a possible location.
        void removePossiblePos(uint32_t loc) noexcept
        {
            assert(loc < Defs::kNumPos);
            m_location.unset(loc);
        }

        // Kills the entity entirely, clearing its spatial presence and semantic value.
        void kill() noexcept
        {
            this->m_value = 0.0f;
            m_location.clear();
        }

        // --- Spatial Queries ---

        // Returns the position if the entity is collapsed to a single square,
        // or Defs::kNoPos if the entity is dead or in superposition.
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

        // --- Data Transfer ---
        void copyFrom(const Fact& other) noexcept
        {
            *this = other;
        }
    };

    // ========================================================================
    // ACTION — A game move
    //
    // Unlike Facts which maintain spatial superposition, Actions represent 
    // a definite edge traversing the graph, hence they use primitive indices.
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

        // --- Accessors ---
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
    // FACT MUTATOR (Proxy Pattern for Zobrist Synchronisation)
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
    // STATE — Complete game state snapshot
    // ========================================================================

    template<ValidGameTraits GT>
    class State
    {
    public:
        using Defs = GameDefs<GT>;
        using FactType = typename Defs::FactType;

    private:
        // By explicitly using Core::Fact<GT>, we avoid the shadowed name resolution warning.
        std::array<Core::Fact<GT>, Defs::kMaxFacts> m_facts;
        uint64_t m_hash = 0;

        // ====================================================================
        // SECURE SECTOR : Strictly reserved for PovUtils access
        // ====================================================================
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

        // --- Read-Only Views ---
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

        // --- Mutators ---
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

        // Recomputes the entire Zobrist hash from scratch. Must be called after 
        // batch initialisation (e.g., FEN parsing) that bypassed FactMutator.
        void recomputeHash() noexcept
        {
            m_hash = 0;
            for (const auto& f : m_facts) {
                m_hash ^= Core::GenericZobrist<GT>::getKey(f);
            }
        }
    };

    // ========================================================================
    // GAME RESULT — Stores scores and the reason for game termination
    // ========================================================================
    template<uint32_t NumPlayers>
    struct GameResult
    {
        std::array<float, NumPlayers * 3> wdl{};
        uint32_t reason = 0; // Game-specific termination code (0 = ongoing/none)

        // Maintains compatibility with std::array API used by Event::reset()
        constexpr void fill(float val) noexcept {
            wdl.fill(val);
            reason = 0;
        }
    };

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
// INJECTION MACRO
// Ensures downstream classes can pull core definitions into their local scope
// without constant namespace typing.
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