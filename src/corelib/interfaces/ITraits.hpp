#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <concepts>
#include <algorithm>

// ============================================================================
// 1. METAPROGRAMMING UTILS
// ============================================================================

template<size_t maxValue>
struct SelectMinimalUInt
{
    using type = std::conditional_t<(maxValue <= UINT8_MAX), uint8_t,
        std::conditional_t<(maxValue <= UINT16_MAX), uint16_t,
        std::conditional_t<(maxValue <= UINT32_MAX), uint32_t,
        uint64_t>>>;
};
template<size_t maxValue>
using SelectMinimalUIntT = typename SelectMinimalUInt<maxValue>::type;

// ============================================================================
// 2. ENUMS & CONCEPTS
// ============================================================================

enum class FactType : uint8_t
{
    ELEMENT = 0,
    META = 1,
    ACTION = 2,
    PAD = 3
};

template<typename GameTag>
struct ITraits;

template<typename T>
concept ValidGameTraits = requires
{
    { T::kNumPlayers } -> std::convertible_to<size_t>;
    { T::kNumElems }   -> std::convertible_to<size_t>;
    { T::kNumMeta }    -> std::convertible_to<size_t>;
    { T::kNumPos }     -> std::convertible_to<size_t>;
    { T::kHasHiddenInfo } -> std::convertible_to<bool>;
};

// ============================================================================
// 3. TYPE DEFINITIONS
// ============================================================================

template<typename GameTag>
struct UIntTypes
{
    using Traits = ITraits<GameTag>;

    static constexpr size_t kNumPlayers = Traits::kNumPlayers;
    static constexpr size_t kNumOwners = Traits::kNumPlayers + 1;   // + Neutral/Environment

    static constexpr size_t kNumElems = Traits::kNumElems;
    static constexpr size_t kNumMeta = Traits::kNumMeta;
    static constexpr size_t kNumAbstractActions = Traits::kNumAbstractActions;

    static constexpr size_t kMetaStartId = kNumElems;
    static constexpr size_t kAbstractActionStartId = kNumElems + kNumMeta;
    static constexpr size_t kTotalVocabSize = kNumElems + kNumMeta + kNumAbstractActions + 1;   // + PAD

	static constexpr size_t kTotalPosVocab = Traits::kNumPos + 2;   // + NONE + UNKNOWN

    using FIdx = SelectMinimalUIntT<kTotalVocabSize>;
    using OIdx = SelectMinimalUIntT<kNumOwners>;
    using VMask = std::conditional_t<(kNumPlayers <= 8), uint8_t,
        std::conditional_t<(kNumPlayers <= 16), uint16_t,
        std::conditional_t<(kNumPlayers <= 32), uint32_t,
        uint64_t>>>;
    using PIdx = SelectMinimalUIntT<kTotalPosVocab>;
};

// ============================================================================
// 4. THE UNIVERSAL FACT STRUCTURE
// ============================================================================

template<typename GameTag>
struct Fact
{
    using Types = UIntTypes<GameTag>;
    using Traits = ITraits<GameTag>;

    using FIdx = typename Types::FIdx;
    using OIdx = typename Types::OIdx;
    using VMask = typename Types::VMask;
    using PIdx = typename Types::PIdx;

	// --- CONSTANTES ---
    static constexpr FIdx kMetaStartId = Types::kNumElems;
    static constexpr FIdx kAbstractActionStartId = Types::kAbstractActionStartId;
    static constexpr FIdx kPADFactId = static_cast<FIdx>(Types::kTotalVocabSize - 1);

    static constexpr OIdx kNeutralOwnerId = static_cast<OIdx>(Traits::kNumPlayers);
    static constexpr VMask kVisibleToAll = static_cast<VMask>(~0);

    static constexpr PIdx kNumPos = Traits::kNumPos;
    static constexpr PIdx kPosNone = static_cast<PIdx>(kNumPos);
    static constexpr PIdx kPosUnknown = static_cast<PIdx>(kNumPos + 1);

    static constexpr bool kHidden = Traits::kHasHiddenInfo;
    using PosStorage = std::conditional_t<kHidden, std::array<float, kNumPos>, PIdx>;

    // --- ATTRIBUTS ---
    float      dataValue;
    PosStorage location;
    VMask      visibleMask;
    FIdx       factIdx;
    OIdx       ownerIdx;
    PIdx       origin;
    FactType   typeIdx;

    // --- CONSTRUCTEURS ---
    Fact()
    {
        clear();
    }

    void clear()
    {
        if constexpr (kHidden)
        {
            std::fill(location.begin(), location.end(), 1.0f / static_cast<float>(kNumPos));
        }
        else
        {
            location = kPosNone;
        }
        typeIdx = FactType::PAD;
        factIdx = kPADFactId;
        ownerIdx = kNeutralOwnerId;
        visibleMask = 0;
        origin = kPosNone;
        dataValue = 0.0f;
    }

    // --- HELPERS POSITIONS ---

    void setDeterministicPos(PIdx posIdx)
    {
        if constexpr (kHidden)
        {
            std::fill(location.begin(), location.end(), 0.0f);
            location[posIdx] = 1.0f;
        }
        else
        {
            location = posIdx;
        }
    }
    void setProbabilisticPos(const std::array<float, kNumPos>& probas)
    {
		static_assert(kHidden, "Impossible to use probabilities in Perfect Info!");
        std::copy(probas.begin(), probas.begin() + kNumPos, location.begin());
    }

    inline void setElement(FIdx id, OIdx owner, VMask visibility, PIdx locPos, float val)
    {
        typeIdx = FactType::ELEMENT;
        factIdx = id;
        ownerIdx = owner;
        visibleMask = visibility;
        origin = kPosNone;
        setDeterministicPos(locPos);
        dataValue = val;
    }

    inline void setMeta(FIdx id, OIdx owner, PIdx locPos, float val)
    {
        typeIdx = FactType::META;
        factIdx = id;
        ownerIdx = owner;
        visibleMask = kVisibleToAll;
        origin = kPosNone;
        setDeterministicPos(locPos);
        dataValue = val;
    }

    inline void setAction(FIdx id, OIdx owner, VMask visibility, PIdx originPos, PIdx locPos, float val)
    {
        typeIdx = FactType::ACTION;
        factIdx = id;
        ownerIdx = owner;
        visibleMask = visibility;
        origin = originPos;
        setDeterministicPos(locPos);
        dataValue = val;
    }
};

// ============================================================================
// 5. STATE STRUCTURES
// ============================================================================

template<typename GameTag>
struct FactStateT
{
    std::array<Fact<GameTag>, ITraits<GameTag>::kNumElems> elemFacts;
    std::array<Fact<GameTag>, ITraits<GameTag>::kNumMeta>  metaFacts;

    FactStateT()
    {
        for (auto& f : elemFacts) f.clear();
        for (auto& f : metaFacts) f.clear();
    }
};

template<typename GameTag>
struct FactStateActionT
{
	FactStateT<GameTag> stateFacts;
	Fact<GameTag> actionFact;

    FactStateActionT()
    {
		stateFacts = FactStateT<GameTag>{};
		actionFact.clear();
    }
};