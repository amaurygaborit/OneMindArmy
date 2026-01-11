#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <concepts>

template<typename GameTag>
struct ITraits;

template<typename GameTag>
using ObsElemsT = typename ITraits<GameTag>::ObsElems;
template<typename GameTag>
using MetaT = typename ITraits<GameTag>::Meta;
template<typename GameTag>
using ActionT = typename ITraits<GameTag>::Action;

template<typename GameTag>
struct ObsStateT
{
    ObsElemsT<GameTag> elems;
    MetaT<GameTag>     meta;
};

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

template<typename T>
concept ValidGameTraits = requires
{
    { T::kNumPlayers } -> std::convertible_to<size_t>;
    { T::kNumElems } -> std::convertible_to<size_t>;
    { T::kNumMeta } -> std::convertible_to<size_t>;
    { T::kActionSpace } -> std::convertible_to<size_t>;
    { T::kNumPos } -> std::convertible_to<size_t>;
    { T::kMaxValidActions } -> std::convertible_to<size_t>;
    typename T::ObsElems;
    typename T::Meta;
    typename T::Action;
}
&& (T::kNumPlayers > 0 && T::kNumPlayers <= 64)
&& (T::kNumElems > 0)
&& (T::kNumMeta > 0)
&& (T::kActionSpace > 0)
&& (T::kNumPos > 0)
&& (T::kMaxValidActions > 0);

template<typename GameTag>
struct UIntTypes
{
    // Garder le compte réel pour les masques
    static constexpr size_t kRealNumPlayers = ITraits<GameTag>::kNumPlayers;

    // Le +1 est uniquement pour l'Owner ID (0 = None/Public, 1..N = Players)
    static constexpr size_t kNumOwnerIds = ITraits<GameTag>::kNumPlayers + 1;

    static constexpr size_t kNumElems = ITraits<GameTag>::kNumElems;
    static constexpr size_t kNumMeta = ITraits<GameTag>::kNumMeta;
    static constexpr size_t kActionSpace = ITraits<GameTag>::kActionSpace;

    // Correction de kTotalFacts pour utiliser kActionSpace
    static constexpr size_t kTotalFacts = kNumElems + kNumMeta + kActionSpace + 1;

    static constexpr size_t kNumPos = ITraits<GameTag>::kNumPos + 1;
    static constexpr size_t kMaxValidActions = ITraits<GameTag>::kMaxValidActions;

    using FIdx = SelectMinimalUIntT<kTotalFacts>;
    using PIdx = SelectMinimalUIntT<kNumPos>;
    using OIdx = SelectMinimalUIntT<kNumOwnerIds>; // Utilise kNumOwnerIds

    // VMask basé sur le nombre REEL de joueurs
    using VMask = std::conditional_t<(kRealNumPlayers <= 8), uint8_t,
        std::conditional_t<(kRealNumPlayers <= 16), uint16_t,
        std::conditional_t<(kRealNumPlayers <= 32), uint32_t,
        uint64_t>>>;
};

enum class FactType : uint8_t
{
    ELEMENT = 0,
    META = 1,
    ACTION = 2,
    N_FACT_TYPES
};

template<typename GameTag>
struct Fact
{
    using Types = UIntTypes<GameTag>;
    using FIdx = typename Types::FIdx;
    using VMask = typename Types::VMask;
    using PIdx = typename Types::PIdx;
    using OIdx = typename Types::OIdx;

    // Utiliser le vrai nombre de joueurs pour la logique
    static constexpr size_t kNumPlayers = Types::kRealNumPlayers;

    FIdx     factIdx;
    VMask    visibleMask;
    PIdx     posIdx;
    OIdx     ownerIdx;
    FactType typeIdx;

    static constexpr VMask kVisibleToAll = static_cast<VMask>(~static_cast<VMask>(0));
    static constexpr VMask kVisibleToNone = static_cast<VMask>(0);

    constexpr bool isElement() const noexcept { return typeIdx == FactType::ELEMENT; }
    constexpr bool isMeta() const noexcept { return typeIdx == FactType::META; }
    constexpr bool isAction() const noexcept { return typeIdx == FactType::ACTION; }

    static constexpr Fact MakePad(FactType type) noexcept
    {
        Fact f{};
        f.factIdx = static_cast<FIdx>(0);
        f.visibleMask = kVisibleToNone;
        f.posIdx = static_cast<PIdx>(0);
        f.ownerIdx = static_cast<OIdx>(0);
        f.typeIdx = type;
        return f;
    }

    static constexpr Fact MakePublicPad() noexcept
    {
        Fact f{};
        f.factIdx = static_cast<FIdx>(0);
        f.visibleMask = kVisibleToAll;
        f.posIdx = static_cast<PIdx>(0);
        f.ownerIdx = static_cast<OIdx>(0);
        f.typeIdx = FactType::ELEMENT;
        return f;
    }

    static constexpr Fact makePrivateElem(FIdx fIdx, PIdx pIdx, OIdx playerZeroIndexed) noexcept
    {
        Fact f{};
        f.factIdx = static_cast<FIdx>(fIdx + 1);
        f.visibleMask = static_cast<VMask>(VMask{ 1 } << playerZeroIndexed);
        f.posIdx = static_cast<PIdx>(pIdx + 1);
        f.ownerIdx = static_cast<OIdx>(playerZeroIndexed + 1);
        f.typeIdx = FactType::ELEMENT;
        return f;
    }

    static constexpr Fact makePublicElem(FIdx fIdx, PIdx pIdx) noexcept
    {
        Fact f{};
        f.factIdx = static_cast<FIdx>(fIdx + 1);
        f.visibleMask = kVisibleToAll;
        f.posIdx = static_cast<PIdx>(pIdx + 1);
        f.ownerIdx = 0;
        f.typeIdx = FactType::ELEMENT;
        return f;
    }

    static constexpr Fact makeMeta(FIdx fIdx) noexcept
    {
        Fact f{};
        f.factIdx = static_cast<FIdx>(fIdx + 1);
        f.visibleMask = kVisibleToNone;
        f.posIdx = 0;
        f.ownerIdx = 0;
        f.typeIdx = FactType::META;
        return f;
    }

    static constexpr Fact makePrivateAction(FIdx fIdx, PIdx pIdx, OIdx playerZeroIndexed) noexcept
    {
        Fact f{};
        f.factIdx = static_cast<FIdx>(fIdx + 1);
        f.visibleMask = static_cast<VMask>(VMask{ 1 } << playerZeroIndexed);
        f.posIdx = static_cast<PIdx>(pIdx + 1);
        f.ownerIdx = static_cast<OIdx>(playerZeroIndexed + 1);
        f.typeIdx = FactType::ACTION;
        return f;
    }

    static constexpr Fact makePublicAction(FIdx fIdx, PIdx pIdx) noexcept
    {
        Fact f{};
        f.factIdx = static_cast<FIdx>(fIdx + 1);
        f.visibleMask = kVisibleToAll;
        f.posIdx = static_cast<PIdx>(pIdx + 1);
        f.ownerIdx = 0;
        f.typeIdx = FactType::ACTION;
        return f;
    }

    constexpr void setVisibleTo(OIdx playerZeroIndexed) noexcept
    {
        visibleMask |= static_cast<VMask>(VMask{ 1 } << playerZeroIndexed);
    }
    constexpr void setVisibleToAll() noexcept
    {
        visibleMask = kVisibleToAll;
    }
    constexpr void clearVisibleTo(OIdx playerZeroIndexed) noexcept
    {
        visibleMask &= ~static_cast<VMask>(VMask{ 1 } << playerZeroIndexed);
    }
    constexpr void clearVisibleToAll() noexcept
    {
        visibleMask = kVisibleToNone;
    }
};

template<typename GameTag>
struct IdxStateT
{
    std::array<Fact<GameTag>, ITraits<GameTag>::kNumElems> elemFacts;
    std::array<Fact<GameTag>, ITraits<GameTag>::kNumMeta>  metaFacts;

    IdxStateT() noexcept
    {
        elemFacts.fill(Fact<GameTag>::MakePad(FactType::ELEMENT));
        metaFacts.fill(Fact<GameTag>::MakePad(FactType::META));
    }
};

template<typename GameTag>
using IdxActionT = Fact<GameTag>;

template<typename GameTag>
struct IdxStateActionT
{
    IdxStateT<GameTag>  idxState;
    IdxActionT<GameTag> idxAction;

    IdxStateActionT() noexcept
        : idxState(), idxAction(Fact<GameTag>::MakePad(FactType::ACTION))
    {}
};