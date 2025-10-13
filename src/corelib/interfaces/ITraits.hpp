#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

template<typename GameTag>
struct GameTraits;

template<typename GameTag>
using ObsElemsT = typename GameTraits<GameTag>::ObsElems;

template<typename GameTag>
using MetaT = typename GameTraits<GameTag>::Meta;

template<typename GameTag>
using ActionT = typename GameTraits<GameTag>::Action;

template<typename GameTag>
struct ObsStateT
{
    ObsElemsT<GameTag> elems;
    MetaT<GameTag>     meta;
};

/// For the model ///
enum class FactType : uint8_t
{
    ELEMENT,
    META,
    ACTION
};

enum class Privacy : uint8_t
{
    PUBLIC,
    PRIVATE
};

struct Fact
{
    uint16_t factIdx;       // index global (embedding id)
    uint16_t posIdx;        // position id (or kPadPos)
    FactType typeIdx;       // ELEMENT / META / ACTION
    Privacy  privacyIdx;    // PUBLIC / PRIVATE

    static constexpr uint16_t kPadPos = UINT16_MAX;

    // Helpers
    constexpr bool isElement() const noexcept { return typeIdx == FactType::ELEMENT; }
    constexpr bool isMeta()    const noexcept { return typeIdx == FactType::META; }
    constexpr bool isAction()  const noexcept { return typeIdx == FactType::ACTION; }
    constexpr bool isPublic()  const noexcept { return privacyIdx == Privacy::PUBLIC; }

    // Factories
    static constexpr void makePrivateElem(std::uint16_t fIdx, std::uint16_t pIdx, Fact& out) noexcept
    {
        out.factIdx = fIdx;
        out.posIdx = pIdx;
        out.typeIdx = FactType::ELEMENT;
        out.privacyIdx = Privacy::PRIVATE;
    }

    static constexpr void makePublicElem(std::uint16_t fIdx, std::uint16_t pIdx, Fact& out) noexcept
    {
        out.factIdx = fIdx;
        out.posIdx = pIdx;
        out.typeIdx = FactType::ELEMENT;
        out.privacyIdx = Privacy::PUBLIC;
    }

    static constexpr void makeMeta(std::uint16_t fIdx, Fact& out) noexcept
    {
        out.factIdx = fIdx;
        out.posIdx = kPadPos;
        out.typeIdx = FactType::META;
        out.privacyIdx = Privacy::PUBLIC;
    }

    static constexpr void makePublicAction(std::uint16_t fIdx, std::uint16_t pIdx, Fact& out) noexcept
    {
        out.factIdx = fIdx;
        out.posIdx = pIdx;
        out.typeIdx = FactType::ACTION;
        out.privacyIdx = Privacy::PUBLIC;
    }

    static constexpr void makePrivateAction(std::uint16_t fIdx, std::uint16_t pIdx, Fact& out) noexcept
    {
        out.factIdx = fIdx;
        out.posIdx = pIdx;
        out.typeIdx = FactType::ACTION;
        out.privacyIdx = Privacy::PRIVATE;
    }
};

template<typename GameTag>
struct IdxStateT
{
    std::array<Fact, GameTraits<GameTag>::kNumElems> elemFacts;
    std::array<Fact, GameTraits<GameTag>::kNumMeta>  metaFacts;
};

using IdxActionT = Fact;