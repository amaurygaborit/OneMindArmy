#pragma once
#include "../../corelib/interfaces/ITraits.hpp"
#include <cstdint>

struct TarotTag {};

template<>
struct ITraits<TarotTag>
{
    static constexpr uint16_t N_ELEMS = 78;
    static constexpr uint16_t N_META = 6;

    struct ObsElems
    {
        uint8_t hand[18];
        uint8_t returnedCards[6];

        uint8_t table[18];
        uint8_t dog[6];
    };

    struct Meta
    {
        uint8_t trait;
        uint8_t trickCount;
        uint8_t currentTrick[4];
    }

    struct Action
    {
        uint8_t cardId;
        uint8_t fromPosId;
		uint8_t toPosId;

        inline bool operator==(const Action& other) const noexcept
        {
            return (other.cardId == cardId)
                &&(other.fromPosId == fromPosId)
                && (other.toPosId == toPosId);
        };
    };
};