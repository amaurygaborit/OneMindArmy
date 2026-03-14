#pragma once

namespace Tarot
{
    class TarotEngine;
    class TarotRequester;
    class TarotRenderer;

    struct TarotTypes
    {
        static constexpr uint32_t kNumPlayers = 4;
        static constexpr uint32_t kNumElems = 78;
        static constexpr uint32_t kNumMeta = 4;
        static constexpr uint32_t kNumPos = 8;
        static constexpr uint32_t kMaxValidActions = 4;
        static constexpr uint32_t kMaxHistory = 8;
        static constexpr uint32_t kActionSpace = 666;

        using GameTypes = TarotTypes;
        using Engine = TarotEngine;
        using Requester = TarotRequester;
        using Renderer = TarotRenderer;
    };
}