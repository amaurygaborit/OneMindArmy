#pragma once

namespace Core
{
    template<typename GT> class InferenceHandler;
    template<typename GT> class TrainingHandler;
}

namespace Tarot
{
    class TarotEngine;
    class TarotRequester;
    class TarotRenderer;

    struct TarotTypes
    {
        using GameTypes = TarotTypes;
        using Engine = TarotEngine;
        using Requester = TarotRequester;
        using Renderer = TarotRenderer;

        static constexpr bool kUseProbabilisticLocation = true;
        static constexpr size_t kNumPlayers = 4;
        static constexpr size_t kNumElems = 78;
        static constexpr size_t kNumMeta = 4;
        static constexpr size_t kNumPos = 8;
        static constexpr size_t kMaxValidActions = 4;
        static constexpr size_t kActionSpace = 666;
    };
}