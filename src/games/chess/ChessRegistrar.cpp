#pragma once
#include "../../corelib/bootstrap/GameTypeRegistry.hpp"
#include "ChessEngine.hpp"
#include "ChessRequester.hpp"
#include "ChessRenderer.hpp"
#include "UCIHandler.hpp"

#include "../../corelib/handler/inference/InferenceHandler.hpp"
#include "../../corelib/handler/training/TrainingHandler.hpp"

static GameTypeRegistry<
    ChessTag,
    ChessEngine,
    ChessRequester,
    ChessRenderer,
    //UCIHandler
	InferenceHandler<ChessTag>
> chessResolver("chess");
