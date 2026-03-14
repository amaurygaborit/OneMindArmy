#pragma once
#include "../../corelib/bootstrap/GameTypeRegistry.hpp"
#include "ChessTypes.hpp"
#include "ChessEngine.hpp"
#include "ChessRequester.hpp"
#include "ChessRenderer.hpp"
#include "UCIHandler.hpp"
#include "../../src/corelib/handlers/InferenceHandler.hpp"

static Core::AutoGameRegister<Chess::ChessTypes> chessResolver("chess");