#pragma once
#include "util/CompilerHints.hpp"
#include "AlignedVec.hpp"
#include "interfaces/IEngine.hpp"
#include "interfaces/IHandler.hpp"
#include "interfaces/IRenderer.hpp"
#include "interfaces/IRequester.hpp"
#include "interfaces/ITraits.hpp"

#include "bootstrap/GameTypeRegistry.hpp"
#include "bootstrap/TypeResolver.hpp"

#include "handler/inference/InferenceHandler.hpp"
#include "handler/training/TrainingHandler.hpp"

#include "model/MCTS.hpp"
#include "model/NeuralNet.cuh"