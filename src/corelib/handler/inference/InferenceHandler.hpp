#pragma once
#include <chrono>
#include <iostream>
#include <vector>
#include <optional>
#include "../../interfaces/IHandler.hpp"
#include "../../model/ThreadPool.hpp"

namespace Core
{
    template<ValidGameTraits GT>
    class InferenceHandler : public IHandler<GT>
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        void specificSetup(const YAML::Node& config) override {}

    public:
        InferenceHandler() = default;
        virtual ~InferenceHandler() = default;

        void execute() override
        {
            State currentState;
            Action selectedAction;

            // 1. INITIALISATION DE L'ÉTAT ET DE L'HISTORIQUE RÉEL
            Vec<uint64_t> realHashHistory(reserve_only, 256);

            if (this->m_baseConfig.autoInitialState)
            {
                this->m_engine->getInitialState(0, currentState);
            }
            else
            {
                this->m_requester->requestInitialState(0, currentState);
            }

            // On ajoute le hash de la position de départ
            realHashHistory.push_back(currentState.hash());

            // 2. DÉMARRAGE DES ARBRES EN LEUR DONNANT L'HISTORIQUE RÉEL
            for (size_t p = 0; p < this->m_baseConfig.numAIs; ++p) {
                this->m_treeSearch[p]->startSearch(currentState, realHashHistory);
            }

            // ====================================================================
            // NOUVELLE API : On vérifie si la position initiale est déjà terminale
            // ====================================================================
            std::optional<GameResult> finalOutcome = this->m_engine->getGameResult(currentState, realHashHistory);

            int turnCount = 0;
            double sumTime = 0.0;

            // Boucle principale (continue tant qu'on n'a pas de résultat)
            while (!finalOutcome.has_value())
            {
                turnCount++;

                std::cout << "\n======================================================\n";
                std::cout << " TURN " << turnCount << "\n";
                std::cout << "======================================================\n";

                size_t player = this->m_engine->getCurrentPlayer(currentState);
                this->m_renderer->renderState(currentState);

                // Rendu des actions valides en passant l'historique !
                this->m_renderer->renderValidActions(currentState,
                    this->m_engine->getValidActions(currentState, realHashHistory));

                double turnTimeMs = 0.0;

                // --- PRISE DE DÉCISION ---
                if (player < this->m_baseConfig.numAIs)
                {
                    auto t0 = std::chrono::high_resolution_clock::now();

                    this->m_threadPool->executeTreeSearch(this->m_treeSearch[player].get(), this->m_baseConfig.numSimulations);

                    float temp = (turnCount < 30) ? this->m_baseConfig.temperature : 0.1f;
                    selectedAction = this->m_treeSearch[player]->selectMove(temp);

                    auto t1 = std::chrono::high_resolution_clock::now();
                    turnTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    sumTime += turnTimeMs;
                }
                else
                {
                    bool valid = false;
                    do
                    {
                        selectedAction = this->m_requester->requestAction(currentState);
                        valid = this->m_engine->isValidAction(currentState, realHashHistory, selectedAction);
                    } while (!valid);
                }

                // --- MONDE RÉEL ---
                this->m_engine->applyAction(selectedAction, currentState);

                // MAGIE : Le nouveau hash est stocké pour l'éternité
                realHashHistory.push_back(currentState.hash());

                // --- MONDE IMAGINAIRE (Mise à jour des arbres MCTS) ---
                for (size_t p = 0; p < this->m_baseConfig.numAIs; ++p) {
                    this->m_treeSearch[p]->advanceRoot(selectedAction, currentState);
                }

                this->m_renderer->renderActionPlayed(selectedAction, player);

                if (player < this->m_baseConfig.numAIs) {
                    std::cout << "-> MCTS Time : " << turnTimeMs << " ms | Mean: "
                        << (sumTime / turnCount) << " ms\n";
                }

                for (size_t p = 0; p < this->m_baseConfig.numAIs; ++p) {
                    std::cout << "-> [AI-" << p << "] Memory usage: "
                        << this->m_treeSearch[p]->getMemoryUsage() * 100.0f << " %\n";
                }

                // --- DEBUG & METRICS ---
                uint64_t currentHash = currentState.hash();
                int repetitionCount = 0;
                for (uint64_t h : realHashHistory) {
                    if (h == currentHash) {
                        repetitionCount++;
                    }
                }
                // (Note: Les index 6 et 7 pour HALF_MOVE et FULL_MOVE sont spécifiques à ton ChessEngine, 
                // mais c'est très bien pour le debug si tu sais ce que tu fais).
                std::cout << "\nHalf move: " << currentState.getMeta(6);
                std::cout << "\nFull move: " << currentState.getMeta(7);
                std::cout << "\nRepetition Count: " << repetitionCount << " for hash: 0x" << std::hex << currentHash << std::dec << "\n";

                // ====================================================================
                // NOUVELLE API : Met à jour finalOutcome. La boucle s'arrêtera s'il a une valeur.
                // ====================================================================
                finalOutcome = this->m_engine->getGameResult(currentState, realHashHistory);
            }

            std::cout << "\n======================================================\n";
            std::cout << " GAME OVER\n";
            std::cout << "======================================================\n";
            this->m_renderer->renderState(currentState);
            this->m_renderer->renderResult(finalOutcome.value());
        }
    };
}