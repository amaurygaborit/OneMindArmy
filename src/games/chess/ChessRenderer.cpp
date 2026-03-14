#include <iostream>
#include <bitset>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#endif

#include "ChessRenderer.hpp"

namespace Chess
{
    USING_GAME_TYPES(ChessTypes);

    ChessRenderer::ChessRenderer()
    {
#ifdef _WIN32
        // Active l'UTF-8
        SetConsoleOutputCP(CP_UTF8);

        // Active les séquences ANSI sur Windows (couleurs, curseur)
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD dwMode = 0;
        if (hOut != INVALID_HANDLE_VALUE && GetConsoleMode(hOut, &dwMode))
        {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
#endif
    }

    void ChessRenderer::specificSetup(const YAML::Node& config)
    {
        std::cout << "ChessRenderer setup called\n";

        m_renderRawState = Core::loadVal<bool>(config["specific"]["render"], "renderRawState", false, true);
        m_replaceRendering = Core::loadVal<bool>(config["specific"]["render"], "replaceRendering", false, true);
    }

    void ChessRenderer::renderFactState(const State& state) const
    {
        std::cout << "--- State Details ---\n";

        for (const auto& fact : state.all())
        {
            // Traduction de l'ID en nom de pièce ou de méta
            if (fact.type() == FactType::ELEMENT)
            {
                std::cout << "Piece : " << kPiecesSymbol[fact.factId() + 1] << "\n";
            }
            else
            {
                std::cout << "Meta  : " << static_cast<int>(fact.factId()) << "\n";
            }

            // Traduction du Propriétaire
            if (fact.ownerId() == Defs::kNoOwner)
            {
                std::cout << "Owner : None\n";
            }
            else
            {
                std::cout << "Owner : " << kColor[fact.ownerId()] << "\n";
            }

            // Traduction de la Case
            if (fact.pos() == Defs::kNoPos)
            {
                std::cout << "Square: Off-board\n";
            }
            else
            {
                std::cout << "Square: " << kSquaresName[fact.pos()] << "\n";
            }

            std::cout << "Status: " << (fact.exists() ? "Alive" : "Dead")
                << " (Value: " << fact.value() << ")\n";
            std::cout << "-------------------------\n";
        }
        std::cout << std::endl;
    }

    void ChessRenderer::renderState(const State& state) const
    {
        if (!m_baseConfig.renderState)
            return;

        if (m_renderRawState)
            renderFactState(state);

        // --- 1. Conversion FactState -> Grid pour affichage ---
        // On utilise une structure pour stocker le symbole ET le propriétaire de la case
        struct Cell {
            int code = 0;
            uint32_t owner = Defs::kNoOwner;
        };
        Cell boardGrid[64];

        for (const auto& fact : state.elems())
        {
            // Sécurité absolue : la pièce doit exister ET avoir une position valide
            if (fact.exists() && fact.pos() != Defs::kNoPos)
            {
                // factId() va de 0 à 5 (PAWN à KING).
                // +7 permet de pointer directement sur les symboles "pleins" 
                // (les index 7 à 12 dans ton dictionnaire kPiecesSymbol).
                boardGrid[fact.pos()].code = static_cast<int>(fact.factId()) + 7;
                boardGrid[fact.pos()].owner = fact.ownerId();
            }
        }

        // --- 2. Rendu ---
        std::cout << "\033[1mCurrent position:\033[0m" << std::endl << std::endl;

        // Lettres Colonnes (Haut)
        std::cout << "   ";
        for (int file = 0; file < 8; ++file) std::cout << " " << static_cast<char>('A' + file) << " ";
        std::cout << std::endl;

        for (int rank = 7; rank >= 0; --rank)
        {
            std::cout << " " << (rank + 1) << " "; // Numéro Ligne
            for (int file = 0; file < 8; ++file)
            {
                bool darkSquare = ((rank + file) % 2 == 0);
                if (darkSquare) std::cout << "\033[48;5;17m";   // Dark blue
                else            std::cout << "\033[48;5;75m";   // Light blue

                std::cout << "\033[1m"; // Bold

                int code = boardGrid[rank * 8 + file].code;
                uint32_t owner = boardGrid[rank * 8 + file].owner;

                // Couleur du texte dynamique basée sur l'ownerId (et non plus sur l'id de la pièce !)
                if (code > 0)
                {
                    if (owner == WHITE)      std::cout << "\033[97m";       // Blanc
                    else if (owner == BLACK) std::cout << "\033[38;5;16m";  // Noir
                }

                // CORRECTION : Remplacement des \u00A0 par des espaces standard (" ")
                std::cout << " " << kPiecesSymbol[code] << " " << "\033[0m";
            }
            std::cout << " " << (rank + 1) << std::endl;
        }

        // Lettres Colonnes (Bas)
        std::cout << "   ";
        for (int file = 0; file < 8; ++file) std::cout << " " << static_cast<char>('A' + file) << " ";
        std::cout << std::endl;

        int playerIndex = static_cast<int>(state.getMeta(SLOT_TURN).ownerId());

        // Sécurité simple pour l'affichage
        const char* pName = (playerIndex == 0) ? kColor[0] : kColor[1];
        std::cout << pName << " to play." << std::endl;
    }

    void ChessRenderer::renderValidActions(const State& state, std::span<const Action> actionList) const
    {
        if (!m_baseConfig.renderValidActions)
            return;

        std::cout << "Legal moves (" << actionList.size() << "):" << std::endl;
        for (int i = 0; i < actionList.size(); ++i)
        {
            int start = actionList[i].source();
            int dest = actionList[i].dest();
            int promo = static_cast<int>(actionList[i].value());
            std::cout << kSquaresName[start] << kSquaresName[dest] << kPromosLetter[promo] << " ";
            // Petit retour à la ligne cosmétique tous les 8 coups pour éviter les lignes trop longues
            if ((i + 1) % 8 == 0) std::cout << "\n";
        }
        std::cout << std::endl;
    }

    void ChessRenderer::renderActionPlayed(const Action& action, const uint32_t player) const
    {
        if (m_replaceRendering)
        {
            std::cout << "\033[u" << "\033[0J";
            std::cout << "\033[s";
        }

        if (!m_baseConfig.renderActionPlayed || action.factId() == Defs::kPadFact)
            return;

        std::cout << kColor[player] << " played: "
            << kSquaresName[action.source()]
            << kSquaresName[action.dest()]
            << kPromosLetter[static_cast<int>(action.value())] << std::endl;
    }

    void ChessRenderer::renderResult(const GameResult& result) const
    {
        if (!m_baseConfig.renderResult)
            return;

        std::cout << "\n=== End of Game ===\n";

        // 1. Affichage de la cause (La nouveauté architecturale)
        auto reason = static_cast<ChessEndReason>(result.reason);
        switch (reason)
        {
        case ChessEndReason::Checkmate:
            std::cout << "Reason: Checkmate" << std::endl;
            break;
        case ChessEndReason::Stalemate:
            std::cout << "Reason: Stalemate" << std::endl;
            break;
        case ChessEndReason::Repetition:
            std::cout << "Reason: Threefold Repetition" << std::endl;
            break;
        case ChessEndReason::FiftyMoveRule:
            std::cout << "Reason: 50-Move Rule" << std::endl;
            break;
        case ChessEndReason::InsufficientMaterial:
            std::cout << "Reason: Insufficient Material" << std::endl;
            break;
        case ChessEndReason::None:
        default:
            // Si la raison est 0, c'est que la partie a été arrêtée manuellement ou par une limite externe
            std::cout << "Reason: Adjudicated / Limit Reached" << std::endl;
            break;
        }

        // 2. Affichage classique du vainqueur via les scores
        float whiteScore = result[0];
        float blackScore = result[1];

        if (whiteScore > blackScore)
            std::cout << "Outcome: White wins!" << std::endl;
        else if (blackScore > whiteScore)
            std::cout << "Outcome: Black wins!" << std::endl;
        else
            std::cout << "Outcome: Draw!" << std::endl;
    }
}