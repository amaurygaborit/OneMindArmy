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

    m_renderRawState = loadVal<bool>(config["specific"]["render"], "renderRawState", 0, 1);
    m_replaceRendering = loadVal<bool>(config["specific"]["render"], "replaceRendering", 0, 1);
}

void ChessRenderer::dispBoard(uint64_t board) const
{
    for (int rank = 7; rank >= 0; --rank)
    {
        for (int file = 0; file < 8; ++file)
        {
            if (board & (1ULL << (8 * rank + file)))
                std::cout << "1 ";
            else
                std::cout << "0 ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void ChessRenderer::renderRawState(const ObsState& obsState) const
{
    std::cout << "\n=== Raw State ===\n";

    std::cout << "\n--- White Pieces ---\n";
    for (int i = 0; i < 6; ++i)
    {
        std::cout << kPiecesName[i] << std::endl;
        dispBoard(obsState.elems.whiteBB[i]);
    }

    std::cout << "\n--- Black Pieces ---\n";
    for (int i = 0; i < 6; ++i)
    {
        std::cout << kPiecesName[i] << std::endl;
        dispBoard(obsState.elems.blackBB[i]);
    }

    uint64_t enPassantBB = obsState.meta.enPassant;
    std::string enPassantName = enPassantBB ? kSquaresName[enPassantBB] : "-";

    std::cout << "--- Meta Information ---\n";
    std::cout << "Turn: " << kColor[obsState.meta.trait] << "\n";
    std::cout << "Castling rights: " << std::bitset<4>(obsState.meta.castlingRights) << "\n";
    std::cout << "En passant: " << enPassantName << "\n";
    std::cout << "Halfmoves since last irreversible move: " << static_cast<int>(obsState.meta.halfmoveClock) << "\n";
    std::cout << "Total move count: " << static_cast<int>(obsState.meta.fullmoveNumber) << "\n";
    std::cout << "Repetitions: " << static_cast<int>(obsState.meta.repetitions) << "\n";
    std::cout << std::endl;
}

void ChessRenderer::renderState(const ObsState& obsState) const
{
    if (m_renderRawState)
        renderRawState(obsState);

    if (!m_baseConfig.renderState)
        return;

    // Display title in bold
    std::cout << "\033[1mCurrent position:\033[0m" << std::endl << std::endl;

    // Display file indices at the top
    std::cout << "   ";
    for (int file = 0; file < 8; ++file)
    {
        std::cout << " " << static_cast<char>('A' + file) << " ";
    }
    std::cout << std::endl;

    // Display the chessboard
    for (int rank = 7; rank >= 0; --rank)
    {
        std::cout << " " << (rank + 1) << " ";
        for (int file = 0; file < 8; ++file)
        {
            // Determine square color based on parity
            bool darkSquare = ((rank + file) % 2 == 0);
            if (darkSquare)
                std::cout << "\033[48;5;17m";   // Dark blue background
            else
                std::cout << "\033[48;5;75m";   // Light blue background

            // Bold text
            std::cout << "\033[1m";

            // Get square code
            int code = 0; // empty square
            for (int ch = 0; ch < 6; ++ch)
            {
                if (obsState.elems.whiteBB[ch] & (1ULL << (rank * 8 + file)))
                {
                    code = 1 + ch; // 1–6 for white pieces
                    break;
                }
            }
            for (int ch = 0; ch < 6; ++ch)
            {
                if (obsState.elems.blackBB[ch] & (1ULL << (rank * 8 + file)))
                {
                    code = 7 + ch; // 7–12 for black pieces
                    break;
                }
            }

            // Choose text color based on piece color
            if (code >= 1 && code <= 6)
                std::cout << "\033[97m";        // White piece: white text
            else if (code >= 7 && code <= 12)
                std::cout << "\033[38;5;16m";   // Black piece: black text

            // Display the square
            std::cout << "\u00A0" << kPiecesSymbol[code] << "\u00A0" << "\033[0m"; // Reset colors
        }
        // Display rank index on the right
        std::cout << " " << (rank + 1);
        std::cout << std::endl;
    }

    // Display file indices at the bottom
    std::cout << "   ";
    for (int file = 0; file < 8; ++file)
    {
        std::cout << " " << static_cast<char>('A' + file) << " ";
    }
    std::cout << std::endl;
    std::cout << kColor[obsState.meta.trait] << " to play." << std::endl;
}

void ChessRenderer::renderValidActions(const ObsState& obsState) const
{
    if (!m_baseConfig.renderValidActions)
        return;

    AlignedVec<Action> validActions(reserve_only, GT::kMaxValidActions);
    m_engine->getValidActions(obsState, validActions);

    std::cout << "Legal moves (" << validActions.size() << "):" << std::endl;
    for (int i = 0; i < validActions.size(); ++i)
    {
        int start = validActions[i].from();
        int dest = validActions[i].to();
        int promo = validActions[i].promo();
        std::cout << kSquaresName[start] << kSquaresName[dest] << kPromosLetter[promo] << " ";
        // Petit retour à la ligne cosmétique tous les 8 coups pour éviter les lignes trop longues
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }
    std::cout << std::endl;
}

void ChessRenderer::renderActionPlayed(const Action& action, const size_t player) const
{
    if (m_replaceRendering)
    {
        std::cout << "\033[u" << "\033[0J";
        std::cout << "\033[s";
    }

    if (!m_baseConfig.renderActionPlayed || action.data == 0)
        return;

    std::cout << kColor[player] << " played: "
        << kSquaresName[action.from()]
        << kSquaresName[action.to()]
        << kPromosLetter[action.promo()] << std::endl;
}

void ChessRenderer::renderResult(const ObsState& obsState) const
{
    if (!m_baseConfig.renderResult)
        return;

    std::cout << "\n=== End of Game ===\n";
}