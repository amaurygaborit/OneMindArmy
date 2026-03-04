#pragma once
#include <string>
#include <string_view>
#include <sstream>
#include <cctype>
#include <stdexcept>
#include <algorithm>

#include "ChessTypes.hpp"

namespace Chess
{
    // Exception spécifique pour le parsing
    class InvalidFenException : public std::runtime_error
    {
    public:
        explicit InvalidFenException(const std::string& message)
            : std::runtime_error("Invalid FEN: " + message) {}
    };

    class FenParser
    {
    private:
        USING_GAME_TYPES(ChessTypes);

        static Vec<std::string_view> splitFen(std::string_view fen)
        {
            Vec<std::string_view> result;
            size_t start = 0;
            size_t end = fen.find(' ');

            while (end != std::string_view::npos)
            {
                if (end > start)
                    result.push_back(fen.substr(start, end - start));
                start = end + 1;
                end = fen.find(' ', start);
            }
            if (start < fen.size())
                result.push_back(fen.substr(start));
            return result;
        }

        // --- 1. BOARD (Pièces) ---
        static void parseBoard(std::string_view boardFen, State& outState)
        {
            int rank = 7;
            int file = 0;

            // Compteurs pour allocation linéaire dans le tableau fixe elems[32]
            // Blancs : indices 0 à 15
            // Noirs  : indices 16 à 31
            size_t whiteIdx = 0;
            size_t blackIdx = 16;

            for (char c : boardFen)
            {
                if (c == '/')
                {
                    rank--;
                    file = 0;
                }
                else if (std::isdigit(c))
                {
                    file += (c - '0');
                }
                else
                {
                    // C'est une pièce
                    if (rank < 0 || file >= 8)
                        throw InvalidFenException("Malformed board ranks/files");

                    size_t pos = static_cast<size_t>(rank * 8 + file);
                    Piece pieceType;
                    Player owner;
                    size_t* targetIdx = nullptr;

                    switch (c)
                    {
                    case 'P': pieceType = PAWN;   owner = WHITE; targetIdx = &whiteIdx; break;
                    case 'N': pieceType = KNIGHT; owner = WHITE; targetIdx = &whiteIdx; break;
                    case 'B': pieceType = BISHOP; owner = WHITE; targetIdx = &whiteIdx; break;
                    case 'R': pieceType = ROOK;   owner = WHITE; targetIdx = &whiteIdx; break;
                    case 'Q': pieceType = QUEEN;  owner = WHITE; targetIdx = &whiteIdx; break;
                    case 'K': pieceType = KING;   owner = WHITE; targetIdx = &whiteIdx; break;

                    case 'p': pieceType = PAWN;   owner = BLACK; targetIdx = &blackIdx; break;
                    case 'n': pieceType = KNIGHT; owner = BLACK; targetIdx = &blackIdx; break;
                    case 'b': pieceType = BISHOP; owner = BLACK; targetIdx = &blackIdx; break;
                    case 'r': pieceType = ROOK;   owner = BLACK; targetIdx = &blackIdx; break;
                    case 'q': pieceType = QUEEN;  owner = BLACK; targetIdx = &blackIdx; break;
                    case 'k': pieceType = KING;   owner = BLACK; targetIdx = &blackIdx; break;
                    default: throw InvalidFenException(std::string("Unknown piece char: ") + c);
                    }

                    // Vérification débordement (Max 16 pièces par couleur)
                    // Si le FEN contient > 16 pièces d'une couleur (très rare, promotions massives), 
                    // on ne peut pas le représenter dans notre structure fixe.
                    if ((owner == WHITE && *targetIdx >= 16) ||
                        (owner == BLACK && *targetIdx >= 32))
                    {
                        throw InvalidFenException("Too many pieces for fixed state size");
                    }

                    // Configuration du Fact
                    outState.modifyElem(*targetIdx)->configureElem(pieceType, owner);
                    outState.modifyElem(*targetIdx)->setPos(pos);

                    (*targetIdx)++;
                    file++;
                }
            }
        }

        // --- 2. ACTIVE COLOR ---
        static void parseActiveColor(std::string_view turn, State& outState)
        {
            uint8_t owner = 0;
            if (turn == "w") owner = WHITE;
            else if (turn == "b") owner = BLACK;
            else throw InvalidFenException("Invalid active color");

            outState.modifyMeta(SLOT_TURN)->configureMeta(TURN, owner);
        }

        // --- 3. CASTLING RIGHTS ---
        static void parseCastlingRights(std::string_view castling, State& outState)
        {
            outState.modifyMeta(SLOT_CASTLING_WK)->configureMeta(CASTLING_KINGSIDE, WHITE, 0.0f);
            outState.modifyMeta(SLOT_CASTLING_WQ)->configureMeta(CASTLING_QUEENSIDE, WHITE, 0.0f);
            outState.modifyMeta(SLOT_CASTLING_BK)->configureMeta(CASTLING_KINGSIDE, BLACK, 0.0f);
            outState.modifyMeta(SLOT_CASTLING_BQ)->configureMeta(CASTLING_QUEENSIDE, BLACK, 0.0f);

            if (castling == "-") return;

            for (char c : castling)
            {
                switch (c)
                {
                case 'K': outState.modifyMeta(SLOT_CASTLING_WK)->setValue(1.0f); break;
                case 'Q': outState.modifyMeta(SLOT_CASTLING_WQ)->setValue(1.0f); break;
                case 'k': outState.modifyMeta(SLOT_CASTLING_BK)->setValue(1.0f); break;
                case 'q': outState.modifyMeta(SLOT_CASTLING_BQ)->setValue(1.0f); break;
                default: break;
                }
            }
        }

        // --- 4. EN PASSANT ---
        static void parseEnPassant(std::string_view ep, State& outState, std::string_view turn)
        {
            int squareIdx = static_cast<int>(Defs::kNoPos);

            if (ep != "-")
            {
                if (ep.size() < 2)
                    throw InvalidFenException("Invalid EP square");
                int col = ep[0] - 'a';
                int row = ep[1] - '1';
                if (col < 0 || col > 7 || row < 0 || row > 7)
                    throw InvalidFenException("Invalid EP coords");

                squareIdx = row * 8 + col;
            }

            if (squareIdx == static_cast<int>(Defs::kNoPos))
            {
                outState.modifyMeta(SLOT_EN_PASSANT)->configureMeta(EN_PASSANT, Defs::kNoOwner, 0.0f);
                return;
            }
            uint8_t owner = Defs::kNoOwner;
            if (turn == "w") owner = WHITE;
            else if (turn == "b") owner = BLACK;
            else throw InvalidFenException("Invalid active color");

            outState.modifyMeta(SLOT_EN_PASSANT)->configureMeta(EN_PASSANT, owner);
			outState.modifyMeta(SLOT_EN_PASSANT)->setPos(squareIdx);
        }

        // --- 5. HALFMOVE CLOCK ---
        static void parseHalfmoveClock(std::string_view half, State& outState)
        {
            int val = 0;
            try { val = std::stoi(std::string(half)); }
            catch (...) {}
            outState.modifyMeta(SLOT_HALF_MOVE)->configureMeta(HALF_MOVE, Defs::kNoOwner, static_cast<float>(val));
        }

        // --- 6. FULLMOVE NUMBER ---
        static void parseFullmoveNumber(std::string_view full, State& outState)
        {
            int val = 1;
            try { val = std::stoi(std::string(full)); }
            catch (...) {}
            outState.modifyMeta(SLOT_FULL_MOVE)->configureMeta(FULL_MOVE, Defs::kNoOwner, static_cast<float>(val));
        }

    public:
        [[nodiscard]] static bool tryGetFenState(std::string_view fen, State& outState) noexcept
        {
            try
            {
                getFenState(fen, outState);
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

        /// @brief Parse une chaîne FEN et remplit l'état FactStateT.
        /// @throw InvalidFenException si le format est incorrect.
        static void getFenState(std::string_view fen, State& outState)
        {
            // 1. Reset complet de l'état (tout à OFF_BOARD / 0)
            // Note : le constructeur de FactStateT appelle déjà clear(), mais on assure le coup.
            outState.clear();

            if (fen.empty()) {
                throw InvalidFenException("FEN string is empty");
            }

            // 2. Découpage des champs (Board, Turn, Castling, EnPassant, Half, Full)
            Vec<std::string_view> fields = splitFen(fen);
            if (fields.size() < 4) {
                // Un FEN valide doit avoir au moins Board, Turn, Castling, EnPassant
                throw InvalidFenException("FEN too short (needs at least 4 fields)");
            }

            // 3. Parsing des champs
            parseBoard(fields[0], outState);
            parseActiveColor(fields[1], outState);
            parseCastlingRights(fields[2], outState);
            parseEnPassant(fields[3], outState, fields[1]);

            // Champs optionnels (Clocks)
            if (fields.size() >= 5) parseHalfmoveClock(fields[4], outState);
            if (fields.size() >= 6) parseFullmoveNumber(fields[5], outState);
        }
    };
}