#pragma once
#include "ChessTraits.hpp"
#include <string>
#include <string_view>
#include <optional>
#include <cctype>
#include <stdexcept>

/// @brief Exception levée lors d'un parsing FEN invalide
class InvalidFenException : public std::runtime_error {
public:
    explicit InvalidFenException(const std::string& message)
        : std::runtime_error("Invalid FEN: " + message) {}
};

/// @brief Parseur FEN (Forsyth-Edwards Notation) pour les échecs
class FenParser {
public:
    /// @brief Parse une notation FEN et remplit l'état observable
    /// @param fen La chaîne FEN à parser
    /// @param out L'état de sortie à remplir
    /// @return true si le parsing a réussi, false sinon
    [[nodiscard]] static bool tryGetFenState(std::string_view fen, ObsStateT<ChessTag>& out) noexcept {
        try {
            getFenState(fen, out);
            return true;
        }
        catch (...) {
            return false;
        }
    }

    /// @brief Parse une notation FEN et remplit l'état observable (lance une exception si invalide)
    /// @param fen La chaîne FEN à parser
    /// @param out L'état de sortie à remplir
    /// @throws InvalidFenException si le FEN est invalide
    static void getFenState(std::string_view fen, ObsStateT<ChessTag>& out) {
        // Reset the state
        out = ObsStateT<ChessTag>{};

        if (fen.empty()) {
            throw InvalidFenException("FEN string is empty");
        }

        // Split FEN into fields
        auto fields = splitFen(fen);
        if (fields.size() < 2) {
            throw InvalidFenException("FEN must have at least 2 fields (position and turn)");
        }
        if (fields.size() > 6) {
            throw InvalidFenException("FEN has too many fields (max 6)");
        }

        // Parse each field
        parsePiecePosition(fields[0], out);
        parseActiveColor(fields[1], out);

        if (fields.size() > 2) parseCastlingRights(fields[2], out);
        if (fields.size() > 3) parseEnPassantSquare(fields[3], out);
        if (fields.size() > 4) parseHalfmoveClock(fields[4], out);
        if (fields.size() > 5) parseFullmoveNumber(fields[5], out);
    }

private:
    /// @brief Divise la chaîne FEN en champs
    [[nodiscard]] static std::vector<std::string_view> splitFen(std::string_view fen) {
        std::vector<std::string_view> fields;
        std::size_t start = 0;

        for (std::size_t i = 0; i < fen.size(); ++i) {
            if (fen[i] == ' ') {
                if (i > start) {
                    fields.push_back(fen.substr(start, i - start));
                }
                start = i + 1;
            }
        }

        // Add last field
        if (start < fen.size()) {
            fields.push_back(fen.substr(start));
        }

        return fields;
    }

    /// @brief Parse la position des pièces (champ 1)
    static void parsePiecePosition(std::string_view position, ObsStateT<ChessTag>& out) {
        int file = 0;  // Column (0-7)
        int rank = 7;  // Row (7-0, from top to bottom)
        int rankCount = 0;

        for (char c : position) {
            if (c == '/') {
                if (file != 8) {
                    throw InvalidFenException("Rank does not have 8 squares");
                }
                rank--;
                file = 0;
                rankCount++;
                if (rank < -1) {
                    throw InvalidFenException("Too many ranks (max 8)");
                }
                continue;
            }

            if (std::isdigit(c)) {
                int emptySquares = c - '0';
                if (emptySquares < 1 || emptySquares > 8) {
                    throw InvalidFenException("Invalid empty square count");
                }
                file += emptySquares;
            }
            else {
                if (file >= 8) {
                    throw InvalidFenException("Too many files in rank (max 8)");
                }

                std::uint64_t bitPosition = 1ULL << (8 * rank + file);

                switch (c) {
                case 'P': out.elems.whiteBB[0] |= bitPosition; break;
                case 'N': out.elems.whiteBB[1] |= bitPosition; break;
                case 'B': out.elems.whiteBB[2] |= bitPosition; break;
                case 'R': out.elems.whiteBB[3] |= bitPosition; break;
                case 'Q': out.elems.whiteBB[4] |= bitPosition; break;
                case 'K': out.elems.whiteBB[5] |= bitPosition; break;
                case 'p': out.elems.blackBB[0] |= bitPosition; break;
                case 'n': out.elems.blackBB[1] |= bitPosition; break;
                case 'b': out.elems.blackBB[2] |= bitPosition; break;
                case 'r': out.elems.blackBB[3] |= bitPosition; break;
                case 'q': out.elems.blackBB[4] |= bitPosition; break;
                case 'k': out.elems.blackBB[5] |= bitPosition; break;
                default:
                    throw InvalidFenException(std::string("Invalid piece character: ") + c);
                }
                file++;
            }
        }

        if (rankCount != 7) {
            throw InvalidFenException("Position must have exactly 8 ranks");
        }
        if (file != 8) {
            throw InvalidFenException("Last rank does not have 8 squares");
        }
    }

    /// @brief Parse le joueur actif (champ 2)
    static void parseActiveColor(std::string_view color, ObsStateT<ChessTag>& out) {
        if (color.size() != 1) {
            throw InvalidFenException("Active color must be 'w' or 'b'");
        }

        if (color[0] == 'w') {
            out.meta.trait = 0;
        }
        else if (color[0] == 'b') {
            out.meta.trait = 1;
        }
        else {
            throw InvalidFenException(std::string("Invalid active color: ") + std::string(color));
        }
    }

    /// @brief Parse les droits de roque (champ 3)
    static void parseCastlingRights(std::string_view castling, ObsStateT<ChessTag>& out) {
        if (castling.empty()) {
            throw InvalidFenException("Castling rights field is empty");
        }

        if (castling == "-") {
            out.meta.castlingRights = 0;
            return;
        }

        std::uint8_t rights = 0;
        bool hasK = false, hasQ = false, hask = false, hasq = false;

        for (char c : castling) {
            switch (c) {
            case 'K':
                if (hasK) throw InvalidFenException("Duplicate 'K' in castling rights");
                rights |= 1;
                hasK = true;
                break;
            case 'Q':
                if (hasQ) throw InvalidFenException("Duplicate 'Q' in castling rights");
                rights |= 2;
                hasQ = true;
                break;
            case 'k':
                if (hask) throw InvalidFenException("Duplicate 'k' in castling rights");
                rights |= 4;
                hask = true;
                break;
            case 'q':
                if (hasq) throw InvalidFenException("Duplicate 'q' in castling rights");
                rights |= 8;
                hasq = true;
                break;
            default:
                throw InvalidFenException(std::string("Invalid castling character: ") + c);
            }
        }

        out.meta.castlingRights = rights;
    }

    /// @brief Parse la case en passant (champ 4)
    static void parseEnPassantSquare(std::string_view enPassant, ObsStateT<ChessTag>& out) {
        if (enPassant.empty()) {
            throw InvalidFenException("En passant field is empty");
        }

        if (enPassant == "-") {
            out.meta.enPassant = 0xFF; // Convention: 0xFF = no en passant
            return;
        }

        if (enPassant.size() != 2) {
            throw InvalidFenException("En passant square must be 2 characters (e.g., 'e3')");
        }

        char fileChar = enPassant[0];
        char rankChar = enPassant[1];

        if (fileChar < 'a' || fileChar > 'h') {
            throw InvalidFenException("En passant file must be between 'a' and 'h'");
        }
        if (rankChar < '1' || rankChar > '8') {
            throw InvalidFenException("En passant rank must be between '1' and '8'");
        }

        int file = fileChar - 'a';  // 0-7
        int rank = rankChar - '1';  // 0-7

        // En passant is only valid on rank 3 (index 2) for white or rank 6 (index 5) for black
        if (rank != 2 && rank != 5) {
            throw InvalidFenException("En passant square must be on rank 3 or rank 6");
        }

        out.meta.enPassant = static_cast<std::uint8_t>(rank * 8 + file);
    }

    /// @brief Parse le compteur de demi-coups (champ 5)
    static void parseHalfmoveClock(std::string_view halfmove, ObsStateT<ChessTag>& out) {
        if (halfmove.empty()) {
            throw InvalidFenException("Halfmove clock field is empty");
        }

        int value = 0;
        for (char c : halfmove) {
            if (!std::isdigit(c)) {
                throw InvalidFenException("Halfmove clock must be a number");
            }
            value = value * 10 + (c - '0');
            if (value > 255) {
                throw InvalidFenException("Halfmove clock too large (max 255)");
            }
        }

        out.meta.halfmoveClock = static_cast<std::uint8_t>(value);
    }

    /// @brief Parse le numéro de coup complet (champ 6)
    static void parseFullmoveNumber(std::string_view fullmove, ObsStateT<ChessTag>& out) {
        if (fullmove.empty()) {
            throw InvalidFenException("Fullmove number field is empty");
        }

        int value = 0;
        for (char c : fullmove) {
            if (!std::isdigit(c)) {
                throw InvalidFenException("Fullmove number must be a number");
            }
            value = value * 10 + (c - '0');
            if (value > 65535) {
                throw InvalidFenException("Fullmove number too large (max 65535)");
            }
        }

        if (value < 1) {
            throw InvalidFenException("Fullmove number must be at least 1");
        }

        out.meta.fullmoveNumber = static_cast<std::uint16_t>(value);
    }
};