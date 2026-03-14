#pragma once

namespace Chess
{
    class ChessEngine;
    class ChessRequester;
    class ChessRenderer;
    class UCIHandler;

    struct ChessTypes
    {        
        // ========================================================================
        // CONSTANTES OBLIGATOIRES (Concept ValidGameTraits)
        // ========================================================================
        
		// 6 types de pièces
        static constexpr uint32_t kNumElemTypes = 6;

        // 7 types de métadonnées
        static constexpr uint32_t kNumMetaTypes = 6;


        // 32 slot de pièces physiques maximum sur le plateau en même temps
        static constexpr uint32_t kMaxElems = 32;

        // 9 slots de métadonnées (Car les droits de roque prennent 4 slots, etc.)
        static constexpr uint32_t kMaxMetas = 8;


        // Players: White, Black
        static constexpr uint32_t kNumPlayers = 2;

        // Board 8x8
        static constexpr uint32_t kNumPos = 64;

        // Max valid actions for any given position
        static constexpr uint32_t kMaxValidActions = 218;

        // Action history size
        static constexpr uint32_t kMaxHistory = 8;

        // Action space for the neural network (8x8 x 73 plans)
        static constexpr uint32_t kActionSpace = 4672;

        using GameTypes = ChessTypes;
        using Engine = ChessEngine;
        using Requester = ChessRequester;
        using Renderer = ChessRenderer;
        using Handler = UCIHandler;
    };

    enum Piece : uint32_t
    {
		PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING
    };

    enum Meta : uint32_t
    {
        TURN = 0,
        CASTLING_KINGSIDE,
        CASTLING_QUEENSIDE,
        EN_PASSANT,
        HALF_MOVE,
        FULL_MOVE
    };

    enum MetaSlot : uint32_t
    {
        SLOT_TURN = 0,

        SLOT_CASTLING_WK,
        SLOT_CASTLING_WQ,
        SLOT_CASTLING_BK,
        SLOT_CASTLING_BQ,

        SLOT_EN_PASSANT,
        SLOT_HALF_MOVE,
        SLOT_FULL_MOVE
    };

    enum Player : uint32_t
    {
        WHITE,
        BLACK
    };

    enum Case : uint32_t
    {
        A1, B1, C1, D1, E1, F1, G1, H1,
        A2, B2, C2, D2, E2, F2, G2, H2,
        A3, B3, C3, D3, E3, F3, G3, H3,
        A4, B4, C4, D4, E4, F4, G4, H4,
        A5, B5, C5, D5, E5, F5, G5, H5,
        A6, B6, C6, D6, E6, F6, G6, H6,
        A7, B7, C7, D7, E7, F7, G7, H7,
        A8, B8, C8, D8, E8, F8, G8, H8
    };

    enum ChessEndReason : uint8_t
    {
        None = 0,
        Checkmate = 1,
        Stalemate = 2,           // Pat
        Repetition = 3,          // Triple répétition
        FiftyMoveRule = 4,       // Règle des 50 coups
        InsufficientMaterial = 5 // Matériel insuffisant
    };
}