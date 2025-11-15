#pragma once
#include "../../corelib/interfaces/ITraits.hpp"

struct ChessTag {};

template<>
struct ITraits<ChessTag>
{
    static constexpr size_t kNumPlayers = 2;        // White, Black
	
    static constexpr size_t kNumElems = 12;         // 6 piece types * 2 colors
	static constexpr size_t kNumMeta = 6;           // trait, castling rights, en passant, halfmove clock, fullmove number, repetitions	
    static constexpr size_t kActionSpace = 4672;    // 8x8x73
    
    static constexpr size_t kNumPos = 64;           // 8x8 board
    static constexpr size_t kMaxValidActions = 218; // 218 maximum actions in any position

    struct ObsElems
    {
        uint64_t whiteBB[6];
		uint64_t blackBB[6];
    };

    struct Meta
    {
		uint8_t trait;              // 0: White to move, 1: Black to move
		uint8_t castlingRights;     // bit 0: White King-side, bit 1: White Queen-side, bit 2: Black King-side, bit 3: Black Queen-side
		uint8_t enPassant;          // target square index (0-63)
		uint8_t halfmoveClock;      // number of halfmoves since last capture or pawn move
		uint8_t fullmoveNumber;     // number of full moves in the game
		uint8_t repetitions;        // number of times the current position has occurred
    };

    struct Action
    {
        uint16_t data = 0;

		inline uint8_t from() const noexcept { return static_cast<uint8_t>(data & 0x3F); }          // bits 0-5
        inline uint8_t to() const noexcept { return static_cast<uint8_t>((data >> 6) & 0x3F); }     // bits 6-11
        inline uint8_t promo() const noexcept { return static_cast<uint8_t>((data >> 12) & 0xF); }  // bits 12-15

		inline void setFrom(uint8_t from) noexcept { data = (data & ~0x3F) | (static_cast<uint8_t>(from) & 0x3F); }
		inline void setTo(uint8_t to) noexcept { data = (data & ~(0x3F << 6)) | ((static_cast<uint8_t>(to) & 0x3F) << 6); }
		inline void setPromo(uint8_t promo) noexcept { data = (data & ~(0xF << 12)) | ((static_cast<uint8_t>(promo) & 0xF) << 12); }

        inline bool operator==(const Action& other) const noexcept
        {
            return (other.data == data);
        };
    };
};