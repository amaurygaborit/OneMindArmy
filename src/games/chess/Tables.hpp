#pragma once
#include <cstdint>

struct Tables
{
	static constexpr uint64_t knightMasks[64] =
	{
		#include "generated/knightMasks.inc"
	};

	static constexpr uint64_t bishopMasks[64] =
	{
		#include "generated/bishopMasks.inc"
	};

	static constexpr int bishopShifts[64] =
	{
		#include "generated/bishopShifts.inc"
	};

	static constexpr uint64_t bishopMagicNumbers[64] =
	{
		#include "generated/bishopMagicNumbers.inc"
	};

	static constexpr uint64_t bishopAttacks[64 * 512] =
	{
		#include "generated/bishopAttacks.inc"
	};

	static constexpr int bishopOffsets[64] =
	{
		#include "generated/bishopOffsets.inc"
	};

	static constexpr uint64_t rookMasks[64] =
	{
		#include "generated/rookMasks.inc"
	};

	static constexpr int rookShifts[64] =
	{
		#include "generated/rookShifts.inc"
	};

	static constexpr uint64_t rookMagicNumbers[64] =
	{
		#include "generated/rookMagicNumbers.inc"
	};

	static constexpr uint64_t rookAttacks[64 * 4096] =
	{
		#include "generated/rookAttacks.inc"
	};

	static constexpr int rookOffsets[64] =
	{
		#include "generated/rookOffsets.inc"
	};

	static constexpr uint64_t kingMasks[64] =
	{
		#include "generated/kingMasks.inc"
	};

	static constexpr uint64_t rayBetween[64 * 64] =
	{
		#include "generated/rayBetween.inc"
	};
};