#include "ChessEngine.hpp"

#include <iostream>
#include <bitset>
#include <cassert>
#include <cctype>
#include <bit>

ChessEngine::ChessEngine()
	: m_validActionsBuf(0)
{
}

void ChessEngine::specificSetup(const YAML::Node& config)
{
	std::cout << "ChessEngine setup called\n";

	m_validActionsBuf.reserve(m_maxValidActions);
}

void ChessEngine::getInitialState(ObsStateT<ChessTag>& out)
{
	FenParser::getFenState("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", out);
	//FenParser::getFenState("rnbqkbnr/1pp2ppp/p2p4/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4", out);
}

uint8_t ChessEngine::getCurrentPlayer(const ObsStateT<ChessTag>& obsState)
{
	return obsState.meta.trait;
}

void ChessEngine::getValidActions(const ObsStateT<ChessTag>& obsState, AlignedVec<ActionT<ChessTag>>& out)
{
	uint8_t meta0 = obsState.meta.trait;
	uint8_t meta1 = obsState.meta.castlingRights;
	uint8_t meta2 = obsState.meta.enPassant;

	uint8_t status =
		((meta0 == 0) ? 1 : 0) |
		((meta1 & 0b0001) ? 1 : 0) << 1 |
		((meta1 & 0b0010) ? 1 : 0) << 2 |
		((meta1 & 0b0100) ? 1 : 0) << 3 |
		((meta1 & 0b1000) ? 1 : 0) << 4 |
		((meta2 != 0) ? 1 : 0) << 5;

	switch (status)
	{
	case  1: MoveGenerator< 1>::generate(obsState, out); break;
	case  0: MoveGenerator< 0>::generate(obsState, out); break;
	case  2: MoveGenerator< 2>::generate(obsState, out); break;
	case  3: MoveGenerator< 3>::generate(obsState, out); break;
	case  4: MoveGenerator< 4>::generate(obsState, out); break;
	case  5: MoveGenerator< 5>::generate(obsState, out); break;
	case  6: MoveGenerator< 6>::generate(obsState, out); break;
	case  7: MoveGenerator< 7>::generate(obsState, out); break;
	case  8: MoveGenerator< 8>::generate(obsState, out); break;
	case  9: MoveGenerator< 9>::generate(obsState, out); break;
	case 10: MoveGenerator<10>::generate(obsState, out); break;
	case 11: MoveGenerator<11>::generate(obsState, out); break;
	case 12: MoveGenerator<12>::generate(obsState, out); break;
	case 13: MoveGenerator<13>::generate(obsState, out); break;
	case 14: MoveGenerator<14>::generate(obsState, out); break;
	case 15: MoveGenerator<15>::generate(obsState, out); break;
	case 16: MoveGenerator<16>::generate(obsState, out); break;
	case 17: MoveGenerator<17>::generate(obsState, out); break;
	case 18: MoveGenerator<18>::generate(obsState, out); break;
	case 19: MoveGenerator<19>::generate(obsState, out); break;
	case 20: MoveGenerator<20>::generate(obsState, out); break;
	case 21: MoveGenerator<21>::generate(obsState, out); break;
	case 22: MoveGenerator<22>::generate(obsState, out); break;
	case 23: MoveGenerator<23>::generate(obsState, out); break;
	case 24: MoveGenerator<24>::generate(obsState, out); break;
	case 25: MoveGenerator<25>::generate(obsState, out); break;
	case 26: MoveGenerator<26>::generate(obsState, out); break;
	case 27: MoveGenerator<27>::generate(obsState, out); break;
	case 28: MoveGenerator<28>::generate(obsState, out); break;
	case 29: MoveGenerator<29>::generate(obsState, out); break;
	case 30: MoveGenerator<30>::generate(obsState, out); break;
	case 31: MoveGenerator<31>::generate(obsState, out); break;
	case 32: MoveGenerator<32>::generate(obsState, out); break;
	case 33: MoveGenerator<33>::generate(obsState, out); break;
	case 34: MoveGenerator<34>::generate(obsState, out); break;
	case 35: MoveGenerator<35>::generate(obsState, out); break;
	case 36: MoveGenerator<36>::generate(obsState, out); break;
	case 37: MoveGenerator<37>::generate(obsState, out); break;
	case 38: MoveGenerator<38>::generate(obsState, out); break;
	case 39: MoveGenerator<39>::generate(obsState, out); break;
	case 40: MoveGenerator<40>::generate(obsState, out); break;
	case 41: MoveGenerator<41>::generate(obsState, out); break;
	case 42: MoveGenerator<42>::generate(obsState, out); break;
	case 43: MoveGenerator<43>::generate(obsState, out); break;
	case 44: MoveGenerator<44>::generate(obsState, out); break;
	case 45: MoveGenerator<45>::generate(obsState, out); break;
	case 46: MoveGenerator<46>::generate(obsState, out); break;
	case 47: MoveGenerator<47>::generate(obsState, out); break;
	case 48: MoveGenerator<48>::generate(obsState, out); break;
	case 49: MoveGenerator<49>::generate(obsState, out); break;
	case 50: MoveGenerator<50>::generate(obsState, out); break;
	case 51: MoveGenerator<51>::generate(obsState, out); break;
	case 52: MoveGenerator<52>::generate(obsState, out); break;
	case 53: MoveGenerator<53>::generate(obsState, out); break;
	case 54: MoveGenerator<54>::generate(obsState, out); break;
	case 55: MoveGenerator<55>::generate(obsState, out); break;
	case 56: MoveGenerator<56>::generate(obsState, out); break;
	case 57: MoveGenerator<57>::generate(obsState, out); break;
	case 58: MoveGenerator<58>::generate(obsState, out); break;
	case 59: MoveGenerator<59>::generate(obsState, out); break;
	case 60: MoveGenerator<60>::generate(obsState, out); break;
	case 61: MoveGenerator<61>::generate(obsState, out); break;
	case 62: MoveGenerator<62>::generate(obsState, out); break;
	case 63: MoveGenerator<63>::generate(obsState, out); break;
	default:
		assert(false && "Status out of range [0..63]"); break;
	}
}

bool ChessEngine::isValidAction(const ObsStateT<ChessTag>& obsState, const ActionT<ChessTag>& action)
{
	m_validActionsBuf.clear();
	getValidActions(obsState, m_validActionsBuf);

	for (int i = 0; i < m_validActionsBuf.size(); ++i)
	{
		if (m_validActionsBuf[i] == action) return true;
	}
	return false;
}

void ChessEngine::applyAction(const ActionT<ChessTag>& action, ObsStateT<ChessTag>& out)
{
	uint8_t meta0 = out.meta.trait;
	uint8_t meta1 = out.meta.castlingRights;
	uint8_t meta2 = out.meta.enPassant;

	uint8_t status =
		((meta0 == 0) ? 1 : 0) |
		((meta1 & 0b0001) ? 1 : 0) << 1 |
		((meta1 & 0b0010) ? 1 : 0) << 2 |
		((meta1 & 0b0100) ? 1 : 0) << 3 |
		((meta1 & 0b1000) ? 1 : 0) << 4 |
		((meta2 != 0) ? 1 : 0) << 5;

	switch (status)
	{
	case  0: MoveGenerator< 0>::apply(action, out); break;
	case  1: MoveGenerator< 1>::apply(action, out); break;
	case  2: MoveGenerator< 2>::apply(action, out); break;
	case  3: MoveGenerator< 3>::apply(action, out); break;
	case  4: MoveGenerator< 4>::apply(action, out); break;
	case  5: MoveGenerator< 5>::apply(action, out); break;
	case  6: MoveGenerator< 6>::apply(action, out); break;
	case  7: MoveGenerator< 7>::apply(action, out); break;
	case  8: MoveGenerator< 8>::apply(action, out); break;
	case  9: MoveGenerator< 9>::apply(action, out); break;
	case 10: MoveGenerator<10>::apply(action, out); break;
	case 11: MoveGenerator<11>::apply(action, out); break;
	case 12: MoveGenerator<12>::apply(action, out); break;
	case 13: MoveGenerator<13>::apply(action, out); break;
	case 14: MoveGenerator<14>::apply(action, out); break;
	case 15: MoveGenerator<15>::apply(action, out); break;
	case 16: MoveGenerator<16>::apply(action, out); break;
	case 17: MoveGenerator<17>::apply(action, out); break;
	case 18: MoveGenerator<18>::apply(action, out); break;
	case 19: MoveGenerator<19>::apply(action, out); break;
	case 20: MoveGenerator<20>::apply(action, out); break;
	case 21: MoveGenerator<21>::apply(action, out); break;
	case 22: MoveGenerator<22>::apply(action, out); break;
	case 23: MoveGenerator<23>::apply(action, out); break;
	case 24: MoveGenerator<24>::apply(action, out); break;
	case 25: MoveGenerator<25>::apply(action, out); break;
	case 26: MoveGenerator<26>::apply(action, out); break;
	case 27: MoveGenerator<27>::apply(action, out); break;
	case 28: MoveGenerator<28>::apply(action, out); break;
	case 29: MoveGenerator<29>::apply(action, out); break;
	case 30: MoveGenerator<30>::apply(action, out); break;
	case 31: MoveGenerator<31>::apply(action, out); break;
	case 32: MoveGenerator<32>::apply(action, out); break;
	case 33: MoveGenerator<33>::apply(action, out); break;
	case 34: MoveGenerator<34>::apply(action, out); break;
	case 35: MoveGenerator<35>::apply(action, out); break;
	case 36: MoveGenerator<36>::apply(action, out); break;
	case 37: MoveGenerator<37>::apply(action, out); break;
	case 38: MoveGenerator<38>::apply(action, out); break;
	case 39: MoveGenerator<39>::apply(action, out); break;
	case 40: MoveGenerator<40>::apply(action, out); break;
	case 41: MoveGenerator<41>::apply(action, out); break;
	case 42: MoveGenerator<42>::apply(action, out); break;
	case 43: MoveGenerator<43>::apply(action, out); break;
	case 44: MoveGenerator<44>::apply(action, out); break;
	case 45: MoveGenerator<45>::apply(action, out); break;
	case 46: MoveGenerator<46>::apply(action, out); break;
	case 47: MoveGenerator<47>::apply(action, out); break;
	case 48: MoveGenerator<48>::apply(action, out); break;
	case 49: MoveGenerator<49>::apply(action, out); break;
	case 50: MoveGenerator<50>::apply(action, out); break;
	case 51: MoveGenerator<51>::apply(action, out); break;
	case 52: MoveGenerator<52>::apply(action, out); break;
	case 53: MoveGenerator<53>::apply(action, out); break;
	case 54: MoveGenerator<54>::apply(action, out); break;
	case 55: MoveGenerator<55>::apply(action, out); break;
	case 56: MoveGenerator<56>::apply(action, out); break;
	case 57: MoveGenerator<57>::apply(action, out); break;
	case 58: MoveGenerator<58>::apply(action, out); break;
	case 59: MoveGenerator<59>::apply(action, out); break;
	case 60: MoveGenerator<60>::apply(action, out); break;
	case 61: MoveGenerator<61>::apply(action, out); break;
	case 62: MoveGenerator<62>::apply(action, out); break;
	case 63: MoveGenerator<63>::apply(action, out); break;
	default:
		assert(false && "Status out of range [0..63]"); break;
	}
}

bool ChessEngine::isFiftyMoveRule(const ObsStateT<ChessTag>& obsState)
{
	return (obsState.meta.halfmoveClock >= 100);
}
bool ChessEngine::isInsufficientMaterial(const ObsStateT<ChessTag>& obsState)
{
	// Pawns, rooks, queens -> sufficient
	if (obsState.elems.whiteBB[0] != 0 || obsState.elems.whiteBB[3] != 0 || obsState.elems.whiteBB[4] != 0 ||
		obsState.elems.blackBB[0] != 0 || obsState.elems.blackBB[3] != 0 || obsState.elems.blackBB[4] != 0)
		return false;

	int whiteKnightsCount = std::popcount(obsState.elems.whiteBB[1]);
	int whiteBishopsCount = std::popcount(obsState.elems.whiteBB[2]);
	int blackKnightsCount = std::popcount(obsState.elems.blackBB[1]);
	int blackBishopsCount = std::popcount(obsState.elems.blackBB[2]);

	int whiteMinor = whiteKnightsCount + whiteBishopsCount;
	int blackMinor = blackKnightsCount + blackBishopsCount;

	// Only allowed configurations for insufficient material:
	// K vs K
	// K+N vs K
	// K+B vs K
	// K+B vs K+B
	// K+N vs K+N
	if ((whiteMinor == 0 || whiteMinor == 1) &&
		(blackMinor == 0 || blackMinor == 1))
	{
		return true;
	}

	// Otherwise, sufficient material
	return false;
}

bool ChessEngine::ourKingInCheck(const ObsStateT<ChessTag>& obsState)
{
	int checkCount = 0;

	uint8_t meta0 = obsState.meta.trait;
	uint8_t meta1 = obsState.meta.castlingRights;
	uint8_t meta2 = obsState.meta.enPassant;

	uint8_t status =
		((meta0 == 0) ? 1 : 0) |
		((meta1 & 0b0001) ? 1 : 0) << 1 |
		((meta1 & 0b0010) ? 1 : 0) << 2 |
		((meta1 & 0b0100) ? 1 : 0) << 3 |
		((meta1 & 0b1000) ? 1 : 0) << 4 |
		((meta2 != 0) ? 1 : 0) << 5;

	switch (status)
	{
	case  0: MoveGenerator< 0>::countCheck(obsState, checkCount); break;
	case  1: MoveGenerator< 1>::countCheck(obsState, checkCount); break;
	case  2: MoveGenerator< 2>::countCheck(obsState, checkCount); break;
	case  3: MoveGenerator< 3>::countCheck(obsState, checkCount); break;
	case  4: MoveGenerator< 4>::countCheck(obsState, checkCount); break;
	case  5: MoveGenerator< 5>::countCheck(obsState, checkCount); break;
	case  6: MoveGenerator< 6>::countCheck(obsState, checkCount); break;
	case  7: MoveGenerator< 7>::countCheck(obsState, checkCount); break;
	case  8: MoveGenerator< 8>::countCheck(obsState, checkCount); break;
	case  9: MoveGenerator< 9>::countCheck(obsState, checkCount); break;
	case 10: MoveGenerator<10>::countCheck(obsState, checkCount); break;
	case 11: MoveGenerator<11>::countCheck(obsState, checkCount); break;
	case 12: MoveGenerator<12>::countCheck(obsState, checkCount); break;
	case 13: MoveGenerator<13>::countCheck(obsState, checkCount); break;
	case 14: MoveGenerator<14>::countCheck(obsState, checkCount); break;
	case 15: MoveGenerator<15>::countCheck(obsState, checkCount); break;
	case 16: MoveGenerator<16>::countCheck(obsState, checkCount); break;
	case 17: MoveGenerator<17>::countCheck(obsState, checkCount); break;
	case 18: MoveGenerator<18>::countCheck(obsState, checkCount); break;
	case 19: MoveGenerator<19>::countCheck(obsState, checkCount); break;
	case 20: MoveGenerator<20>::countCheck(obsState, checkCount); break;
	case 21: MoveGenerator<21>::countCheck(obsState, checkCount); break;
	case 22: MoveGenerator<22>::countCheck(obsState, checkCount); break;
	case 23: MoveGenerator<23>::countCheck(obsState, checkCount); break;
	case 24: MoveGenerator<24>::countCheck(obsState, checkCount); break;
	case 25: MoveGenerator<25>::countCheck(obsState, checkCount); break;
	case 26: MoveGenerator<26>::countCheck(obsState, checkCount); break;
	case 27: MoveGenerator<27>::countCheck(obsState, checkCount); break;
	case 28: MoveGenerator<28>::countCheck(obsState, checkCount); break;
	case 29: MoveGenerator<29>::countCheck(obsState, checkCount); break;
	case 30: MoveGenerator<30>::countCheck(obsState, checkCount); break;
	case 31: MoveGenerator<31>::countCheck(obsState, checkCount); break;
	case 32: MoveGenerator<32>::countCheck(obsState, checkCount); break;
	case 33: MoveGenerator<33>::countCheck(obsState, checkCount); break;
	case 34: MoveGenerator<34>::countCheck(obsState, checkCount); break;
	case 35: MoveGenerator<35>::countCheck(obsState, checkCount); break;
	case 36: MoveGenerator<36>::countCheck(obsState, checkCount); break;
	case 37: MoveGenerator<37>::countCheck(obsState, checkCount); break;
	case 38: MoveGenerator<38>::countCheck(obsState, checkCount); break;
	case 39: MoveGenerator<39>::countCheck(obsState, checkCount); break;
	case 40: MoveGenerator<40>::countCheck(obsState, checkCount); break;
	case 41: MoveGenerator<41>::countCheck(obsState, checkCount); break;
	case 42: MoveGenerator<42>::countCheck(obsState, checkCount); break;
	case 43: MoveGenerator<43>::countCheck(obsState, checkCount); break;
	case 44: MoveGenerator<44>::countCheck(obsState, checkCount); break;
	case 45: MoveGenerator<45>::countCheck(obsState, checkCount); break;
	case 46: MoveGenerator<46>::countCheck(obsState, checkCount); break;
	case 47: MoveGenerator<47>::countCheck(obsState, checkCount); break;
	case 48: MoveGenerator<48>::countCheck(obsState, checkCount); break;
	case 49: MoveGenerator<49>::countCheck(obsState, checkCount); break;
	case 50: MoveGenerator<50>::countCheck(obsState, checkCount); break;
	case 51: MoveGenerator<51>::countCheck(obsState, checkCount); break;
	case 52: MoveGenerator<52>::countCheck(obsState, checkCount); break;
	case 53: MoveGenerator<53>::countCheck(obsState, checkCount); break;
	case 54: MoveGenerator<54>::countCheck(obsState, checkCount); break;
	case 55: MoveGenerator<55>::countCheck(obsState, checkCount); break;
	case 56: MoveGenerator<56>::countCheck(obsState, checkCount); break;
	case 57: MoveGenerator<57>::countCheck(obsState, checkCount); break;
	case 58: MoveGenerator<58>::countCheck(obsState, checkCount); break;
	case 59: MoveGenerator<59>::countCheck(obsState, checkCount); break;
	case 60: MoveGenerator<60>::countCheck(obsState, checkCount); break;
	case 61: MoveGenerator<61>::countCheck(obsState, checkCount); break;
	case 62: MoveGenerator<62>::countCheck(obsState, checkCount); break;
	case 63: MoveGenerator<63>::countCheck(obsState, checkCount); break;
	default:
		assert(false && "Status out of range [0..63]"); break;
	}
	return (checkCount > 0);
}

bool ChessEngine::isTerminal(const ObsStateT<ChessTag>& obsState, AlignedVec<float>& out)
{
	// draw
	if (isFiftyMoveRule(obsState) || isInsufficientMaterial(obsState))
	{
		return true;
	}

	m_validActionsBuf.clear();
	getValidActions(obsState, m_validActionsBuf);

	// No valid move
	if (m_validActionsBuf.empty())
	{
		// Stalemate
		if (!ourKingInCheck(obsState))
		{
			return true;
		}
		// Black wins
		else if (obsState.meta.trait == 0)
		{
			out[0] = -1.f;	// White loses
			out[1] = 1.f;	// Black wins
			return true;
		}
		// White wins
		else
		{
			out[0] = 1.f;	// White wins
			out[1] = -1.f;	// Black loses
			return true;
		}
	}
	return false;
}

void ChessEngine::obsToIdx(const ObsStateT<ChessTag>& obsState, IdxStateT<ChessTag>& out)
{

}
void ChessEngine::idxToObs(const IdxStateT<ChessTag>& idxInput, ObsStateT<ChessTag>& out)
{

}

void ChessEngine::actionToIdx(const ActionT<ChessTag>& action, IdxActionT& out)
{
	const uint8_t fromIdx = action.from();
	const uint8_t toIdx = action.to();
	const uint8_t promo = action.promo();

	const uint8_t rankFrom = fromIdx / 8;
	const uint8_t fileFrom = fromIdx % 8;
	const uint8_t rankTo = toIdx / 8;
	const uint8_t fileTo = toIdx % 8;

	const int8_t rankDiff = rankTo - rankFrom;
	const int8_t fileDiff = fileTo - fileFrom;

	const uint16_t encodedFrom = fromIdx * 73;

	uint16_t encodedTo = 0;
	if (promo == 0)
	{
		// Sliding Pieces
		if ((std::abs(rankDiff) == std::abs(fileDiff))
			|| (rankDiff == 0)
			|| (fileDiff == 0))
		{
			if (rankDiff > 0 && fileDiff == 0) encodedTo = 0 + (rankDiff - 1);
			else if (rankDiff > 0 && fileDiff > 0)  encodedTo = 7 + (rankDiff - 1);
			else if (rankDiff == 0 && fileDiff > 0) encodedTo = 14 + (fileDiff - 1);
			else if (rankDiff < 0 && fileDiff > 0)  encodedTo = 21 + (fileDiff - 1);
			else if (rankDiff < 0 && fileDiff == 0)	encodedTo = 28 + (-rankDiff - 1);
			else if (rankDiff < 0 && fileDiff < 0)	encodedTo = 35 + (-rankDiff - 1);
			else if (rankDiff == 0 && fileDiff < 0)	encodedTo = 42 + (-fileDiff - 1);
			else if (rankDiff > 0 && fileDiff < 0)	encodedTo = 49 + (-fileDiff - 1);
			else throw std::runtime_error("ChessEngine::getValidEncodedActions: Invalid sliding piece move");
		}
		// Knight Moves
		else if ((std::abs(rankDiff) == 2 && std::abs(fileDiff) == 1)
			|| (std::abs(rankDiff) == 1 && std::abs(fileDiff) == 2))
		{
			if (rankDiff == 2 && fileDiff == 1)   encodedTo = 56;
			else if (rankDiff == 1 && fileDiff == 2)   encodedTo = 57;
			else if (rankDiff == -1 && fileDiff == 2)  encodedTo = 58;
			else if (rankDiff == -2 && fileDiff == 1)  encodedTo = 59;
			else if (rankDiff == -2 && fileDiff == -1) encodedTo = 60;
			else if (rankDiff == -1 && fileDiff == -2) encodedTo = 61;
			else if (rankDiff == 1 && fileDiff == -2)  encodedTo = 62;
			else if (rankDiff == 2 && fileDiff == -1)  encodedTo = 63;
			else throw std::runtime_error("ChessEngine::getValidEncodedActions: Invalid knight move");
		}
		else
			throw std::runtime_error("ChessEngine::getValidEncodedActions: Invalid piece move");
	}
	else
	{
		// Promotion Type (Queen and Bishop are 0)
		int promoType = 0;
		if (promo == 2) promoType = 1;
		else if (promo == 4) promoType = 2;

		if (std::abs(rankDiff) == 1 && fileDiff == 0)  encodedTo = 64 + promoType;
		else if (std::abs(rankDiff) == 1 && fileDiff == -1) encodedTo = 67 + promoType;
		else if (std::abs(rankDiff) == 1 && fileDiff == 1)  encodedTo = 70 + promoType;
		else throw std::runtime_error("ChessEngine::getValidEncodedActions: Invalid promotion move");
	}

	Fact::makePublicAction(encodedFrom + encodedTo, toIdx, out);
}
void ChessEngine::idxToAction(const IdxActionT& idxAction, ActionT<ChessTag>& out)
{

}