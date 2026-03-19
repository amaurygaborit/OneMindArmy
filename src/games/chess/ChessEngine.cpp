#include "ChessEngine.hpp"

#include <iostream>
#include <bitset>
#include <cassert>
#include <cctype>
#include <bit>

namespace Chess
{
	USING_GAME_TYPES(ChessTypes);

	ChessEngine::ChessEngine()
	{
	}

	void ChessEngine::specificSetup(const YAML::Node& config)
	{
		std::cout << "ChessEngine setup called\n";

		ZobristHasher::ignoreMetaType(HALF_MOVE);
		ZobristHasher::ignoreMetaType(FULL_MOVE);
	}

	void ChessEngine::stateToBB(const State& state, StateBB& outStateBB) const
	{
		for (size_t i = 0; i < Defs::kMaxElems; ++i)
		{
			const auto& f = state.getElem(i);
			if (f.exists())
			{
				if (f.ownerId() == WHITE)
					outStateBB.whiteBB[f.factId()] |= (1ULL << f.pos());
				else
					outStateBB.blackBB[f.factId()] |= (1ULL << f.pos());
			}
		}
		outStateBB.enPassant = static_cast<uint8_t>(state.getMeta(SLOT_EN_PASSANT).value());
	}

	void ChessEngine::getInitialState(const uint32_t player, State& outState) const
	{
		FenParser::getFenState("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", outState);
		//FenParser::getFenState("3k4/8/p3p3/P1p1P1p1/2P3P1/8/3K4/8 w - - 0 1", outState);
	}

	uint32_t ChessEngine::getCurrentPlayer(const State& state) const
	{
		return state.getMeta(SLOT_TURN).ownerId();
	}

	ActionList ChessEngine::getValidActions(const State& state, std::span<const uint64_t> hashHistory) const
	{
		ActionList actionList{};
		StateBB stateBB{};
		stateToBB(state, stateBB);

		bool isWhite = (state.getMeta(SLOT_TURN).ownerId() == WHITE);
		bool wk = state.getMeta(SLOT_CASTLING_WK).exists();
		bool wq = state.getMeta(SLOT_CASTLING_WQ).exists();
		bool bk = state.getMeta(SLOT_CASTLING_BK).exists();
		bool bq = state.getMeta(SLOT_CASTLING_BQ).exists();

		auto epVal = static_cast<size_t>(state.getMeta(SLOT_EN_PASSANT).pos());
		bool hasEp = epVal != Defs::kNoPos;
		stateBB.enPassant = static_cast<uint8_t>(epVal);

		// Bit 0: White, 1: WK, 2: WQ, 3: BK, 4: BQ, 5: HasEP
		uint8_t status =
			(isWhite ? 1 : 0) |
			(wk ? 2 : 0) |
			(wq ? 4 : 0) |
			(bk ? 8 : 0) |
			(bq ? 16 : 0) |
			(hasEp ? 32 : 0);

		switch (status)
		{
		case  1: MoveGenerator< 1>::generate(stateBB, actionList); break;
		case  0: MoveGenerator< 0>::generate(stateBB, actionList); break;
		case  2: MoveGenerator< 2>::generate(stateBB, actionList); break;
		case  3: MoveGenerator< 3>::generate(stateBB, actionList); break;
		case  4: MoveGenerator< 4>::generate(stateBB, actionList); break;
		case  5: MoveGenerator< 5>::generate(stateBB, actionList); break;
		case  6: MoveGenerator< 6>::generate(stateBB, actionList); break;
		case  7: MoveGenerator< 7>::generate(stateBB, actionList); break;
		case  8: MoveGenerator< 8>::generate(stateBB, actionList); break;
		case  9: MoveGenerator< 9>::generate(stateBB, actionList); break;
		case 10: MoveGenerator<10>::generate(stateBB, actionList); break;
		case 11: MoveGenerator<11>::generate(stateBB, actionList); break;
		case 12: MoveGenerator<12>::generate(stateBB, actionList); break;
		case 13: MoveGenerator<13>::generate(stateBB, actionList); break;
		case 14: MoveGenerator<14>::generate(stateBB, actionList); break;
		case 15: MoveGenerator<15>::generate(stateBB, actionList); break;
		case 16: MoveGenerator<16>::generate(stateBB, actionList); break;
		case 17: MoveGenerator<17>::generate(stateBB, actionList); break;
		case 18: MoveGenerator<18>::generate(stateBB, actionList); break;
		case 19: MoveGenerator<19>::generate(stateBB, actionList); break;
		case 20: MoveGenerator<20>::generate(stateBB, actionList); break;
		case 21: MoveGenerator<21>::generate(stateBB, actionList); break;
		case 22: MoveGenerator<22>::generate(stateBB, actionList); break;
		case 23: MoveGenerator<23>::generate(stateBB, actionList); break;
		case 24: MoveGenerator<24>::generate(stateBB, actionList); break;
		case 25: MoveGenerator<25>::generate(stateBB, actionList); break;
		case 26: MoveGenerator<26>::generate(stateBB, actionList); break;
		case 27: MoveGenerator<27>::generate(stateBB, actionList); break;
		case 28: MoveGenerator<28>::generate(stateBB, actionList); break;
		case 29: MoveGenerator<29>::generate(stateBB, actionList); break;
		case 30: MoveGenerator<30>::generate(stateBB, actionList); break;
		case 31: MoveGenerator<31>::generate(stateBB, actionList); break;
		case 32: MoveGenerator<32>::generate(stateBB, actionList); break;
		case 33: MoveGenerator<33>::generate(stateBB, actionList); break;
		case 34: MoveGenerator<34>::generate(stateBB, actionList); break;
		case 35: MoveGenerator<35>::generate(stateBB, actionList); break;
		case 36: MoveGenerator<36>::generate(stateBB, actionList); break;
		case 37: MoveGenerator<37>::generate(stateBB, actionList); break;
		case 38: MoveGenerator<38>::generate(stateBB, actionList); break;
		case 39: MoveGenerator<39>::generate(stateBB, actionList); break;
		case 40: MoveGenerator<40>::generate(stateBB, actionList); break;
		case 41: MoveGenerator<41>::generate(stateBB, actionList); break;
		case 42: MoveGenerator<42>::generate(stateBB, actionList); break;
		case 43: MoveGenerator<43>::generate(stateBB, actionList); break;
		case 44: MoveGenerator<44>::generate(stateBB, actionList); break;
		case 45: MoveGenerator<45>::generate(stateBB, actionList); break;
		case 46: MoveGenerator<46>::generate(stateBB, actionList); break;
		case 47: MoveGenerator<47>::generate(stateBB, actionList); break;
		case 48: MoveGenerator<48>::generate(stateBB, actionList); break;
		case 49: MoveGenerator<49>::generate(stateBB, actionList); break;
		case 50: MoveGenerator<50>::generate(stateBB, actionList); break;
		case 51: MoveGenerator<51>::generate(stateBB, actionList); break;
		case 52: MoveGenerator<52>::generate(stateBB, actionList); break;
		case 53: MoveGenerator<53>::generate(stateBB, actionList); break;
		case 54: MoveGenerator<54>::generate(stateBB, actionList); break;
		case 55: MoveGenerator<55>::generate(stateBB, actionList); break;
		case 56: MoveGenerator<56>::generate(stateBB, actionList); break;
		case 57: MoveGenerator<57>::generate(stateBB, actionList); break;
		case 58: MoveGenerator<58>::generate(stateBB, actionList); break;
		case 59: MoveGenerator<59>::generate(stateBB, actionList); break;
		case 60: MoveGenerator<60>::generate(stateBB, actionList); break;
		case 61: MoveGenerator<61>::generate(stateBB, actionList); break;
		case 62: MoveGenerator<62>::generate(stateBB, actionList); break;
		case 63: MoveGenerator<63>::generate(stateBB, actionList); break;
		default:
			assert(false && "Status out of range [0..63]"); break;
		}

		return actionList;
	}
	bool ChessEngine::isValidAction(const State& state, std::span<const uint64_t> hashHistory, const Action& action) const
	{
		ActionList actionList = getValidActions(state, hashHistory);

		for (size_t a = 0; a < actionList.size(); ++a)
		{
			if (actionList[a] == action) return true;
		}
		return false;
	}

	bool ChessEngine::isFiftyMoveRule(const State& state) const
	{
		// 100 demi-coups (ply) = 50 coups complets sans prise ni poussée de pion
		return (state.getMeta(SLOT_HALF_MOVE).value() >= 100.0f);
	}
	bool ChessEngine::isInsufficientMaterial(const State& state) const
	{
		int whiteKnights = 0, blackKnights = 0;
		int whiteBishops = 0, blackBishops = 0;
		int whiteBishopColor = -1, blackBishopColor = -1;

		// Parcours unique des 32 slots de pièces
		for (const auto& piece : state.elems())
		{
			if (!piece.exists() || piece.pos() == Defs::kNoPos)
				continue;

			bool isWhite = (piece.ownerId() == WHITE);

			switch (piece.factId())
			{
				// Si on trouve un Pion, une Tour ou une Dame -> Matériel suffisant
			case PAWN:
			case ROOK:
			case QUEEN:
				return false;

			case KNIGHT:
				if (isWhite) whiteKnights++;
				else         blackKnights++;
				break;

			case BISHOP:
				if (isWhite) {
					whiteBishops++;
					// Détermination de la couleur de la case : (Ligne + Colonne) % 2
					whiteBishopColor = ((piece.pos() / 8) + (piece.pos() % 8)) % 2;
				}
				else {
					blackBishops++;
					blackBishopColor = ((piece.pos() / 8) + (piece.pos() % 8)) % 2;
				}
				break;

			case KING:
				break;
			}
		}

		int totalMinors = whiteKnights + blackKnights + whiteBishops + blackBishops;

		// 1. K vs K (0 pièce mineure)
		if (totalMinors == 0) return true;

		// 2. K+B vs K ou K+N vs K (Exactement 1 pièce mineure sur tout le plateau)
		if (totalMinors == 1) return true;

		// 3. K+B vs K+B avec des Fous de MÊME couleur (FIDE 9.6)
		if (totalMinors == 2 && whiteBishops == 1 && blackBishops == 1)
		{
			if (whiteBishopColor == blackBishopColor) return true;
		}

		// Tous les autres cas (K+N vs K+N, K+B vs K+N, etc.) peuvent techniquement mener à un mat !
		return false;
	}
	bool ChessEngine::ourKingInCheck(const State& state) const
	{
		StateBB stateBB{};
		stateToBB(state, stateBB);

		int checkCount = 0;

		bool isWhite = (state.getMeta(SLOT_TURN).ownerId() == WHITE);
		bool wk = state.getMeta(SLOT_CASTLING_WK).exists();
		bool wq = state.getMeta(SLOT_CASTLING_WQ).exists();
		bool bk = state.getMeta(SLOT_CASTLING_BK).exists();
		bool bq = state.getMeta(SLOT_CASTLING_BQ).exists();

		auto epVal = static_cast<size_t>(state.getMeta(SLOT_EN_PASSANT).pos());
		bool hasEp = epVal != Defs::kNoPos;
		stateBB.enPassant = static_cast<uint8_t>(epVal);

		// Bit 0: White, 1: WK, 2: WQ, 3: BK, 4: BQ, 5: HasEP
		uint8_t status =
			(isWhite ? 1 : 0) |
			(wk ? 2 : 0) |
			(wq ? 4 : 0) |
			(bk ? 8 : 0) |
			(bq ? 16 : 0) |
			(hasEp ? 32 : 0);

		switch (status)
		{
		case  0: MoveGenerator< 0>::countCheck(stateBB, checkCount); break;
		case  1: MoveGenerator< 1>::countCheck(stateBB, checkCount); break;
		case  2: MoveGenerator< 2>::countCheck(stateBB, checkCount); break;
		case  3: MoveGenerator< 3>::countCheck(stateBB, checkCount); break;
		case  4: MoveGenerator< 4>::countCheck(stateBB, checkCount); break;
		case  5: MoveGenerator< 5>::countCheck(stateBB, checkCount); break;
		case  6: MoveGenerator< 6>::countCheck(stateBB, checkCount); break;
		case  7: MoveGenerator< 7>::countCheck(stateBB, checkCount); break;
		case  8: MoveGenerator< 8>::countCheck(stateBB, checkCount); break;
		case  9: MoveGenerator< 9>::countCheck(stateBB, checkCount); break;
		case 10: MoveGenerator<10>::countCheck(stateBB, checkCount); break;
		case 11: MoveGenerator<11>::countCheck(stateBB, checkCount); break;
		case 12: MoveGenerator<12>::countCheck(stateBB, checkCount); break;
		case 13: MoveGenerator<13>::countCheck(stateBB, checkCount); break;
		case 14: MoveGenerator<14>::countCheck(stateBB, checkCount); break;
		case 15: MoveGenerator<15>::countCheck(stateBB, checkCount); break;
		case 16: MoveGenerator<16>::countCheck(stateBB, checkCount); break;
		case 17: MoveGenerator<17>::countCheck(stateBB, checkCount); break;
		case 18: MoveGenerator<18>::countCheck(stateBB, checkCount); break;
		case 19: MoveGenerator<19>::countCheck(stateBB, checkCount); break;
		case 20: MoveGenerator<20>::countCheck(stateBB, checkCount); break;
		case 21: MoveGenerator<21>::countCheck(stateBB, checkCount); break;
		case 22: MoveGenerator<22>::countCheck(stateBB, checkCount); break;
		case 23: MoveGenerator<23>::countCheck(stateBB, checkCount); break;
		case 24: MoveGenerator<24>::countCheck(stateBB, checkCount); break;
		case 25: MoveGenerator<25>::countCheck(stateBB, checkCount); break;
		case 26: MoveGenerator<26>::countCheck(stateBB, checkCount); break;
		case 27: MoveGenerator<27>::countCheck(stateBB, checkCount); break;
		case 28: MoveGenerator<28>::countCheck(stateBB, checkCount); break;
		case 29: MoveGenerator<29>::countCheck(stateBB, checkCount); break;
		case 30: MoveGenerator<30>::countCheck(stateBB, checkCount); break;
		case 31: MoveGenerator<31>::countCheck(stateBB, checkCount); break;
		case 32: MoveGenerator<32>::countCheck(stateBB, checkCount); break;
		case 33: MoveGenerator<33>::countCheck(stateBB, checkCount); break;
		case 34: MoveGenerator<34>::countCheck(stateBB, checkCount); break;
		case 35: MoveGenerator<35>::countCheck(stateBB, checkCount); break;
		case 36: MoveGenerator<36>::countCheck(stateBB, checkCount); break;
		case 37: MoveGenerator<37>::countCheck(stateBB, checkCount); break;
		case 38: MoveGenerator<38>::countCheck(stateBB, checkCount); break;
		case 39: MoveGenerator<39>::countCheck(stateBB, checkCount); break;
		case 40: MoveGenerator<40>::countCheck(stateBB, checkCount); break;
		case 41: MoveGenerator<41>::countCheck(stateBB, checkCount); break;
		case 42: MoveGenerator<42>::countCheck(stateBB, checkCount); break;
		case 43: MoveGenerator<43>::countCheck(stateBB, checkCount); break;
		case 44: MoveGenerator<44>::countCheck(stateBB, checkCount); break;
		case 45: MoveGenerator<45>::countCheck(stateBB, checkCount); break;
		case 46: MoveGenerator<46>::countCheck(stateBB, checkCount); break;
		case 47: MoveGenerator<47>::countCheck(stateBB, checkCount); break;
		case 48: MoveGenerator<48>::countCheck(stateBB, checkCount); break;
		case 49: MoveGenerator<49>::countCheck(stateBB, checkCount); break;
		case 50: MoveGenerator<50>::countCheck(stateBB, checkCount); break;
		case 51: MoveGenerator<51>::countCheck(stateBB, checkCount); break;
		case 52: MoveGenerator<52>::countCheck(stateBB, checkCount); break;
		case 53: MoveGenerator<53>::countCheck(stateBB, checkCount); break;
		case 54: MoveGenerator<54>::countCheck(stateBB, checkCount); break;
		case 55: MoveGenerator<55>::countCheck(stateBB, checkCount); break;
		case 56: MoveGenerator<56>::countCheck(stateBB, checkCount); break;
		case 57: MoveGenerator<57>::countCheck(stateBB, checkCount); break;
		case 58: MoveGenerator<58>::countCheck(stateBB, checkCount); break;
		case 59: MoveGenerator<59>::countCheck(stateBB, checkCount); break;
		case 60: MoveGenerator<60>::countCheck(stateBB, checkCount); break;
		case 61: MoveGenerator<61>::countCheck(stateBB, checkCount); break;
		case 62: MoveGenerator<62>::countCheck(stateBB, checkCount); break;
		case 63: MoveGenerator<63>::countCheck(stateBB, checkCount); break;
		default:
			assert(false && "Status out of range [0..63]"); break;
		}
		return (checkCount > 0);
	}
	std::optional<GameResult> ChessEngine::getGameResult(
		const State& state,
		std::span<const uint64_t> hashHistory) const
	{
		// Définition des templates WDL absolus (Game Agnostic compatible)
		// Format: [White_Win, White_Draw, White_Loss, Black_Win, Black_Draw, Black_Loss]
		constexpr std::array<float, 6> WDL_DRAW = { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f };
		constexpr std::array<float, 6> WDL_WHITE = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
		constexpr std::array<float, 6> WDL_BLACK = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f };

		// ==================================================================
		// 1. HARD CAP — Plafond absolu de longueur de partie
		// ==================================================================
		if (static_cast<int>(hashHistory.size()) >= 300)
		{
			return GameResult{ WDL_DRAW, static_cast<uint32_t>(ChessEndReason::MaxPlyReached) };
		}

		// ==================================================================
		// 2. RÈGLE DES 50 COUPS
		// ==================================================================
		if (isFiftyMoveRule(state))
		{
			return GameResult{ WDL_DRAW, static_cast<uint32_t>(ChessEndReason::FiftyMoveRule) };
		}

		// ==================================================================
		// 3. MATÉRIEL INSUFFISANT
		// ==================================================================
		if (isInsufficientMaterial(state))
		{
			return GameResult{ WDL_DRAW, static_cast<uint32_t>(ChessEndReason::InsufficientMaterial) };
		}

		// ==================================================================
		// 4. TRIPLE RÉPÉTITION (FIDE 9.2)
		// ==================================================================
		{
			const uint64_t currentHash = state.hash();
			int            repetitions = 0;

			for (const uint64_t h : hashHistory)
			{
				if (h == currentHash)
					++repetitions;
			}

			if (repetitions >= 3)
			{
				return GameResult{ WDL_DRAW, static_cast<uint32_t>(ChessEndReason::Repetition) };
			}
		}

		// ==================================================================
		// 5. MAT / PAT
		// ==================================================================
		const ActionList actionList = getValidActions(state, hashHistory);

		if (actionList.empty())
		{
			if (!ourKingInCheck(state))
			{
				// Aucun coup légal, roi non en échec → Pat
				return GameResult{ WDL_DRAW, static_cast<uint32_t>(ChessEndReason::Stalemate) };
			}

			// Aucun coup légal, roi en échec → Mat
			const bool whiteToMove = (state.getMeta(SLOT_TURN).ownerId() == WHITE);
			if (whiteToMove)
				return GameResult{ WDL_BLACK, static_cast<uint32_t>(ChessEndReason::Checkmate) }; // Noirs gagnent
			else
				return GameResult{ WDL_WHITE, static_cast<uint32_t>(ChessEndReason::Checkmate) }; // Blancs gagnent
		}

		// La partie continue
		return std::nullopt;
	}

	GameResult ChessEngine::buildResignResult(uint32_t losingPlayer) const
	{
		// Format WDL : pour chaque joueur p, 3 floats consécutifs :
		//   scores[p*3 + 0] = P(Win)
		//   scores[p*3 + 1] = P(Draw)
		//   scores[p*3 + 2] = P(Loss)
		//
		// Le joueur qui résigne a perdu avec certitude  → {0, 0, 1}
		// Les autres joueurs ont gagné avec certitude   → {1, 0, 0}

		std::array<float, Defs::kNumPlayers * 3> scores{};  // zero-init

		for (size_t p = 0; p < Defs::kNumPlayers; ++p)
		{
			if (p == losingPlayer)
				scores[p * 3 + 2] = 1.0f;  // Loss = 1
			else
				scores[p * 3 + 0] = 1.0f;  // Win  = 1
			// Draw reste à 0 dans les deux cas
		}

		return GameResult{ scores, static_cast<uint32_t>(ChessEndReason::Resigned) };
	}

	void ChessEngine::changeStatePov(uint32_t viewer, State& outState) const
	{
		for (uint32_t idx = 0; idx < Defs::kMaxElems; ++idx)
		{
			PovUtils::doRotateOwnerAndMirrorElem(outState, idx, viewer);
		}

		PovUtils::doRotateOwnerOnlyMeta(outState, SLOT_TURN, viewer);

		// 2. ROQUES : On croise les slots Blancs [1, 2] avec les slots Noirs [3, 4]
		PovUtils::doRotateOwnerOnlyMeta(outState, SLOT_CASTLING_WK, viewer);
		PovUtils::doRotateOwnerOnlyMeta(outState, SLOT_CASTLING_WQ, viewer);
		PovUtils::doRotateOwnerOnlyMeta(outState, SLOT_CASTLING_BK, viewer);
		PovUtils::doRotateOwnerOnlyMeta(outState, SLOT_CASTLING_BQ, viewer);

		// 3. EN-PASSANT : Symétrie sur sa propre case (Slot 5)
		PovUtils::doRotateOwnerAndMirrorElem(outState, SLOT_EN_PASSANT, viewer);
	}
	void ChessEngine::changeActionPov(uint32_t viewer, Action& outAction) const
	{
		PovUtils::doRotateOwnerAndMirrorAction(outAction, viewer);
	}

	void ChessEngine::applyAction(const Action& action, State& outState) const
	{
		// 1. DÉCODAGE
		const uint32_t iFrom = action.source();
		const uint32_t iTo = action.dest();
		const uint32_t promoVal = static_cast<uint32_t>(action.value());

		// CORRECTION : On accède au meta via son SLOT
		const bool isWhiteTurn = (outState.getMeta(SLOT_TURN).ownerId() == WHITE);

		const size_t myStart = isWhiteTurn ? 0 : 16;
		const size_t myEnd = isWhiteTurn ? 16 : 32;
		const size_t opStart = isWhiteTurn ? 16 : 0;
		const size_t opEnd = isWhiteTurn ? 32 : 16;

		// 2. TROUVER LA PIÈCE DÉPLACÉE
		int movingIdx = -1;
		for (uint32_t i = myStart; i < myEnd; ++i) {
			if (outState.getElem(i).pos() == iFrom) {
				movingIdx = i;
				break;
			}
		}
		if (movingIdx == -1) return;

		const auto originalType = outState.getElem(movingIdx).factId();

		// CORRECTION MAGIQUE : Plus besoin de WP ou BP, PAWN suffit pour les deux couleurs !
		const bool isPawn = (originalType == PAWN);
		const bool isKing = (originalType == KING);
		const bool isRook = (originalType == ROOK);

		// 3. LECTURE EN-PASSANT (Via son SLOT)
		uint32_t epMetaVal = outState.getMeta(SLOT_EN_PASSANT).pos();

		// 4. GESTION DES PRISES
		bool captureOccurred = false;
		uint32_t captureSqIdx = iTo; // Par défaut, on tue sur la case d'arrivée

		if (isPawn && (iTo == epMetaVal) && (epMetaVal != Defs::kNoPos)) {
			captureSqIdx = isWhiteTurn ? (iTo - 8) : (iTo + 8); // Ajustement si prise en passant
		}

		// Exécution de la mort de la pièce adverse
		for (size_t i = opStart; i < opEnd; ++i) {
			if (outState.getElem(i).pos() == captureSqIdx) {
				outState.modifyElem(i)->kill();
				captureOccurred = true;
				break;
			}
		}

		// 5. ROQUE
		if (isKing && std::abs(static_cast<int>(iTo) - static_cast<int>(iFrom)) == 2) {
			uint32_t rookFrom = Defs::kNoPos, rookTo = Defs::kNoPos;
			if (iTo > iFrom) { // Petit Roque
				rookFrom = isWhiteTurn ? H1 : H8;
				rookTo = isWhiteTurn ? F1 : F8;
			}
			else {           // Grand Roque
				rookFrom = isWhiteTurn ? A1 : A8;
				rookTo = isWhiteTurn ? D1 : D8;
			}
			for (size_t i = myStart; i < myEnd; ++i) {
				if (outState.getElem(i).pos() == rookFrom) {
					outState.modifyElem(i)->setPos(rookTo);
					break;
				}
			}
		}

		// 6. PROMOTION & DÉPLACEMENT PRINCIPAL
		{
			auto mutator = outState.modifyElem(movingIdx);

			if (isPawn && promoVal > 0) {
				uint32_t newPieceType = PAWN;

				// CORRECTION MAGIQUE 2 : Plus de if(isWhiteTurn) ! Le switch est divisé par 2.
				// La pièce promue garde l'ownerId qu'elle avait déjà.
				switch (promoVal) {
				case 1: newPieceType = QUEEN; break;
				case 2: newPieceType = ROOK; break;
				case 3: newPieceType = BISHOP; break;
				case 4: newPieceType = KNIGHT; break;
				}

				// On utilise le setter qu'on a rajouté plus tôt dans Atom<GT>
				mutator->setFactId(newPieceType);
			}

			mutator->setPos(iTo);
		}

		// 7. MISE À JOUR DES MÉTADONNÉES (Tout passe en SLOT_xxx)

		// A. Droits de Roque
		if (isKing) {
			if (isWhiteTurn) {
				if (outState.getMeta(SLOT_CASTLING_WK).exists()) outState.modifyMeta(SLOT_CASTLING_WK)->kill();
				if (outState.getMeta(SLOT_CASTLING_WQ).exists()) outState.modifyMeta(SLOT_CASTLING_WQ)->kill();
			}
			else {
				if (outState.getMeta(SLOT_CASTLING_BK).exists()) outState.modifyMeta(SLOT_CASTLING_BK)->kill();
				if (outState.getMeta(SLOT_CASTLING_BQ).exists()) outState.modifyMeta(SLOT_CASTLING_BQ)->kill();
			}
		}
		else if (isRook) {
			if (iFrom == H1 && outState.getMeta(SLOT_CASTLING_WK).exists()) outState.modifyMeta(SLOT_CASTLING_WK)->kill();
			if (iFrom == A1 && outState.getMeta(SLOT_CASTLING_WQ).exists()) outState.modifyMeta(SLOT_CASTLING_WQ)->kill();
			if (iFrom == H8 && outState.getMeta(SLOT_CASTLING_BK).exists()) outState.modifyMeta(SLOT_CASTLING_BK)->kill();
			if (iFrom == A8 && outState.getMeta(SLOT_CASTLING_BQ).exists()) outState.modifyMeta(SLOT_CASTLING_BQ)->kill();
		}

		if (captureOccurred) {
			if (captureSqIdx == H1 && outState.getMeta(SLOT_CASTLING_WK).exists()) outState.modifyMeta(SLOT_CASTLING_WK)->kill();
			if (captureSqIdx == A1 && outState.getMeta(SLOT_CASTLING_WQ).exists()) outState.modifyMeta(SLOT_CASTLING_WQ)->kill();
			if (captureSqIdx == H8 && outState.getMeta(SLOT_CASTLING_BK).exists()) outState.modifyMeta(SLOT_CASTLING_BK)->kill();
			if (captureSqIdx == A8 && outState.getMeta(SLOT_CASTLING_BQ).exists()) outState.modifyMeta(SLOT_CASTLING_BQ)->kill();
		}

		// B. En-Passant
		if (isPawn && std::abs(static_cast<int>(iTo) - static_cast<int>(iFrom)) == 16) {
			uint32_t target = isWhiteTurn ? (iFrom + 8) : (iFrom - 8);
			auto ep = outState.modifyMeta(SLOT_EN_PASSANT);
			ep->setPos(target);
			ep->setValue(1.0f);
			ep->setOwner(isWhiteTurn ? BLACK : WHITE);
		}
		else if (epMetaVal != Defs::kNoPos) {
			outState.modifyMeta(SLOT_EN_PASSANT)->kill();
		}

		// C. Halfmove Clock
		if (isPawn || captureOccurred) {
			outState.modifyMeta(SLOT_HALF_MOVE)->setValue(0.0f);
		}
		else {
			outState.modifyMeta(SLOT_HALF_MOVE)->addValue(1.0f);
		}

		// D. Fullmove Number
		if (!isWhiteTurn) {
			outState.modifyMeta(SLOT_FULL_MOVE)->addValue(1.0f);
		}

		// E. Changement de Tour
		auto turn = outState.modifyMeta(SLOT_TURN);
		turn->setOwner(isWhiteTurn ? BLACK : WHITE);
	}

	uint32_t ChessEngine::actionToIdx(const Action& action) const
	{
		// 1. Extraction des données depuis le FactT
		const uint8_t fromIdx = static_cast<uint8_t>(action.source());
		const uint8_t toIdx = static_cast<uint8_t>(action.dest());

		// Rappel convention : 0=None, 1=Q, 2=R, 3=B, 4=N
		const int promoVal = static_cast<int>(action.value());

		const int rankFrom = fromIdx / 8;
		const int fileFrom = fromIdx % 8;
		const int rankTo = toIdx / 8;
		const int fileTo = toIdx % 8;

		const int rankDiff = rankTo - rankFrom;
		const int fileDiff = fileTo - fileFrom;

		// 73 plans par case de départ
		const uint16_t encodedFrom = fromIdx * 73;
		uint16_t encodedTo = 0;

		// 2. Logique d'encodage
		// Convention : Une promotion en Reine (1) est encodée comme un mouvement normal de la Dame.
		// Seules les sous-promotions (2=R, 3=B, 4=N) vont dans les plans spéciaux.
		if (promoVal == 0 || promoVal == 1)
		{
			// --- A. Sliding Pieces (Queen, Rook, Bishop, Pawn regular, King) ---
			// Vérifie si c'est un mouvement en ligne droite ou diagonale
			if ((std::abs(rankDiff) == std::abs(fileDiff)) || (rankDiff == 0) || (fileDiff == 0))
			{
				// Nord
				if (rankDiff > 0 && fileDiff == 0)      encodedTo = 0 + (rankDiff - 1);
				// Nord-Est
				else if (rankDiff > 0 && fileDiff > 0)  encodedTo = 7 + (rankDiff - 1);
				// Est
				else if (rankDiff == 0 && fileDiff > 0) encodedTo = 14 + (fileDiff - 1);
				// Sud-Est
				else if (rankDiff < 0 && fileDiff > 0)  encodedTo = 21 + (fileDiff - 1);
				// Sud
				else if (rankDiff < 0 && fileDiff == 0) encodedTo = 28 + (-rankDiff - 1);
				// Sud-Ouest
				else if (rankDiff < 0 && fileDiff < 0)  encodedTo = 35 + (-rankDiff - 1);
				// Ouest
				else if (rankDiff == 0 && fileDiff < 0) encodedTo = 42 + (-fileDiff - 1);
				// Nord-Ouest
				else if (rankDiff > 0 && fileDiff < 0)  encodedTo = 49 + (-fileDiff - 1);

				else throw std::runtime_error("ChessEngine::actionToIdx(): Invalid sliding piece move");
			}
			// --- B. Knight Moves ---
			else if ((std::abs(rankDiff) == 2 && std::abs(fileDiff) == 1) ||
				(std::abs(rankDiff) == 1 && std::abs(fileDiff) == 2))
			{
				if (rankDiff == 2 && fileDiff == 1) encodedTo = 56;
				else if (rankDiff == 1 && fileDiff == 2) encodedTo = 57;
				else if (rankDiff == -1 && fileDiff == 2) encodedTo = 58;
				else if (rankDiff == -2 && fileDiff == 1) encodedTo = 59;
				else if (rankDiff == -2 && fileDiff == -1) encodedTo = 60;
				else if (rankDiff == -1 && fileDiff == -2) encodedTo = 61;
				else if (rankDiff == 1 && fileDiff == -2) encodedTo = 62;
				else if (rankDiff == 2 && fileDiff == -1) encodedTo = 63;

				else throw std::runtime_error("ChessEngine::actionToIdx(): Invalid knight move");
			}
			else
			{
				throw std::runtime_error("ChessEngine::actionToIdx(): Invalid piece move (not sliding nor knight)");
			}
		}
		else
		{
			// --- C. Underpromotions (Rook, Bishop, Knight) ---
			// Plans 64 à 72

			// Mapping basé sur votre logique précédente :
			// Bishop (3) -> 0
			// Rook   (2) -> 1
			// Knight (4) -> 2
			int promoType = 0; // Par défaut Bishop (3)
			if (promoVal == 2)      promoType = 1; // Rook
			else if (promoVal == 4) promoType = 2; // Knight

			// Direction de la promotion (Capture gauche, Avance, Capture droite)
			// Note: Du point de vue absolu (rankDiff), mais fileDiff change selon la capture

			// Avance simple (ex: a7a8)
			if (std::abs(rankDiff) == 1 && fileDiff == 0)
				encodedTo = 64 + promoType;

			// Capture diagonale gauche (ex: b7a8 ou b2a1 ?)
			// Attention : fileDiff = -1 veut dire "vers la gauche" (colonne B -> A)
			else if (std::abs(rankDiff) == 1 && fileDiff == -1)
				encodedTo = 67 + promoType;

			// Capture diagonale droite (ex: a7b8)
			else if (std::abs(rankDiff) == 1 && fileDiff == 1)
				encodedTo = 70 + promoType;

			else throw std::runtime_error("ChessEngine::actionToIdx(): Invalid promotion move");
		}

		return static_cast<uint32_t>(encodedFrom + encodedTo);
	}
}