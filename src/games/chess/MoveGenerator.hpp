#pragma once
#include <bit>
#include <cstring>

#include "ChessTypes.hpp"
#include "Tables.hpp"
#include "../../corelib/util/CompilerHints.hpp"

namespace Chess
{
    using map = uint64_t;

    // initialisation compile‑time des magic tables
    alignas(64) static constexpr Tables tables{};

    static constexpr map FILE_A = 0x0101010101010101ULL;
    static constexpr map FILE_H = 0x8080808080808080ULL;
    static constexpr map RANK_1 = 0x00000000000000FFULL;
    static constexpr map RANK_8 = 0xFF00000000000000ULL;
    static constexpr map RANK_4 = 0x00000000FF000000ULL;
    static constexpr map RANK_5 = 0x000000FF00000000ULL;

    static ALWAYS_INLINE
        int popLSB(uint64_t& bb) noexcept
    {
        if (bb == 0) return 0;

        int idx = std::countr_zero(bb);
        bb &= bb - 1;
        return idx;
    }

    // renvoie 0 si x==0, sinon 0xFFFF'FFFF'FFFF'FFFF
    static ALWAYS_INLINE constexpr
        uint64_t nzmask(uint64_t x) noexcept
    {
        return -((x | -x) >> 63);
    }

    struct StateBB
    {
        uint64_t whiteBB[6]; // P, N, B, R, Q, K
        uint64_t blackBB[6]; // P, N, B, R, Q, K
        uint8_t enPassant;
    };

    template<uint8_t Status>
    struct BoardStatusFor
    {
        static constexpr bool isWhite = Status & 0b000001;
        static constexpr bool wCastlingK = Status & 0b000010;
        static constexpr bool wCastlingQ = Status & 0b000100;
        static constexpr bool bCastlingK = Status & 0b001000;
        static constexpr bool bCastlingQ = Status & 0b010000;
        static constexpr bool hasEnPassant = Status & 0b100000;
    };

    // isWhite[0], castlingRights[1-4], enPassant[5]
    template<uint8_t StatusFlag>
    class MoveGenerator
    {
    private:
        USING_GAME_TYPES(ChessTypes);

        static constexpr const auto& bs = BoardStatusFor<StatusFlag>{};

        // Board: encapsulate all bitboards and flags
        struct Board
        {
            map ourPawn, ourKnight, ourBishop, ourRook, ourQueen, ourKing;
            map oppPawn, oppKnight, oppBishop, oppRook, oppQueen, oppKing;

            map ourOcc, oppOcc, occ;

            map enPassantBB;

            ALWAYS_INLINE
                void setBoard(const StateBB& state) noexcept
            {
                if constexpr (bs.isWhite)
                {
                    ourPawn = state.whiteBB[0]; ourKnight = state.whiteBB[1];
                    ourBishop = state.whiteBB[2]; ourRook = state.whiteBB[3];
                    ourQueen = state.whiteBB[4]; ourKing = state.whiteBB[5];

                    oppPawn = state.blackBB[0]; oppKnight = state.blackBB[1];
                    oppBishop = state.blackBB[2]; oppRook = state.blackBB[3];
                    oppQueen = state.blackBB[4]; oppKing = state.blackBB[5];
                }
                else
                {
                    ourPawn = state.blackBB[0]; ourKnight = state.blackBB[1];
                    ourBishop = state.blackBB[2]; ourRook = state.blackBB[3];
                    ourQueen = state.blackBB[4]; ourKing = state.blackBB[5];

                    oppPawn = state.whiteBB[0]; oppKnight = state.whiteBB[1];
                    oppBishop = state.whiteBB[2]; oppRook = state.whiteBB[3];
                    oppQueen = state.whiteBB[4]; oppKing = state.whiteBB[5];
                }

                ourOcc = ourPawn | ourKnight | ourBishop | ourRook | ourQueen | ourKing;
                oppOcc = oppPawn | oppKnight | oppBishop | oppRook | oppQueen | oppKing;
                occ = ourOcc | oppOcc;

                enPassantBB = (state.enPassant < 64) ? (1ULL << state.enPassant) : 0ULL;
            }
        };

        // Calcul des cases attaquées par l’adversaire
        static ALWAYS_INLINE
            uint64_t computeAttacks(const Board& b) noexcept
        {
            map atk = 0ULL;

            // pions
            if constexpr (bs.isWhite)
                atk |= ((b.oppPawn & ~FILE_A) >> 9) | ((b.oppPawn & ~FILE_H) >> 7);
            else
                atk |= ((b.oppPawn & ~FILE_A) << 7) | ((b.oppPawn & ~FILE_H) << 9);

            // cavaliers
            map tmp = b.oppKnight;
            while (tmp)
            {
                int sq = popLSB(tmp);
                atk |= tables.knightMasks[sq];
            }

            // fous & dames diagonaux
            map occWOK = b.occ & (~b.ourKing);
            tmp = b.oppBishop | b.oppQueen;
            while (tmp)
            {
                int sq = popLSB(tmp);
                int idx = (int)(((occWOK & tables.bishopMasks[sq]) * tables.bishopMagicNumbers[sq]) >> tables.bishopShifts[sq]);
                atk |= tables.bishopAttacks[tables.bishopOffsets[sq] + idx];
            }

            // tours & dames orthogonaux
            tmp = b.oppRook | b.oppQueen;
            while (tmp)
            {
                int sq = popLSB(tmp);
                int idx = (int)(((occWOK & tables.rookMasks[sq]) * tables.rookMagicNumbers[sq]) >> tables.rookShifts[sq]);
                atk |= tables.rookAttacks[tables.rookOffsets[sq] + idx];
            }

            // roi adverse (un seul bit)
            tmp = b.oppKing;
            int idx = popLSB(tmp);
            atk |= tables.kingMasks[idx];

            return atk;
        }

        // Calcule checkMask et pinners (OPTIMISATION int8_t)
        static ALWAYS_INLINE
            void computeCheckAndPins(const Board& b, int8_t* pinnerOf,
                map& checkMask, int& checkCount, int kingSq) noexcept
        {
            // Pawns
            map pawnAttacks;
            if constexpr (bs.isWhite)
                pawnAttacks = (b.ourKing & ~FILE_A) << 7 | (b.ourKing & ~FILE_H) << 9;
            else pawnAttacks = (b.ourKing & ~FILE_A) >> 9 | (b.ourKing & ~FILE_H) >> 7;

            map pawnCheckers = pawnAttacks & b.oppPawn;
            map knightCheckers = tables.knightMasks[kingSq] & b.oppKnight;
            map kingChecker = tables.kingMasks[kingSq] & b.oppKing;

            checkMask |= pawnCheckers | knightCheckers | kingChecker;
            checkCount += (int)bool(pawnCheckers)
                + (int)bool(knightCheckers)
                + (int)bool(kingChecker);

            // Sliders Pieces
            // Bishops & Queens
            int bishopIdx = (int)(((b.oppOcc & tables.bishopMasks[kingSq])
                * tables.bishopMagicNumbers[kingSq])
                >> tables.bishopShifts[kingSq]);
            map bishopAtk = tables.bishopAttacks[tables.bishopOffsets[kingSq] + bishopIdx];

            // Rooks & Queens
            int rookIdx = (int)(((b.oppOcc & tables.rookMasks[kingSq])
                * tables.rookMagicNumbers[kingSq])
                >> tables.rookShifts[kingSq]);
            map rookAtk = tables.rookAttacks[tables.rookOffsets[kingSq] + rookIdx];

            // Slider Checkers with no enemy pieces between
            map bishopCheckers = bishopAtk & (b.oppBishop | b.oppQueen);
            map rookCheckers = rookAtk & (b.oppRook | b.oppQueen);
            map sliderCheckers = bishopCheckers | rookCheckers;

            while (sliderCheckers)
            {
                int sq = popLSB(sliderCheckers);

                // Ray between king and a slider piece with no enemy pieces between
                map between = tables.rayBetween[kingSq * 64 + sq];

                // 1) Friendly pieces dans between
                map x = (between & ~(1ULL << sq)) & b.ourOcc;

                map nonZero = (x | (map)(-(int64_t)x)) >> 63;
                map zeroFriendly = nonZero ^ 1ULL;

                // 4) mask0 = all 1s si zeroFriendly==1, sinon 0
                map mask0 = (map)(-(int64_t)zeroFriendly);

                // 5) OR du ray only when no friend
                checkMask |= mask0 & between;
                checkCount += (int)zeroFriendly;

                map y = x & (x - 1);
                map nonZeroX = (x | (map)(-(int64_t)x)) >> 63;
                map nonZeroY = (y | (map)(-(int64_t)y)) >> 63;
                map exactlyOne = nonZeroX & (nonZeroY ^ 1ULL);

                int xIdx = popLSB(x);
                int pinnerIdx = sq * (int)exactlyOne - (1 - (int)exactlyOne);
                pinnerOf[xIdx] = static_cast<int8_t>(pinnerIdx);
            }
        }

        // Récupère le rayon de pin entre roi et pièce (Accepte int8_t)
        static ALWAYS_INLINE
            map getPinRay(const int8_t* pinnerOf, int kingSq, int pieceSq) noexcept
        {
            int pinnerSq = pinnerOf[pieceSq];
            int rawIndex = kingSq * 64 + pinnerSq;
            map v = (map)(pinnerSq + 1);
            map t = (map)((int64_t)(v - 1) >> 63);
            map mask = ~t;
            int index = rawIndex & (int)mask;

            return tables.rayBetween[index];
        }

        // Add legal moves
        static ALWAYS_INLINE
            void addLegalMoves(ActionList& actionList, uint8_t pieceId, int pieceSquare, map pieceMask, map promo) noexcept
        {
            while (pieceMask)
            {
                Action action;
                uint8_t ownerId = bs.isWhite ? 0 : 1;
                int to = popLSB(pieceMask);

                if (UNLIKELY(promo))
                {
                    for (int i = 1; i < 5; ++i)
                    {
                        action.configure(pieceId, ownerId, pieceSquare, to, static_cast<float>(i));
                        actionList.push_back(action);
                    }
                }
                else
                {
                    action.configure(pieceId, ownerId, pieceSquare, to, 0.0f);
                    actionList.push_back(action);
                }
            }
        }

    public:
        MoveGenerator() = default;

        static void countCheck(const StateBB& state, int& out) noexcept
        {
            Board b;
            b.setBoard(state);

            int kingSq = popLSB(b.ourKing);
            map checkMask = 0;
            out = 0;

            int8_t pinnerOf[64];
            std::memset(pinnerOf, -1, sizeof(pinnerOf));

            computeCheckAndPins(b, pinnerOf, checkMask, out, kingSq);
        }

        // ====================================================================
        // EARLY EXIT OPTIMIZATION : bool hasAnyLegalMove()
        // Permet de statuer sur le Mat/Pat sans générer tous les coups.
        // ====================================================================
        static bool hasAnyLegalMove(const StateBB& state) noexcept
        {
            Board b;
            b.setBoard(state);

            map tmp = b.ourKing;
            int kingSq = popLSB(tmp);

            map checkMask = 0;
            int checkCount = 0;

            int8_t pinnerOf[64];
            std::memset(pinnerOf, -1, sizeof(pinnerOf));

            computeCheckAndPins(b, pinnerOf, checkMask, checkCount, kingSq);
            map atkSquares = computeAttacks(b);

            // 1. Mouvements du Roi (Rapide, sauve du calcul si pas mat/pat)
            map kingAtk = tables.kingMasks[kingSq] & (~b.ourOcc) & (~atkSquares);
            if (kingAtk) return true;

            // 2. Double Échec (Seul le roi peut bouger, or on vient de vérifier qu'il ne peut pas)
            if (checkCount >= 2) return false;

            // Normalisation du checkMask (Branch Predictor friendly)
            if (!checkMask) checkMask = ~0ULL;

            // 3. Cavaliers (Rapide, pas de calcul de rayon magique)
            tmp = b.ourKnight;
            while (tmp) {
                int knightSq = popLSB(tmp);
                map knightAtk = tables.knightMasks[knightSq] & (~b.ourOcc) & checkMask;
                int v = pinnerOf[knightSq] + 1;
                map pinMask = (map)((int64_t)(v - 1) >> 63);
                if (knightAtk & pinMask) return true;
            }

            // 4. Pions (Très probables en milieu de partie)
            int signPawn;
            map singlePush, doublePush, capsL, capsR;

            if constexpr (bs.isWhite) {
                signPawn = -1;
                singlePush = (b.ourPawn << 8) & (~b.occ) & checkMask;
                doublePush = (singlePush << 8) & (~b.occ) & RANK_4 & checkMask;
                capsL = ((b.ourPawn & ~FILE_A) << 7) & (b.oppOcc | b.enPassantBB) & checkMask;
                capsR = ((b.ourPawn & ~FILE_H) << 9) & (b.oppOcc | b.enPassantBB) & checkMask;
            }
            else {
                signPawn = 1;
                singlePush = (b.ourPawn >> 8) & (~b.occ) & checkMask;
                doublePush = (singlePush >> 8) & (~b.occ) & RANK_5 & checkMask;
                capsL = ((b.ourPawn & ~FILE_A) >> 9) & (b.oppOcc | b.enPassantBB) & checkMask;
                capsR = ((b.ourPawn & ~FILE_H) >> 7) & (b.oppOcc | b.enPassantBB) & checkMask;
            }

            map tmpPawn = singlePush;
            while (tmpPawn) {
                int to = popLSB(tmpPawn);
                int from = to + signPawn * 8;
                if ((1ULL << to) & getPinRay(pinnerOf, kingSq, from)) return true;
            }

            tmpPawn = doublePush;
            while (tmpPawn) {
                int to = popLSB(tmpPawn);
                int from = to + signPawn * 16;
                if ((1ULL << to) & getPinRay(pinnerOf, kingSq, from)) return true;
            }

            tmpPawn = capsL;
            while (tmpPawn) {
                int to = popLSB(tmpPawn);
                int from = to + signPawn * 8 + 1;
                map pawnAtk = 1ULL << to;
                map pinMask = getPinRay(pinnerOf, kingSq, from);

                if constexpr (bs.hasEnPassant) {
                    map epCand = pawnAtk & b.enPassantBB;
                    if (epCand) {
                        int rowDiff = (from ^ kingSq) >> 3;
                        map rowEq = (((map)rowDiff | (map)(-(int64_t)rowDiff)) >> 63) ^ 1ULL;
                        map rowEqMask = (map)(-(int64_t)rowEq);
                        map pawnSq = 1ULL << from;
                        map capSq = 1ULL << (from - 1);
                        map occ2 = b.occ & ~(pawnSq | capSq);
                        map blockers = tables.rayBetween[kingSq * 64 + from] & occ2;
                        map blockZero = (((blockers | (map)(-(int64_t)blockers)) >> 63) ^ 1ULL);
                        map noBlockers = (map)(-(int64_t)blockZero);
                        map atkSqBit = (atkSquares >> (from - 2)) & 1ULL;
                        map atkMask = (map)(-(int64_t)atkSqBit);
                        map cond = epCand & rowEqMask & noBlockers & atkMask;
                        map epInvMask = ~(map)(-(int64_t)cond);
                        pinMask &= epInvMask;
                    }
                }
                if (pawnAtk & pinMask) return true;
            }

            tmpPawn = capsR;
            while (tmpPawn) {
                int to = popLSB(tmpPawn);
                int from = to + signPawn * 8 - 1;
                map pawnAtk = 1ULL << to;
                map pinMask = getPinRay(pinnerOf, kingSq, from);

                if constexpr (bs.hasEnPassant) {
                    map epCand = pawnAtk & b.enPassantBB;
                    if (epCand) {
                        int rowDiff = (from ^ kingSq) >> 3;
                        map rowEq = (((map)rowDiff | (map)(-(int64_t)rowDiff)) >> 63) ^ 1ULL;
                        map rowEqMask = (map)(-(int64_t)rowEq);
                        map pawnSq = 1ULL << from;
                        map capSq = 1ULL << (from + 1);
                        map occ2 = b.occ & ~(pawnSq | capSq);
                        map blockers = tables.rayBetween[kingSq * 64 + from] & occ2;
                        map blockZero = (((blockers | (map)(-(int64_t)blockers)) >> 63) ^ 1ULL);
                        map noBlockers = (map)(-(int64_t)blockZero);
                        map atkSqBit = (atkSquares >> (from + 2)) & 1ULL;
                        map atkMask = (map)(-(int64_t)atkSqBit);
                        map cond = epCand & rowEqMask & noBlockers & atkMask;
                        map epInvMask = ~(map)(-(int64_t)cond);
                        pinMask &= epInvMask;
                    }
                }
                if (pawnAtk & pinMask) return true;
            }

            // 5. Pièces glissantes (Fous & Dames)
            tmp = b.ourBishop | b.ourQueen;
            while (tmp) {
                int sq = popLSB(tmp);
                int idx = (int)(((b.occ & tables.bishopMasks[sq]) * tables.bishopMagicNumbers[sq]) >> tables.bishopShifts[sq]);
                map atk = tables.bishopAttacks[tables.bishopOffsets[sq] + idx] & (~b.ourOcc) & checkMask;
                if (atk & getPinRay(pinnerOf, kingSq, sq)) return true;
            }

            // Pièces glissantes (Tours & Dames)
            tmp = b.ourRook | b.ourQueen;
            while (tmp) {
                int sq = popLSB(tmp);
                int idx = (int)(((b.occ & tables.rookMasks[sq]) * tables.rookMagicNumbers[sq]) >> tables.rookShifts[sq]);
                map atk = tables.rookAttacks[tables.rookOffsets[sq] + idx] & (~b.ourOcc) & checkMask;
                if (atk & getPinRay(pinnerOf, kingSq, sq)) return true;
            }

            // 6. Roque (Vérifié en dernier car coûteux et rarement l'unique coup possible)
            if (checkCount == 0 && (bs.wCastlingK || bs.wCastlingQ || bs.bCastlingK || bs.bCastlingQ)) {
                const int startSq = bs.isWhite ? 4 : 60;
                if (kingSq == startSq) {
                    if constexpr (bs.isWhite) {
                        if constexpr (bs.wCastlingK) {
                            if (!(((b.occ | atkSquares) & 0x60) || (~b.ourRook & 0x80))) return true;
                        }
                        if constexpr (bs.wCastlingQ) {
                            if (!(((b.occ & 0xE) | (atkSquares & 0xC)) || (~b.ourRook & 0x1))) return true;
                        }
                    }
                    else {
                        if constexpr (bs.bCastlingK) {
                            if (!(((b.occ | atkSquares) & 0x6000000000000000ULL) || (~b.ourRook & 0x8000000000000000ULL))) return true;
                        }
                        if constexpr (bs.bCastlingQ) {
                            if (!(((b.occ & 0x0E00000000000000ULL) | (atkSquares & 0x0C00000000000000ULL)) || (~b.ourRook & 0x0100000000000000ULL))) return true;
                        }
                    }
                }
            }

            return false;
        }

        static void generate(const StateBB& state, ActionList& out) noexcept
        {
            Board b;
            b.setBoard(state);

            map tmp = b.ourKing;
            int kingSq = popLSB(tmp);

            map checkMask = 0;
            int checkCount = 0;

            int8_t pinnerOf[64];
            std::memset(pinnerOf, -1, sizeof(pinnerOf));
            computeCheckAndPins(b, pinnerOf, checkMask, checkCount, kingSq);

            // OPTIMISATION : Prédiction de branche (1 seul CPU jump contre 4 bit-shifts lourds)
            if (!checkMask) checkMask = ~0ULL;

            map atkSquares = computeAttacks(b);

            if (LIKELY(checkCount < 2))
            {
                map singlePush, doublePush, capsL, capsR;
                int signPawn;

                if constexpr (bs.isWhite)
                {
                    signPawn = -1;
                    singlePush = (b.ourPawn << 8) & (~b.occ);
                    doublePush = (singlePush << 8) & (~b.occ) & RANK_4 & checkMask;
                    singlePush &= checkMask;

                    capsL = ((b.ourPawn & ~FILE_A) << 7)
                        & (b.oppOcc | b.enPassantBB)
                        & checkMask;
                    capsR = ((b.ourPawn & ~FILE_H) << 9)
                        & (b.oppOcc | b.enPassantBB)
                        & checkMask;
                }
                else
                {
                    signPawn = 1;
                    singlePush = (b.ourPawn >> 8) & (~b.occ);
                    doublePush = (singlePush >> 8) & (~b.occ) & RANK_5 & checkMask;
                    singlePush &= checkMask;

                    capsL = ((b.ourPawn & ~FILE_A) >> 9)
                        & (b.oppOcc | b.enPassantBB)
                        & checkMask;
                    capsR = ((b.ourPawn & ~FILE_H) >> 7)
                        & (b.oppOcc | b.enPassantBB)
                        & checkMask;
                }

                while (singlePush)
                {
                    int to = popLSB(singlePush);
                    int from = to + signPawn * 8;

                    map pawnAtk = 1ULL << to;
                    map pinRay = getPinRay(pinnerOf, kingSq, from);
                    map finalMask = pawnAtk & pinRay;

                    map isPromoting;
                    if constexpr (bs.isWhite)
                        isPromoting = (finalMask & RANK_8) >> to;
                    else
                        isPromoting = (finalMask & RANK_1) >> to;

                    addLegalMoves(out, PAWN, from, finalMask, isPromoting);
                }
                while (doublePush)
                {
                    int to = popLSB(doublePush);
                    int from = to + signPawn * 16;

                    map pawnAtk = 1ULL << to;
                    map pinRay = getPinRay(pinnerOf, kingSq, from);
                    map finalMask = pawnAtk & pinRay;

                    addLegalMoves(out, PAWN, from, finalMask, 0);
                }
                while (capsL)
                {
                    int to = popLSB(capsL);
                    int from = to + signPawn * 8 + 1;

                    map pawnAtk = 1ULL << to;
                    map pinMask = getPinRay(pinnerOf, kingSq, from);

                    if constexpr (bs.hasEnPassant)
                    {
                        map epCand = pawnAtk & b.enPassantBB;
                        // OPTIMISATION : Ne calcule les masques EP que si le coup EST une prise en passant !
                        if (epCand) {
                            int rowDiff = (from ^ kingSq) >> 3;
                            map rowEq = (((map)rowDiff | (map)(-(int64_t)rowDiff)) >> 63) ^ 1ULL;
                            map rowEqMask = (map)(-(int64_t)rowEq);

                            map pawnSq = 1ULL << from;
                            map capSq = 1ULL << (from - 1);
                            map occ2 = b.occ & ~(pawnSq | capSq);

                            map blockers = tables.rayBetween[kingSq * 64 + from] & occ2;
                            map blockZero = (((blockers | (map)(-(int64_t)blockers)) >> 63) ^ 1ULL);
                            map noBlockers = (map)(-(int64_t)blockZero);

                            map atkSqBit = (atkSquares >> (from - 2)) & 1ULL;
                            map atkMask = (map)(-(int64_t)atkSqBit);

                            map cond = epCand & rowEqMask & noBlockers & atkMask;
                            map epInvMask = ~(map)(-(int64_t)cond);

                            pinMask &= epInvMask;
                        }
                    }

                    map finalMask = pawnAtk & pinMask;

                    map isPromoting;
                    if constexpr (bs.isWhite)
                        isPromoting = (finalMask & RANK_8) >> to;
                    else
                        isPromoting = (finalMask & RANK_1) >> to;

                    addLegalMoves(out, PAWN, from, finalMask, isPromoting);
                }
                while (capsR)
                {
                    int to = popLSB(capsR);
                    int from = to + signPawn * 8 - 1;

                    map pawnAtk = 1ULL << to;
                    map pinMask = getPinRay(pinnerOf, kingSq, from);

                    if constexpr (bs.hasEnPassant)
                    {
                        map epCand = pawnAtk & b.enPassantBB;
                        // OPTIMISATION : Ne calcule les masques EP que si le coup EST une prise en passant !
                        if (epCand) {
                            int rowDiff = (from ^ kingSq) >> 3;
                            map rowEq = (((map)rowDiff | (map)(-(int64_t)rowDiff)) >> 63) ^ 1ULL;
                            map rowEqMask = (map)(-(int64_t)rowEq);

                            map pawnSq = 1ULL << from;
                            map capSq = 1ULL << (from + 1);
                            map occ2 = b.occ & ~(pawnSq | capSq);

                            map blockers = tables.rayBetween[kingSq * 64 + from] & occ2;
                            map blockZero = (((blockers | (map)(-(int64_t)blockers)) >> 63) ^ 1ULL);
                            map noBlockers = (map)(-(int64_t)blockZero);

                            map atkSqBit = (atkSquares >> (from + 2)) & 1ULL;
                            map atkMask = (map)(-(int64_t)atkSqBit);

                            map cond = epCand & rowEqMask & noBlockers & atkMask;
                            map epInvMask = ~(map)(-(int64_t)cond);

                            pinMask &= epInvMask;
                        }
                    }

                    map finalMask = pawnAtk & pinMask;

                    map isPromoting;
                    if constexpr (bs.isWhite)
                        isPromoting = (finalMask & RANK_8) >> to;
                    else
                        isPromoting = (finalMask & RANK_1) >> to;

                    addLegalMoves(out, PAWN, from, finalMask, isPromoting);
                }

                // Knight Pseudo-legal moves
                tmp = b.ourKnight;
                while (tmp)
                {
                    int knightSq = popLSB(tmp);
                    map knightAtk = tables.knightMasks[knightSq] & (~b.ourOcc)
                        & checkMask;

                    int v = pinnerOf[knightSq] + 1;
                    map pinMask = (map)((int64_t)(v - 1) >> 63);
                    map finalMask = knightAtk & pinMask;

                    addLegalMoves(out, KNIGHT, knightSq, finalMask, 0);
                }

                // Bishop and Queen legal moves
                tmp = b.ourBishop | b.ourQueen;
                while (tmp)
                {
                    int bishopSq = popLSB(tmp);
                    int bishopIdx = (int)(((b.occ & tables.bishopMasks[bishopSq])
                        * tables.bishopMagicNumbers[bishopSq])
                        >> tables.bishopShifts[bishopSq]);
                    map bishopAtk = tables.bishopAttacks[tables.bishopOffsets[bishopSq] + bishopIdx]
                        & (~b.ourOcc) & checkMask;

                    map pinRay = getPinRay(pinnerOf, kingSq, bishopSq);
                    map finalMask = bishopAtk & pinRay;

                    bool isQueen = (b.ourQueen & (1ULL << bishopSq)) != 0;

                    addLegalMoves(out, isQueen ? QUEEN : BISHOP, bishopSq, finalMask, 0);
                }

                // Rook and Queen legal moves
                tmp = b.ourRook | b.ourQueen;
                while (tmp)
                {
                    int rookSq = popLSB(tmp);
                    int rookIdx = (int)(((b.occ & tables.rookMasks[rookSq])
                        * tables.rookMagicNumbers[rookSq])
                        >> tables.rookShifts[rookSq]);
                    map rookAtk = tables.rookAttacks[tables.rookOffsets[rookSq] + rookIdx]
                        & (~b.ourOcc) & checkMask;

                    map pinRay = getPinRay(pinnerOf, kingSq, rookSq);
                    map finalMask = rookAtk & pinRay;

                    bool isQueen = (b.ourQueen & (1ULL << rookSq)) != 0;

                    addLegalMoves(out, isQueen ? QUEEN : ROOK, rookSq, finalMask, 0);
                }
            }

            map maskKingCheckCastling = 0;
            if constexpr (bs.wCastlingK || bs.wCastlingQ || bs.bCastlingK || bs.bCastlingQ)
            {
                map check1 = checkCount & 0x1;
                map check2 = (checkCount & 0x2) >> 1;
                maskKingCheckCastling = check1 | check2;
            }

            map addCastlingSq = 0;
            const int startSq = bs.isWhite ? 4 : 60;

            if (kingSq == startSq)
            {
                if constexpr (bs.isWhite)
                {
                    if constexpr (bs.wCastlingK)
                    {
                        map mask1 = ((b.occ | atkSquares) & 0x20) >> 5;
                        map mask2 = ((b.occ | atkSquares) & 0x40) >> 6;
                        map maskRook = (~b.ourRook & 0x80) >> 7;

                        map mask = mask1 | mask2 | maskRook | maskKingCheckCastling;
                        map maskAll = (map)(-(int64_t)mask);
                        addCastlingSq |= 0x40 & ~maskAll;
                    }
                    if constexpr (bs.wCastlingQ)
                    {
                        map mask1 = (b.occ & 0x2) >> 1;
                        map mask2 = ((b.occ | atkSquares) & 0x4) >> 2;
                        map mask3 = ((b.occ | atkSquares) & 0x8) >> 3;
                        map maskRook = (~b.ourRook & 0x1);

                        map mask = mask1 | mask2 | mask3 | maskRook | maskKingCheckCastling;
                        map maskAll = (map)(-(int64_t)mask);
                        addCastlingSq |= 0x4 & ~maskAll;
                    }
                }
                else
                {
                    if constexpr (bs.bCastlingK)
                    {
                        map mask1 = ((b.occ | atkSquares) & 0x2000000000000000) >> 61;
                        map mask2 = ((b.occ | atkSquares) & 0x4000000000000000) >> 62;
                        map maskRook = (~b.ourRook & 0x8000000000000000) >> 63;

                        map mask = mask1 | mask2 | maskRook | maskKingCheckCastling;
                        map maskAll = (map)(-(int64_t)mask);
                        addCastlingSq |= 0x4000000000000000 & ~maskAll;
                    }
                    if constexpr (bs.bCastlingQ)
                    {
                        map mask1 = (b.occ & 0x0200000000000000) >> 57;
                        map mask2 = ((b.occ | atkSquares) & 0x0400000000000000) >> 58;
                        map mask3 = ((b.occ | atkSquares) & 0x0800000000000000) >> 59;
                        map maskRook = (~b.ourRook & 0x0100000000000000) >> 56;

                        map mask = mask1 | mask2 | mask3 | maskRook | maskKingCheckCastling;
                        map maskAll = (map)(-(int64_t)mask);
                        addCastlingSq |= 0x0400000000000000 & ~maskAll;
                    }
                }
            }

            map kingAtk = (tables.kingMasks[kingSq] | addCastlingSq) & (~b.ourOcc) & (~atkSquares);
            addLegalMoves(out, KING, kingSq, kingAtk, 0);
        }
    };
}