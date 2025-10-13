#pragma once
#include "ChessTraits.hpp"
#include "../../corelib/AlignedVec.hpp"

#include "Tables.hpp"
#include "../../corelib/util/CompilerHints.hpp"

#include <immintrin.h>

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
    // tzcnt renvoie 0..63 si bb!=0, 64 si bb==0
    unsigned long idx = (unsigned long)(_tzcnt_u64(bb) & 63u);
    bb &= bb - 1;
    return static_cast<int>(idx);
}

// renvoie 0 si x==0, sinon 0xFFFF'FFFF'FFFF'FFFF
static ALWAYS_INLINE constexpr
uint64_t nzmask(uint64_t x) noexcept
{
    // si x≠0 alors (x|-x) a son bit de signe à 1 → ((x|-x)>>63)==1
    // sinon (x|-x)==0 → ((...)>>63)==0
    // on a donc un bit 0/1 qu’on passe à deux’s‑complement pour obtenir 0 ou -1(=all‑ones).
    return -((x | -x) >> 63);
}

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
    static constexpr const auto& bs = BoardStatusFor<StatusFlag>{};

    // Board: encapsulate all bitboards and flags
    struct Board
    {
        map ourPawn, ourKnight, ourBishop, ourRook, ourQueen, ourKing;
        map oppPawn, oppKnight, oppBishop, oppRook, oppQueen, oppKing;

        map ourOcc, oppOcc, occ;

        map enPassantBB;

        ALWAYS_INLINE
            void setBoard(const ObsStateT<ChessTag>& state) noexcept
        {
            if constexpr (bs.isWhite)
            {
                ourPawn = state.elems.whiteBB[0]; ourKnight = state.elems.whiteBB[1];
                ourBishop = state.elems.whiteBB[2]; ourRook = state.elems.whiteBB[3];
                ourQueen = state.elems.whiteBB[4]; ourKing = state.elems.whiteBB[5];

                oppPawn = state.elems.blackBB[0]; oppKnight = state.elems.blackBB[1];
                oppBishop = state.elems.blackBB[2]; oppRook = state.elems.blackBB[3];
                oppQueen = state.elems.blackBB[4]; oppKing = state.elems.blackBB[5];
            }
            else
            {
                ourPawn = state.elems.blackBB[0]; ourKnight = state.elems.blackBB[1];
                ourBishop = state.elems.blackBB[2]; ourRook = state.elems.blackBB[3];
                ourQueen = state.elems.blackBB[4]; ourKing = state.elems.blackBB[5];

                oppPawn = state.elems.whiteBB[0]; oppKnight = state.elems.whiteBB[1];
                oppBishop = state.elems.whiteBB[2]; oppRook = state.elems.whiteBB[3];
                oppQueen = state.elems.whiteBB[4]; oppKing = state.elems.whiteBB[5];
            }

            ourOcc = ourPawn | ourKnight | ourBishop | ourRook | ourQueen | ourKing;
            oppOcc = oppPawn | oppKnight | oppBishop | oppRook | oppQueen | oppKing;
            occ = ourOcc | oppOcc;

            enPassantBB = state.meta.enPassant != 0 ? 1ULL << state.meta.enPassant : 0ULL;
        }
    };
    static inline Board b;

    static inline int pinnerOf[64];

    // Calcul des cases attaquées par l’adversaire
    static ALWAYS_INLINE
        uint64_t computeAttacks() noexcept
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

    // Calcule checkMask et pinners
    static ALWAYS_INLINE
        void computeCheckAndPins(map& checkMask, int& checkCount, int kingSq) noexcept
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

            /*
            // Debug variables
            std::cout << "sq: " << sq << std::endl;
            std::cout << "Idx: " << kingSq * 64 + sq << std::endl;
            ///
            std::cout << "between: " << std::endl;
            dispBoard(between);
            ///
            std::cout << "x: " << std::endl;
            dispBoard(x);
            std::cout << "nonZero: " << std::endl;
            dispBoard(nonZero);
            std::cout << "zeroFriendly: " << std::endl;
            dispBoard(zeroFriendly);
            std::cout << "mask0: " << std::endl;
            dispBoard(mask0);
            std::cout << "exactlyOne: " << std::endl;
            dispBoard(exactlyOne);
            */

            int xIdx = popLSB(x);
            int pinnerIdx = sq * (int)exactlyOne - (1 - (int)exactlyOne);
            pinnerOf[xIdx] = pinnerIdx;

            /*
            // Debug variables
            std::cout << "xIdx: " << xIdx << std::endl;
            std::cout << "pinnerIdx: " << pinnerIdx << std::endl;
            std::cout << "pinnerOf[xIdx]: " << pinnerOf[xIdx] << std::endl;
            */
        }

    }

    // Récupère le rayon de pin entre roi et pièce
    static ALWAYS_INLINE
        map getPinRay(int kingSq, int pieceSq) noexcept
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
        void addLegalMoves(AlignedVec<ActionT<ChessTag>>& out, int pieceSquare, map pieceMask, map promo) noexcept
    {
        while (pieceMask)
        {
            ActionT<ChessTag> action;
            action.setFrom(pieceSquare);
            action.setTo(popLSB(pieceMask));

            UNLIKELY(if (promo))
            {
                for (int i = 1; i < 5; ++i)
                {
                    action.setPromo(i);
                    out.emplace_back(action);
                }
            }
            else
            {
                action.setPromo(0);
                out.emplace_back(action);
            }
        }
    }

public:
    MoveGenerator() = default;
    static void countCheck(const ObsStateT<ChessTag>& state, int& out) noexcept
    {
        b.setBoard(state);

        int kingSq = popLSB(b.ourKing);
        map checkMask = 0;
        out = 0;
        computeCheckAndPins(checkMask, out, kingSq);
    }

    static void generate(const ObsStateT<ChessTag>& state, AlignedVec<ActionT<ChessTag>>& out) noexcept
    {
        b.setBoard(state);

        // KingBB
        map tmp = b.ourKing;
        int kingSq = popLSB(tmp);

        map checkMask = 0;
        int checkCount = 0;
        std::memset(pinnerOf, -1, sizeof(pinnerOf));
        computeCheckAndPins(checkMask, checkCount, kingSq);

        map x = checkMask | (map)(-(int64_t)checkMask);
        map zeroFlag = (x >> 63) ^ 1ULL;
        checkMask |= (map)(-(int64_t)zeroFlag);
        map atkSquares = computeAttacks();

        LIKELY(if (checkCount < 2))
        {
            map singlePush, doublePush, capsL, capsR;
            int signPawn;

            // If White
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

            // Pawn Pseudo-legal moves
            while (singlePush)
            {
                int to = popLSB(singlePush);
                int from = to + signPawn * 8;

                map pawnAtk = 1ULL << to;
                map pinRay = getPinRay(kingSq, from);
                map finalMask = pawnAtk & pinRay;

                map isPromoting;
                if constexpr (bs.isWhite)
                    isPromoting = (finalMask & RANK_8) >> to;
                else
                    isPromoting = (finalMask & RANK_1) >> to;

                addLegalMoves(out, from, finalMask, isPromoting);
            }
            while (doublePush)
            {
                int to = popLSB(doublePush);
                int from = to + signPawn * 16;

                map pawnAtk = 1ULL << to;
                map pinRay = getPinRay(kingSq, from);
                map finalMask = pawnAtk & pinRay;

                addLegalMoves(out, from, finalMask, 0);
            }
            while (capsL)
            {
                int to = popLSB(capsL);
                int from = to + signPawn * 8 + 1;

                map pawnAtk = 1ULL << to;
                map pinMask = getPinRay(kingSq, from);

                if constexpr (bs.hasEnPassant)
                {
                    // a) EP candidate -> flag dans pawnAtk & enPassantBB
                    map epCand = pawnAtk & b.enPassantBB;

                    // b) rowEqMask = all1 si from/8 == kingSq/8, sinon 0
                    int rowDiff = (from ^ kingSq) >> 3;
                    map rowEq = (((map)rowDiff | (map)(-(int64_t)rowDiff)) >> 63) ^ 1ULL;
                    map rowEqMask = (map)(-(int64_t)rowEq);

                    // c) simulate occ2 = occ sans le pion bougeant et sans le pion capturé
                    map pawnSq = 1ULL << from;              // Case du pion
                    map capSq = 1ULL << (from - 1);         // Case du pion capturé
                    map occ2 = b.occ & ~(pawnSq | capSq);

                    // d) blockers = cases entre roi et from après capture
                    map betweenKF = tables.rayBetween[kingSq * 64 + from];
                    map blockers = betweenKF & occ2;

                    // e) noBlockersMask = all1 si blockers == 0, sinon 0
                    map blockZero = (((blockers | (map)(-(int64_t)blockers)) >> 63) ^ 1ULL);
                    map noBlockers = (map)(-(int64_t)blockZero);

                    // f) attacker behind from: atkSquares & (1<<(from-2))
                    map atkSqBit = (atkSquares >> (from - 2)) & 1ULL;
                    map atkMask = (map)(-(int64_t)atkSqBit);

                    // g) cond = epCand & rowEqMask & noBlockers & atkMask
                    map cond = epCand & rowEqMask & noBlockers & atkMask;

                    // h) epInvalidMask = ~(-cond) → 0 if cond!=0, all1 if cond==0
                    map epInvMask = ~(map)(-(int64_t)cond);

                    // i) mise à jour du pinMask
                    pinMask &= epInvMask;
                }

                map finalMask = pawnAtk & pinMask;

                // Promotion
                map isPromoting;
                if constexpr (bs.isWhite)
                    isPromoting = (finalMask & RANK_8) >> to;
                else
                    isPromoting = (finalMask & RANK_1) >> to;

                addLegalMoves(out, from, finalMask, isPromoting);
            }
            while (capsR)
            {
                int to = popLSB(capsR);
                int from = to + signPawn * 8 - 1;

                map pawnAtk = 1ULL << to;
                map pinMask = getPinRay(kingSq, from);

                if constexpr (bs.hasEnPassant)
                {
                    // a) EP candidate -> flag dans pawnAtk & enPassantBB
                    map epCand = pawnAtk & b.enPassantBB;

                    // b) rowEqMask = all1 si from/8 == kingSq/8, sinon 0
                    int rowDiff = (from ^ kingSq) >> 3;
                    map rowEq = (((map)rowDiff | (map)(-(int64_t)rowDiff)) >> 63) ^ 1ULL;
                    map rowEqMask = (map)(-(int64_t)rowEq);

                    // c) simulate occ2 = occ sans le pion bougeant et sans le pion capturé
                    map pawnSq = 1ULL << from;              // Case du pion
                    map capSq = 1ULL << (from + 1);         // Case du pion capturé
                    map occ2 = b.occ & ~(pawnSq | capSq);

                    // d) blockers = cases entre roi et from après capture
                    map betweenKF = tables.rayBetween[kingSq * 64 + from];
                    map blockers = betweenKF & occ2;

                    // e) noBlockersMask = all1 si blockers == 0, sinon 0
                    map blockZero = (((blockers | (map)(-(int64_t)blockers)) >> 63) ^ 1ULL);
                    map noBlockers = (map)(-(int64_t)blockZero);

                    // f) attacker behind from: atkSquares & (1<<(from+2))
                    map atkSqBit = (atkSquares >> (from + 2)) & 1ULL;
                    map atkMask = (map)(-(int64_t)atkSqBit);

                    // g) cond = epCand & rowEqMask & noBlockers & atkMask
                    map cond = epCand & rowEqMask & noBlockers & atkMask;

                    // h) epInvalidMask = ~(-cond) → 0 if cond!=0, all1 if cond==0
                    map epInvMask = ~(map)(-(int64_t)cond);

                    // i) mise à jour du pinMask
                    pinMask &= epInvMask;
                }

                map finalMask = pawnAtk & pinMask;

                map isPromoting;
                if constexpr (bs.isWhite)
                    isPromoting = (finalMask & RANK_8) >> to;
                else
                    isPromoting = (finalMask & RANK_1) >> to;

                addLegalMoves(out, from, finalMask, isPromoting);
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

                addLegalMoves(out, knightSq, finalMask, 0);
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

                map pinRay = getPinRay(kingSq, bishopSq);
                map finalMask = bishopAtk & pinRay;

                addLegalMoves(out, bishopSq, finalMask, 0);
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

                map pinRay = getPinRay(kingSq, rookSq);
                map finalMask = rookAtk & pinRay;

                /*
                std::cout << "rookAtk:" << std::endl;
                dispBoard(rookAtk);
                std::cout << "pinRay:" << std::endl;
                dispBoard(pinRay);
                */

                addLegalMoves(out, rookSq, finalMask, 0);
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

        map kingAtk = (tables.kingMasks[kingSq] | addCastlingSq) & (~b.ourOcc) & (~atkSquares);
        addLegalMoves(out, kingSq, kingAtk, 0);
    }

    static void apply(const ActionT<ChessTag>& move, ObsStateT<ChessTag>& out) noexcept
    {
        int start = move.from();
        int dest = move.to();
        int promotion = move.promo();

        map startMask = 1ULL << start;
        map destMask = 1ULL << dest;
        bool whitePlayed = !out.meta.trait;

        int offsetPromo;
        if constexpr (bs.isWhite)
            offsetPromo = 5 - promotion;
        else
            offsetPromo = 11 - promotion;

        const map oldEnPassant = 1ULL << out.meta.enPassant;
        out.meta.enPassant = 0;

		bool pawnOrCapture = false;
        for (int ch = 0; ch < 12; ++ch)
        {
            map& stateRef = (ch < 6)
                ? out.elems.whiteBB[ch]
                : out.elems.blackBB[ch - 6];
            map bb = stateRef;

            // capture
            if (bb & destMask)
                pawnOrCapture = true;

            if (bb & startMask)
            {
                // pawn move
                if (ch == 0 || ch == 6)
                    pawnOrCapture = true;

                stateRef &= ~startMask;
                if (!promotion)
                    stateRef |= destMask;

                // White Castling
                if constexpr (bs.wCastlingK)
                {
                    if (start == 4)
                    {
                        out.meta.castlingRights &= 0b1100;
                        if (dest - start == 2)
                        {
                            out.elems.whiteBB[3] &= ~(1ULL << 7);
                            out.elems.whiteBB[3] |= 1ULL << 5;
                        }
                    }
                    if ((start == 7) || (dest == 7))
                        out.meta.castlingRights &= 0b1110;
                }
                if constexpr (bs.wCastlingQ)
                {
                    if (start == 4)
                    {
                        out.meta.castlingRights &= 0b1100;
                        if (start - dest == 2)
                        {
                            out.elems.whiteBB[3] &= ~1ULL;
                            out.elems.whiteBB[3] |= 1ULL << 3;
                        }
                    }
                    if ((start == 0) || (dest == 0))
                        out.meta.castlingRights &= 0b1101;
                }

                // Black Castling
                if constexpr (bs.bCastlingK)
                {
                    if (start == 60)
                    {
                        out.meta.castlingRights &= 0b0011;
                        if (dest - start == 2)
                        {
                            out.elems.blackBB[3] &= ~(1ULL << 63);
                            out.elems.blackBB[3] |= 1ULL << 61;
                        }
                    }
                    if ((start == 63) || (dest == 63))
                        out.meta.castlingRights &= 0b1011;
                }
                if constexpr (bs.bCastlingQ)
                {
                    if (start == 60)
                    {
                        out.meta.castlingRights &= 0b0011;
                        if (start - dest == 2)
                        {
                            out.elems.blackBB[3] &= ~(1ULL << 56);
                            out.elems.blackBB[3] |= 1ULL << 59;
                        }
                    }
                    if ((start == 56) || (dest == 56))
                        out.meta.castlingRights &= 0b0111;
                }

                // Apply En‑Passant
                if constexpr (bs.hasEnPassant)
                {
                    if constexpr (bs.isWhite)
                    {
                        if ((destMask == oldEnPassant) && (ch == 0))
                            out.elems.blackBB[0] &= ~(oldEnPassant >> 8);
                    }
                    else
                    {
                        if ((destMask == oldEnPassant) && (ch == 6))
                            out.elems.whiteBB[0] &= ~(oldEnPassant << 8);
                    }
                }

                // Create new En-Passant square
                if constexpr (bs.isWhite)
                {
                    if ((dest - start == 16) && (ch == 0))
                    {
                        map startMaskShifted = startMask << 8;
                        out.meta.enPassant = static_cast<uint8_t>(popLSB(startMaskShifted));
                    }
                }
                else
                {
                    if ((start - dest == 16) && (ch == 6))
                    {
                        map startMaskShifted = startMask >> 8;
                        out.meta.enPassant = static_cast<uint8_t>(popLSB(startMaskShifted));
                    }
                }
            }
            stateRef &= ~(bb & destMask);
        }

        if (promotion)
        {
            if (offsetPromo < 6)
                out.elems.whiteBB[offsetPromo] |= destMask;
            else
                out.elems.blackBB[offsetPromo - 6] |= destMask;
        }

		// Halfmove clock update
        if (pawnOrCapture) out.meta.halfmoveClock = 0;
        else out.meta.halfmoveClock++;

		// Fullmove number increment
		if (out.meta.trait == 1) out.meta.fullmoveNumber++;

        // Trait switch
        out.meta.trait ^= 1;
    }
};