#pragma once
#include "ChessTraits.hpp"
#include "../../corelib/interfaces/IEngine.hpp"
#include "FenParser.hpp"
#include <chrono>

struct PerftTest
{
    std::string name;
    std::string fen;
    int depth;
    uint64_t expected;
};

class PerftTool
{
private:
    std::shared_ptr<IEngine<ChessTag>> m_engine;

    static constexpr const char* kSquaresName[64] =
    {
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"
    };
    static constexpr const char* kPromosLetter[5] = { "", "q", "r", "b", "n" };

public:
    PerftTool();
    PerftTool(std::shared_ptr<IEngine<ChessTag>> engine);

    uint64_t perft(const ObsStateT<ChessTag>& root, int maxDepth);

    void runNormal(const AlignedVec<PerftTest>& normalTests);
    void runDivide(const AlignedVec<PerftTest>& divideTests);
};