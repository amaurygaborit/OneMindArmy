#include "PerftTool.hpp"

#include <iostream>
#include <cstring>

PerftTool::PerftTool()
{
}

PerftTool::PerftTool(std::shared_ptr<IEngine<ChessTag>> engine) : m_engine(std::move(engine))
{
}

uint64_t PerftTool::perft(const ObsStateT<ChessTag>& root, int maxDepth)
{
    const int D = maxDepth;
    AlignedVec<ObsStateT<ChessTag>> obsStates(D + 1);

    size_t maxActions = m_engine->getMaxValidActions();
    AlignedVec<AlignedVec<ActionT<ChessTag>>> actions(reserve_only, D + 1);
    for (int d = 0; d <= D; ++d)
        actions.emplace_back(maxActions);

    AlignedVec<size_t> cursor(D + 1);
    AlignedVec<size_t> count(D + 1);

    // --- initialisation racine ---
    obsStates[0] = root;
	actions[0].clear();
    m_engine->getValidActions(obsStates[0], actions[0]);
    count[0] = actions[0].size();

    uint64_t nodes = 0;
    int depth = 0;

    // --- DFS itératif ---
    while (true)
    {
        if (cursor[depth] < count[depth])
        {
            // 1) Récupère le coup
            ActionT<ChessTag> mv = actions[depth][cursor[depth]++];

            // 2) Copie l'état
            std::memcpy(
                &obsStates[depth + 1],
                &obsStates[depth],
                sizeof(ObsStateT<ChessTag>)
            );

            // 3) Applique le coup
            m_engine->applyAction(mv, obsStates[depth + 1]);

            if (depth + 1 == D)
            {
                ++nodes;
            }
            else
            {
                ++depth;
                actions[depth].clear();
                m_engine->getValidActions(obsStates[depth], actions[depth]);
                count[depth] = actions[depth].size();
                cursor[depth] = 0;
            }
        }
        else
        {
            if (depth == 0) break;
            --depth;
        }
    }
    return nodes;
}

void PerftTool::runNormal(const AlignedVec<PerftTest>& normalTests)
{
    int nameSpaces = 0;
    int depthSpaces = 0;
    int expSpaces = 0;
    int gotSpaces = 0;
    for (auto const& t : normalTests)
    {
		// Count the maximum spaces for name, depth, exp, got
        if (t.name.length() > nameSpaces)
            nameSpaces = t.name.length();
        if (std::to_string(t.depth).length() > depthSpaces)
            depthSpaces = std::to_string(t.depth).length();
		if (std::to_string(t.expected).length() > expSpaces)
            expSpaces = std::to_string(t.expected).length();
		if (std::to_string(t.expected).length() > gotSpaces)
            gotSpaces = std::to_string(t.expected).length();
    }
    for (auto const& t : normalTests)
    {
        ObsStateT<ChessTag> state;
        FenParser::getFenState(t.fen, state);

        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t nodes = perft(state, t.depth);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bool pass = nodes == t.expected;
        std::cout << "[Normal] " << t.name << std::string(nameSpaces - t.name.length(), ' ')
			<< " d=" << t.depth << std::string(depthSpaces - std::to_string(t.depth).length(), ' ')
			<< " exp=" << t.expected << std::string(expSpaces - std::to_string(t.expected).length(), ' ')
			<< " got=" << nodes << std::string(gotSpaces - std::to_string(nodes).length(), ' ')
            << (pass ? " [PASS]" : " [FAIL]")
            << " time=" << ms << "ms\n";
    }
    std::cout << "Finished!" << std::endl;
}

void PerftTool::runDivide(const AlignedVec<PerftTest>& divideTests)
{
    for (auto const& t : divideTests)
    {
        std::cout << "[Divide] " << t.name << " d=" << t.depth << "\n";

        ObsStateT<ChessTag> root;
        FenParser::getFenState(t.fen, root);

        // actions racine
        AlignedVec<ActionT<ChessTag>> rootActs(reserve_only, m_engine->getMaxValidActions());
        m_engine->getValidActions(root, rootActs);
        AlignedVec<std::pair<ActionT<ChessTag>, uint64_t>> result(reserve_only, rootActs.size());

		double totalTime = 0.0;
        uint64_t totalNodes = 0;

        for (size_t i = 0; i < rootActs.size(); ++i)
        {
            ObsStateT<ChessTag> tmp = root;
            m_engine->applyAction(rootActs[i], tmp);

            auto t0 = std::chrono::high_resolution_clock::now();
            uint64_t nodes = (t.depth > 1)
                ? perft(tmp, t.depth - 1)
                : 1;
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            totalTime += ms;
            totalNodes += nodes;

            result.emplace_back(rootActs[i], nodes);

            std::cout << kSquaresName[rootActs[i].from()] << kSquaresName[rootActs[i].to()] << kPromosLetter[rootActs[i].promo()] << ": " << nodes << "\n";
        }
        if (t.expected > 0)
        {
            bool pass = totalNodes == t.expected;
            std::cout << "TOTAL=" << totalNodes << " time=" << totalTime << "ms" << (pass ? " [PASS]" : " [FAIL]") << "\n\n";
        }
        else
        {
            std::cout << "TOTAL=" << totalNodes << " time=" << totalTime << "ms" << "\n\n";
        }
    }
}