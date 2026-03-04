#include "PerftTool.hpp"

#include <iostream>
#include <cstring>
#include <span>

namespace Chess
{
    PerftTool::PerftTool()
    {
    }

    PerftTool::PerftTool(std::shared_ptr<ChessEngine> engine) : m_engine(std::move(engine))
    {
    }

    uint64_t PerftTool::perft(const State& root, int maxDepth)
    {
        if (maxDepth <= 0) return 1ULL; // Convention Perft : D=0 vaut 1 nœud (la racine)

        const int D = maxDepth;

        // Allocation contiguë sur le Tas (ou la Pile si tu faisais un std::array)
        Vec<State> states(D + 1);
        Vec<ActionList> actions(D + 1);
        Vec<size_t> cursor(D + 1);

        // Initialisation propre de tous les curseurs
        for (int i = 0; i <= D; ++i) cursor[i] = 0;

        // --- Initialisation racine ---
        states[0] = root;

        // NOUVELLE API : On passe un span vide pour l'historique des hashs pendant un Perft !
        std::span<const uint64_t> emptyHistory{};
        actions[0] = m_engine->getValidActions(states[0], emptyHistory);

        // Optimisation extrême : Si on demande D=1, on renvoie juste la taille
        if (D == 1) return actions[0].size();

        uint64_t nodes = 0;
        int depth = 0;

        // --- DFS Itératif ---
        while (true)
        {
            if (cursor[depth] < actions[depth].size())
            {
                // 1) Récupère le coup par référence const
                const Action& mv = actions[depth][cursor[depth]++];

                // 2) Copie l'état 
                states[depth + 1] = states[depth];

                // 3) Applique le coup
                m_engine->applyAction(mv, states[depth + 1]);

                // 4) On descend d'un niveau
                ++depth;

                // NOUVELLE API : Span vide pour ignorer l'historique
                actions[depth] = m_engine->getValidActions(states[depth], emptyHistory);

                // ==========================================================
                // LA MAGIE DU BULK COUNTING
                // Si on a atteint l'avant-dernière profondeur, on a juste 
                // besoin de compter les coups valides générés.
                // Inutile d'appliquer ces actions et de copier le plateau !
                // ==========================================================
                if (depth == D - 1)
                {
                    nodes += actions[depth].size(); // On ajoute tout d'un coup
                    --depth;                        // On remonte immédiatement
                }
                else
                {
                    cursor[depth] = 0; // On prépare l'exploration de ce nouveau nœud
                }
            }
            else
            {
                // On a exploré toutes les branches de cette profondeur
                if (depth == 0) break;
                --depth;
            }
        }

        return nodes;
    }

    void PerftTool::runNormal(const Vec<PerftTest>& normalTests)
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
            State state;
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

    void PerftTool::runDivide(const Vec<PerftTest>& divideTests)
    {
        for (auto const& t : divideTests)
        {
            std::cout << "[Divide] " << t.name << " d=" << t.depth << "\n";

            State root;
            FenParser::getFenState(t.fen, root);

            // actions racine (Span vide)
            std::span<const uint64_t> emptyHistory{};
            ActionList rootActs = m_engine->getValidActions(root, emptyHistory);

            Vec<std::pair<Action, uint64_t>> result(Core::reserve_only, rootActs.size());

            double totalTime = 0.0;
            uint64_t totalNodes = 0;

            for (size_t i = 0; i < rootActs.size(); ++i)
            {
                State tmp = root;
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

                std::cout << kSquaresName[rootActs[i].source()]
                    << kSquaresName[rootActs[i].dest()]
                    << kPromosLetter[static_cast<int>(rootActs[i].value())]
                    << ": " << nodes
                    << "\n";
            }
            if (t.expected > 0)
            {
                bool pass = totalNodes == t.expected;
                std::cout << "TOTAL=" << totalNodes
                    << " time=" << totalTime << "ms"
                    << (pass ? " [PASS]" : " [FAIL]")
                    << "\n\n";
            }
            else
            {
                std::cout << "TOTAL=" << totalNodes
                    << " time=" << totalTime << "ms"
                    << "\n\n";
            }
        }
    }
}