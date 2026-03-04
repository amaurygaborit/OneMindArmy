#include "ChessRequester.hpp"

#include <iostream>
#include <bitset>

namespace Chess
{
    USING_GAME_TYPES(ChessTypes);

    void ChessRequester::specificSetup(const YAML::Node& config)
    {
        std::cout << "ChessRequester setup called\n";
    }

    Action ChessRequester::convertToAction(const std::string& moveStr, const State state) const
    {
        Action action{};
        int promo = 0;
        if (moveStr.length() == 5)
        {
            switch (moveStr[4])
            {
            case 'q': promo = 1; break;
            case 'r': promo = 2; break;
            case 'b': promo = 3; break;
            case 'n': promo = 4; break;
            default: promo = 0; break;
            }
        }
        else if (moveStr.length() >= 4)
        {
            int startFile = moveStr[0] - 'a';   // Convert file (a-h) to index (0-7)
            int startRank = moveStr[1] - '1';   // Convert rank (1-8) to index (0-7)
            int endFile = moveStr[2] - 'a';
            int endRank = moveStr[3] - '1';

            int from = static_cast<int>(startRank * 8 + startFile);
            int to = static_cast<int>(endRank * 8 + endFile);

            for (const auto& f : state.elems())
            {
                if (f.pos() == from)
                {
                    action.configure(f.factId(), state.getMeta(SLOT_TURN).ownerId(), from, to, static_cast<float>(promo));
                    break;
                }
            }
        }

		return action;
    }

    void ChessRequester::requestInitialState(const uint32_t player, State& outState) const
    {
        std::cout << "Enter fen:\n";

    }

    Action ChessRequester::requestAction(const State& state) const
    {
        std::string moveStr;
        std::cin >> moveStr;

        return convertToAction(moveStr, state);
    }
}