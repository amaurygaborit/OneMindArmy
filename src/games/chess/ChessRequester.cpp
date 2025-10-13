#include "ChessRequester.hpp"

#include <iostream>
#include <bitset>

void ChessRequester::specificSetup(const YAML::Node& config)
{
    std::cout << "ChessRequester setup called\n";
}

void ChessRequester::convertToAction(std::string& moveStr, ActionT<ChessTag>& out) const
{
    out = ActionT<ChessTag>{};
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

        out.setFrom(startRank * 8 + startFile);
        out.setTo(endRank * 8 + endFile);
        out.setPromo(promo);
    }
}

void ChessRequester::requestAction(const ObsStateT<ChessTag>& obsState, ActionT<ChessTag>& out) const
{
    std::string moveStr;
    do
    {
        std::cin >> moveStr;
        convertToAction(moveStr, out);
    } while (!m_engine->isValidAction(obsState, out));
}
