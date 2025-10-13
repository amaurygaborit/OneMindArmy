#include "UCIHandler.hpp"

#include <iostream>
#include <sstream>

UCIHandler::UCIHandler()
{
    // TEST
    m_perftTests =
    {
        { "Good Test", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48 },
        { "Good Test", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039 },
        { "Good Test", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862 },
        { "Good Test", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603 },
        { "Good Test", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690 },
        { "Good Test", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 6, 8031647685 },
        { "Most Legal Moves", "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1", 1, 218 },
        { "Most Legal Moves", "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1", 2, 99 },
        { "Most Legal Moves", "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1", 3, 19073 },
        { "Most Legal Moves", "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1", 4, 85043 },
        { "Most Legal Moves", "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1", 5, 13853661 },
        { "Most Legal Moves", "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1", 6, 115892741 },
        { "Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20 },
        { "Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400 },
        { "Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902 },
        { "Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281 },
        { "Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609 },
        { "Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324	},
        { "Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 7, 3195901860 },
        { "Discover Promo", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 1, 24 },
        { "Discover Promo", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 2, 496 },
        { "Discover Promo", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 3, 9483 },
        { "Discover Promo", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 4, 182838 },
        { "Discover Promo", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 5, 3605103 },
        { "Discover Promo", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 6, 71179139 },
        { "Other 1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6 },
        { "Other 1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264 },
        { "Other 1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467 },
        { "Other 1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333 },
        { "Other 1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292 },
        { "Other 1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 6, 706045033 }
    };
}

void UCIHandler::specificSetup(const YAML::Node& config)
{
    std::cout << "UCIHandler setup called\n";
    m_perftTool = PerftTool(m_engine);
}

void UCIHandler::parseLine(const std::string& line)
{
    m_parsedLine.clear();
    std::istringstream iss(line);
    std::string word;
    while (iss >> word)
    {
        word.erase(std::remove(word.begin(), word.end(), ' '), word.end());
        if (!word.empty())
            m_parsedLine.push_back(word);
    }
}

void UCIHandler::cmdInvalid()
{
    std::cout << "Invalid command\n";
}

void UCIHandler::cmdUCI()
{
    std::cout << "id name OMAChess\n";
    std::cout << "id author M.Lockwood\n";
    std::cout << "uciok\n";
}

void UCIHandler::cmdIsReady()
{
    std::cout << "readyok\n";
}

void UCIHandler::cmdUCINewGame()
{
    std::cout << "New game started\n";
}

void UCIHandler::cmdPosition()
{
    if (m_parsedLine.size() == 2 && m_parsedLine[1] == "startpos")
    {
        m_fenState = kStartpos;
    }
    else if (m_parsedLine.size() == 8 && m_parsedLine[1] == "fen")
    {
        // combine parsedLine[2]...parsedLine[n] into FEN string
        std::string fen = m_parsedLine[2];
        for (size_t i = 3; i < m_parsedLine.size(); ++i)
        {
            fen += " " + m_parsedLine[i];
        }
        m_fenState = fen;
    }
}

void UCIHandler::cmdGo()
{
    if (m_parsedLine.size() < 2)
    {
        cmdInvalid();
        return;
	}
    if (m_parsedLine[1] == "perft")
    {
        if (m_parsedLine.size() != 3)
        {
            cmdInvalid();
            return;
        }
        if (m_parsedLine[2] == "normalTests")
        {
            m_perftTool.runNormal(m_perftTests);
        }
        else if (m_parsedLine[2] == "divideTests")
        {
            m_perftTool.runDivide(m_perftTests);
        }
        else
        {
            cmdInvalid();
        }
    }
    else if (m_parsedLine[1] == "divPerft")
    {
        if (m_parsedLine.size() != 3)
        {
            cmdInvalid();
            return;
		}
        else
        {
            PerftTest test = {
				"Custom Test",
                m_fenState,
                std::stoi(m_parsedLine[2]),
				0
            };
            AlignedVec<PerftTest> tests = { test };
            m_perftTool.runDivide(tests);
        }
    }
    else
    {
        cmdInvalid();
	}
}

void UCIHandler::cmdStop()
{
    std::cout << "Search stopped\n";
}

void UCIHandler::execute()
{
    std::string line;
    while (true)
    {
        if (!std::getline(std::cin, line)) break;

        parseLine(line);
        if (m_parsedLine.empty()) continue;

        const std::string& cmd = m_parsedLine[0];
        if      (cmd == "quit")         break;
        else if (cmd == "uci")          cmdUCI();
        else if (cmd == "isready")      cmdIsReady();
        else if (cmd == "ucinewgame")   cmdUCINewGame();
        else if (cmd == "position")     cmdPosition();
        else if (cmd == "go")           cmdGo();
        else if (cmd == "stop")         cmdStop();
        else                            cmdInvalid();
    }
}