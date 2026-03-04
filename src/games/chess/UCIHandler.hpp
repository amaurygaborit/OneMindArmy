#pragma once
#include "../../corelib/interfaces/IHandler.hpp"
#include "PerftTool.hpp"

namespace Chess
{
	class UCIHandler : public Core::IHandler<ChessTypes>
	{
	private:
		PerftTool m_perftTool;
		Vec<PerftTest> m_perftTests;
		Vec<std::string> m_parsedLine;

		static constexpr const char* kStartpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
		std::string m_fenState;

	private:
		void parseLine(const std::string& line);

		void cmdInvalid();
		void cmdUCI();
		void cmdIsReady();
		void cmdUCINewGame();
		void cmdPosition();
		void cmdGo();
		void cmdStop();

	protected:
		void specificSetup(const YAML::Node& config) override;

	public:
		UCIHandler();

		void execute() override;

		// Additional UCI controls
		//void setOption(const std::string& name, const std::string& value);
		//uint64_t runPerft(int depth);
		//void goDepth(int depth);

	};
}