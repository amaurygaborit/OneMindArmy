#pragma once
#include "../../corelib/interfaces/IRenderer.hpp"
#include "ChessTraits.hpp"

class ChessRenderer : public IRenderer<ChessTag>
{
private:
	static constexpr const char* kColor[2] = { "White", "Black" };
	static constexpr const char* kPiecesName[6] = { "Pawns", "Knights", "Bishops", "Rooks", "Queens", "King" };
	static constexpr const char* kPiecesSymbol[13] =
	{
		"\u00A0",    // 0 : Empty square
		"\u2659",    // 1 : White pawn
		"\u2658",    // 2 : White knight
		"\u2657",    // 3 : White bishop
		"\u2656",    // 4 : White rook
		"\u2655",    // 5 : White queen
		"\u2654",    // 6 : White king
		"\u265F",    // 7 : Black pawn
		"\u265E",    // 8 : Black knight
		"\u265D",    // 9 : Black bishop
		"\u265C",    //10 : Black rook
		"\u265B",    //11 : Black queen
		"\u265A"     //12 : Black king
	};
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

	bool m_isRenderRawState = false;

private:
	void dispBoard(uint64_t board) const;
	void renderRawState(const ObsStateT<ChessTag>& obsState) const;

protected:
	void specificSetup(const YAML::Node& config) override;

public:
	ChessRenderer();
	~ChessRenderer() = default;

	void renderState(const ObsStateT<ChessTag>& obsState) const override;
	void renderValidActions(const ObsStateT<ChessTag>& obsState) const override;
	void renderActionPlayed(const ActionT<ChessTag>& action, const size_t idPlayer) const override;
	void renderResult(const ObsStateT<ChessTag>& obsState) const override;
};