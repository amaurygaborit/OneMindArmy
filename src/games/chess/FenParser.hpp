#pragma once
#include "ChessTraits.hpp"
#include "../../corelib/AlignedVec.hpp"

#include <ctype.h>

class FenParser
{
public:
	static void getFenState(const std::string& fen, ObsStateT<ChessTag>& out)
	{
		// Reset the state
		out = ObsStateT<ChessTag>{};

		int file = 0;	//Column
		int rank = 7;	//Row
		int field = 0;	//Field
		bool processingEnPassant = false;

		for (char c : fen)
		{
			if (field == 0)
			{
				switch (c)
				{
				case 'P': out.elems.whiteBB[0] |= 1ULL << (8 * rank + file); file++; break;
				case 'N': out.elems.whiteBB[1] |= 1ULL << (8 * rank + file); file++; break;
				case 'B': out.elems.whiteBB[2] |= 1ULL << (8 * rank + file); file++; break;
				case 'R': out.elems.whiteBB[3] |= 1ULL << (8 * rank + file); file++; break;
				case 'Q': out.elems.whiteBB[4] |= 1ULL << (8 * rank + file); file++; break;
				case 'K': out.elems.whiteBB[5] |= 1ULL << (8 * rank + file); file++; break;

				case 'p': out.elems.blackBB[0] |= 1ULL << (8 * rank + file); file++; break;
				case 'n': out.elems.blackBB[1] |= 1ULL << (8 * rank + file); file++; break;
				case 'b': out.elems.blackBB[2] |= 1ULL << (8 * rank + file); file++; break;
				case 'r': out.elems.blackBB[3] |= 1ULL << (8 * rank + file); file++; break;
				case 'q': out.elems.blackBB[4] |= 1ULL << (8 * rank + file); file++; break;
				case 'k': out.elems.blackBB[5] |= 1ULL << (8 * rank + file); file++; break;

				case '/': rank--; file = 0; break;  // Move to next rank
				default: file += (c - '0'); break;	// Convert char digit to int and advance file
				}
			}

			if (field == 1)
			{
				if (c == 'w')
				{
					out.meta.trait = 0;
				}
				else if (c == 'b')
				{
					out.meta.trait = 1;
				}
			}

			if (field == 2)
			{
				if (c != '-')
				{
					switch (c)
					{
					case 'K': out.meta.castlingRights |= 1; break; // White can castle kingside
					case 'Q': out.meta.castlingRights |= 2; break; // White can castle queenside
					case 'k': out.meta.castlingRights |= 4; break; // Black can castle kingside
					case 'q': out.meta.castlingRights |= 8; break; // Black can castle queenside
					default: break;
					}
				}
			}

			if (field == 3)
			{
				if (c != '-')
				{
					if (!processingEnPassant)
					{
						file = c - 'a';           // Convert file (a-h) to index (0-7)
						processingEnPassant = true;
					}
					else
					{
						rank = c - '1';       // Convert rank (1-8) to index (0-7)
						out.meta.enPassant = (uint8_t)(rank * 8 + file); // Store the en-passant target square
						processingEnPassant = false;
					}
				}
			}

			if (field == 4)
			{
				// Halfmove clock
				if (isdigit(c))
				{
					out.meta.halfmoveClock = c - '0'; // Convert char digit to float
				}
			}

			if (field == 5)
			{
				// Fullmove number
				if (isdigit(c))
				{
					out.meta.fullmoveNumber = c - '0'; // Convert char digit to int
				}
			}

			// Check if we've reached the end of the section
			if (c == ' ')
			{
				field++;
			}
		}
	}
};