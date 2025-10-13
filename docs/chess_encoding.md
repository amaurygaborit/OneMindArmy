STATE
- Actuel + Historique (x8) [0]
	- Joueur actuel
		- Pions: 1
		- Cavaliers: 1
		- Fous: 1
		- Tours: 1
		- Dames: 1
		- Roi: 1
	- Joueur adverse
		- Pions: 1
		- Cavaliers: 1
		- Fous: 1
		- Tours: 1
		- Dames: 1
		- Roi: 1
- Actuel [96]
	- Trait: 1
	- Roques: 1
	- En passant: 1
	- Demi-coups depuis coup irreversible: 1
	- Coups complet: 1
	- Repetitions: 1

Total: 102

////////////////////////////////////////
STATE One-Hot Encoding cannaux 8x8
- Actuel + Historique (x8)
	- Joueur actuel
		- Pions: 1
		- Cavaliers: 1
		- Fous: 1
		- Tours: 1
		- Dames: 1
		- Roi: 1
	- Joueur adverse
		- Pions: 1
		- Cavaliers: 1
		- Fous: 1
		- Tours: 1
		- Dames: 1
		- Roi: 1
- Actuel
	- Trait: 1
	- Roques: 4
	- En passant: 1
	- Demi-coups depuis coup irreversible (/50): 1
	- Coups complet (/1000): 1
	- Repetition 1 fois: 1
	- Repetition 2 fois: 1

Total: 106 x 8x8 = 6784
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



ACTION
	- From
	- To
	- Promo
Total: 3

////////////////////////////////////////
ACTION One-Hot Encoding cannaux 8x8
Pour chaque case
	- Sliding Piece: direction: 8
					 distance:  7
	- Knight:		 direction: 8
	- Promotion:	 type:		9
						- 3 directions (avance, capture gauche, capture droite)
						- 3 promotion (dame, tour, cavalier)

Total: 73 x 8x8 = 4672
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


--- TAROT ---
[Element, Position]

- PrivateState / Player:
		Element		|				Position
					| Player1 | Player2 | Player3 | Player4
	- Coeur1		|		  |		 	|		  |
		...
	- CoeurRoi		|		  |		 	|		  |
	- Carreau1		|		  |		 	|		  |
		...
	- CarreauRoi	|		  |		 	|		  |
	- Trefle1		|		  |		 	|		  |
		...
	- TrefleRoi		|		  |		 	|		  |
	- Pique1		|		  |		 	|		  |
		...
	- PiqueRoi		|		  |		 	|		  |
	- Atout1		|		  |		 	|		  |
		...
	- Atout21		|		  |		 	|		  |

- PublicState (same for every Player):
		Element		|				   Position
					| Chien | Table1 | Table2 | Table3 | Table4 | Team Atk | Team Def
	- Coeur1		|		|		 |		  |		   |		|		   |
		...
	- CoeurRoi		|		|		 |		  |		   |		|		   |
	- Carreau1		|		|		 |		  |		   |		|		   |
		...
	- CarreauRoi	|		|		 |		  |		   |		|		   |
	- Trefle1		|		|		 |		  |		   |		|		   |
		...
	- TrefleRoi		|		|		 |		  |		   |		|		   |
	- Pique1		|		|		 |		  |		   |		|		   |
		...
	- PiqueRoi		|		|		 |		  |		   |		|		   |
	- Atout1		|		|		 |		  |		   |		|		   |
		...
	- Atout21		|		|		 |		  |		   |		|		   |


	- GamePhase
	- Pari
	- FirstPlayerRound
	- BettingPlayer
	- ChelemPlayer
	- DealerPlayer

Pour chaque joueur --> initialState = PrivateState

MCTS constructor:
	if (autoCreatePrivateState)
		.createPrivateState()
player.getPrivateState()

engine.getInitialState(StateT& state) !!!!!!!
engine.getCurrentPlayer(const PublicState& state)
engine.getValidActions(const ObservedState& state, Tensor<ActionT>& validActions)
engine.isValidAction(const ObservedState& state, const ActionT& action)
engine.applyActions(ObservedState& state, const ActionT& action)
engine.isTerminal(const ObservedState& state, Tensor<float>& values)

struct ObservedState:
	- PrivateState
	- PublicState

si on sépare les deux

Inference for Player1:
RealPrivateStateP1
	+
PublicState
	+
ActionsHistory



Pour produire K fullStates coherents :
- Option 1 (simple): utiliser diverse sampling - run decoder K times with temperature / top-p and stochasticity;
  each run yields a distinct fullState (masks ensure uniqueness inside a run).
- Option 2 (beam search with diversity): beam search with diversity penalty to get top-K distinct assignments.
- Option 3 (one-shot set decoder): tricky - more complex to implement.


observedState(PublicState, PrivateState of current)
engine->getInitialState(&observedState)
player[current]->chooseAction(observedState)
	mcts->startResearch(observedState)
		expand():
			- 