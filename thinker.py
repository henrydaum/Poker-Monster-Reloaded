import hashlib
import logging
import re

logger = logging.getLogger(__name__)

RULES_PROMPT = """HOW TO PLAY POKER MONSTER:
Object/ways to win: Each player starts with 15 health, and a 20-card deck. The first player to run out of health or cards in their deck loses the game.

To start the game: Each player chooses the deck they want to play (red = Monster, green = Hero). Flip a coin to see who goes first. Both players shuffle their decks and draw 4 cards. Finally, the person going first starts.

Turn order: This is a turn-based game, and each turn has the same pattern. At the start of the turn, the player whose turn it is draws a card. Then, if they have any power cards, then they gain 1 power for each one they have (this is skipped on turn 1). From there, they can play the cards in their hand using the set of rules explained on the other side of this card. At the end of the turn, the player whose turn it is passes their turn to their opponent; and they start theirs. This pattern repeats until somebody wins the game.

Note: The player going first doesn't draw a card on their initial turn.

Card rules: Any card can be played face-up or face-down. Face-down cards are known as power cards, and these give the power needed to play cards face-up. In this way, every card has a dual usage. To play a power card, simply take any card from your hand and place it onto the board face-down. When you do this, you have made a power card. Face-up cards are different. While power cards are hidden from your opponent, face-up cards are revealed as you play them. To do this requires power, which is explained below. To play a card from your hand in the face-up configuration, simply pay its power cost, then follow its text. Each card basically does what it says, except short cards are discarded after use, while long cards remain on the board until killed (their health is at the bottom), and then are discarded.

You can (and should, in the early turns of the game) play 1 power card per turn, but it's not required. You can play any number of face-up cards you have power for. Playing a power card gives 1 power that can be used immediately.

Power: The power cost to play a face-up card is in the top-right corner of the card. Power is given by power cards; when you play a power card, you get 1 power, and when your turn starts, you get 1 power for each power card you control. Power resets to 0 when the turn ends.

Shake hands at the end. Good luck!

GAME CONTENTS
Hero's Deck:
3x Awakening (3 short) Flip over your power cards, revealing them. Short cards return to your hand. Long cards stay on the board face-up.
3x Healthy Eating (2 short) Draw a card. You can play an extra power card this turn.
2x The Sun (2 long, 2hp) The Monster can only play 1 face-up card per turn.
2x The Moon (3 long, 2hp) The Monster can't play any more power cards.
2x A Playful Pixie (4 long, 4hp) At the start of your turn, you get to steal the top card of the Monster's deck. You can play it as though it were yours. (Repeat this every turn.)
2x A Pearlescent Dragon (5 long, 4hp) At the start of your turn, you get to steal 5 health from the Monster. (Repeat this every turn.)
1x Last Stand (0 short) Shuffle 3 cards from your discard pile back into your deck. Until your next turn starts, your health can't reach 0.
3x Reconsider (1 short) Look at the top three cards of your deck, then put them back in any order you choose.
2x Noble Sacrifice (1 short) As an extra cost to play this, you must sacrifice a long card. Look at your opponent's hand, and discard a card from it.
= 20 cards total

Monster's Deck:
3x Monster's Pawn (3 long, 3hp) Your first short card each turn costs no power to play.
1x Power Trip (0 short) Gain +2 power (for this turn only).
3x Go All In (3 short) Choose a player. They draw 3 cards and lose 5 health.
1x Fold (0 short) Choose a player. They gain 4 health and discard the top 2 cards of their deck.
3x Poker Face (2 short) Deal 4 damage (to any player or long card).
3x Cheap Shot (2 short) Deal 2 damage (to any player or long card). Draw a card.
1x The 'Ol Switcheroo (3 short) The Hero and the Monster switch health.
2x Ultimatum (1 short) Search your deck for any two cards you want with different names and reveal them. Your opponent chooses one of them. Put that card into your hand, and shuffle the other card into your deck.
3x Peek (1 short) Look at the top two cards of your deck, and put one into your hand and the other on the bottom of your deck.
= 20 cards total"""

SEQUENCE_CHOICE_PROMPT = """{rules}

CURRENT GAME STATE:
{gamestate_text}

AVAILABLE PLAYS (including winrate if available):
{choices_text}

INSTRUCTIONS:
You are playing this card game. Each play above is a complete sequence of actions that has been verified as legal. Choose the one that best advances your position. Try your best to win the game, as though you were trying to become a professional player.

Respond with your 3-5 sentences of reasoning, then output your final answer on its own line in the format:
CHOICE: <number>"""

def signature_hash(signature: tuple) -> str:
    """Deterministic hash of a sequence signature for dedup."""
    raw = str(signature)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def describe_sequence(steps: list[dict]) -> str:
    parts = []
    for s in steps:
        action_text = s.get("action_text", s.get("action_id", "?"))
        if "] " in str(action_text):
            action_text = str(action_text).split("] ", 1)[1]
        parts.append(action_text)
    return " → ".join(parts)

class Thinker:
    def __init__(self, embedder, llm):
        self.embedder = embedder
        self.llm = llm

    def choose_sequence(self, legal_sequences, gamestate_text, graph=None):
        lines = []
        for i, seq in enumerate(legal_sequences[:40]):
            desc = describe_sequence(seq["steps"])
            line = f"[{i}] {desc}"

            if graph:
                action_ids = tuple(str(s["action_id"]) for s in seq["steps"])
                action_sig = signature_hash(action_ids)
                stats = graph.get_sequence_stats(action_sig)
                if stats is not None and stats['total'] >= 3:
                    winrate = stats['wins'] / stats['total'] * 100
                    line += f" ({winrate:.0f}% WR, n={stats['total']})"

            lines.append(line)
        choices_text = "\n".join(lines)

        prompt = SEQUENCE_CHOICE_PROMPT.format(
            rules=RULES_PROMPT,
            gamestate_text=gamestate_text,
            choices_text=choices_text,
        )

        try:
            response = self.llm.invoke(prompt, temperature=0.3)
            choice, reasoning = self._parse_choice(response, len(legal_sequences))
            logger.info(f"LLM chose [{describe_sequence(legal_sequences[choice]['steps'])}]: {reasoning}")

            return legal_sequences[choice]

        except Exception as e:
            logger.error(f"Sequence choice failed: {e}, falling back to random")
            import random
            return random.choice(legal_sequences)

    @staticmethod
    def _parse_choice(response: str, num_choices: int) -> tuple[int, str]:
        """Extract the choice number and reasoning from LLM response."""
        if not response:
            return 0, ""

        # Split reasoning from the CHOICE line
        reasoning = ""
        match = re.search(r"CHOICE:\s*(\d+)", response, re.IGNORECASE)
        if match:
            reasoning = response[:match.start()].strip()
            val = int(match.group(1))
            if 0 <= val < num_choices:
                return val, reasoning

        # Fallback: find the last bare integer in the response
        numbers = re.findall(r"\b(\d+)\b", response)
        for num_str in reversed(numbers):
            val = int(num_str)
            if 0 <= val < num_choices:
                return val, response.strip()

        logger.warning(f"Could not parse choice from response, defaulting to 0")
        return 0, response.strip()