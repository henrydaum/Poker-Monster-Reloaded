import random
import os
import logging
from pathlib import Path

from poker_monster.engine import GameEngine, PHASE_AWAITING_INPUT, PHASE_CHOOSING_FROM_DECK_TOP2, PHASE_CHOOSING_ULTIMATUM_CARD, PHASE_REORDERING_DECK_TOP3, PHASE_DISCARDING_CARD_FROM_OPP_HAND
from graph import KnowledgeGraph, StepInfo
from services.embedClass import SentenceTransformerEmbedder
from services.llmClass import OpenAILLM
from thinker import Thinker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

SEQUENCE_BOUNDARIES = [
    PHASE_AWAITING_INPUT,
    PHASE_CHOOSING_FROM_DECK_TOP2,
    PHASE_CHOOSING_ULTIMATUM_CARD,
    PHASE_REORDERING_DECK_TOP3,
    PHASE_DISCARDING_CARD_FROM_OPP_HAND,
]

if __name__ == "__main__":
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    embedder.load()
    llm = OpenAILLM("gpt-4o-mini")
    llm.load()

    thinker = Thinker(embedder, llm)

    hero_graph = KnowledgeGraph(db_path=BASE_DIR / "hero_graph.db")
    monster_graph = KnowledgeGraph(db_path=BASE_DIR / "monster_graph.db")

    NUM_GAMES = 1
    for i in range(NUM_GAMES):
        hero_graph.start_new_episode()
        monster_graph.start_new_episode()

        graphs = {
            "hero": hero_graph,
            "monster": monster_graph
        }

        engine = GameEngine()
        engine.reset(hero_type="computer_random", monster_type="computer_ai")

        pending_steps = {
            "hero": None,
            "monster": None
        }

        # Track active sequences per player
        active_sequences = {
            "hero": [],    # remaining steps to execute
            "monster": [],
        }

        while engine.get_results() is None:
            current_player = engine.gs.turn_priority
            current_graph = graphs[current_player]

            gamestate_text, actions_text = engine.get_display_text()
            print(gamestate_text)
            print(actions_text)

            # Record the previous step (now that we know dst)
            prev_step = pending_steps[current_player]
            if prev_step is not None:
                prev_step.dst_id = engine.gs.game_phase
                prev_step.dst_text = gamestate_text
                current_graph.record_step(prev_step, boundaries=SEQUENCE_BOUNDARIES)
                pending_steps[current_player] = None

            # Prepare the new step
            current_step = StepInfo()
            current_step.src_id = engine.gs.game_phase
            current_step.src_text = gamestate_text

            # --- Pick an action ---
            action_id = None

            if engine.gs.me.player_type == "computer_random":
                legal_actions = engine.get_legal_actions(actions_text)
                action_id = random.choice(legal_actions)

            elif engine.gs.me.player_type == "computer_ai":
                remaining = active_sequences[current_player]

                # If no active sequence, choose a new one
                if not remaining:
                    all_sequences = current_graph.get_unique_sequences()
                    legal_sequences = engine.get_legal_sequences(all_sequences)

                    if len(legal_sequences) > 1:
                        chosen = thinker.choose_sequence(legal_sequences, gamestate_text)
                    elif len(legal_sequences) == 1:
                        chosen = legal_sequences[0]
                    else:
                        chosen = None

                    if chosen:
                        remaining = list(chosen["steps"])
                    else:
                        remaining = []

                # Consume next step from sequence, or fall back to random
                if remaining:
                    action_id = int(remaining.pop(0)["action_id"])
                    active_sequences[current_player] = remaining
                else:
                    legal_actions = engine.get_legal_actions(actions_text)
                    action_id = random.choice(legal_actions)

            # --- Execute the action ---
            action_str = engine.get_action_text(actions_text, action_id)
            print(f"Taking action: {action_str}")

            legal, reason = engine.iterate(action_id)
            if legal:
                current_step.action_id = action_id
                current_step.action_text = action_str
                pending_steps[current_player] = current_step
            else:
                # Sequence went stale (shouldn't happen, but be safe)
                print(f"Illegal action: {action_str}. Reason: {reason}")
                active_sequences[current_player] = []  # abandon sequence

        # --- Game over: record final pending steps ---
        rewards = engine.get_results()
        print(f"WINNER: {engine.gs.winner} - {rewards}")

        for player, step in pending_steps.items():
            if step is not None:
                gamestate_text, _ = engine.get_display_text()
                step.dst_id = engine.gs.game_phase
                step.dst_text = gamestate_text
                graphs[player].record_step(step, boundaries=SEQUENCE_BOUNDARIES)

        for player, graph in graphs.items():
            graph.finalize_episode(rewards[player])