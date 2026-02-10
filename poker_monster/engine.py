from numpy import mean
from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from contextlib import nullcontext
from collections import deque
import random
import multiprocessing
import sys

# Core Gameplay Classes
from poker_monster.cardClass import Card, Awakening, HealthyEating, APLayfulPixie, APearlescentDragon, LastStand, Reconsider, NobleSacrifice, MonstersPawn, PowerTrip, PokerFace, CheapShot, TheOlSwitcheroo, Ultimatum, Peek, create_card
from poker_monster.playerClass import Player
from poker_monster.gamestateClass import GameState
from poker_monster.actionClass import Action, InvalidAction, TargetHero, TargetMonster, GetCardInfo, Cancel, EndTurn, SelectFromHand, SelectFromBattlefield, SelectFromOwnBattlefield, SelectFromOppHand, SelectFromDeckTop2, SelectFromGraveyard, SelectFromDeck, SelectFromUltimatum, SelectFromDeckTop3, PlayFaceUp, PlayFaceDown

hero_card_data = [
    (3, "Awakening", "hero", 3, "short", None, "Flip over your power cards, revealing them. Any that are short cards return to your hand. Any that are long cards stay on the board face-up."),
    (3, "Healthy Eating", "hero", 2, "short", None, "Draw a card. You can play an extra power card this turn."),
    (2, "The Sun", "hero", 2, "long", 2, "The Monster can only play 1 face-up card per turn."),
    (2, "The Moon", "hero", 3, "long", 2, "The Monster can't play any more power cards."),
    (2, "A Playful Pixie", "hero", 4, "long", 4, "At the start of your turn, you get to steal the top card of the Monster's deck. (Repeat this every turn.)"),
    (2, "A Pearlescent Dragon", "hero", 5, "long", 4, "At the start of your turn, you get to steal 5 health from the Monster. (Repeat this every turn.)"),
    (1, "Last Stand", "hero", 0, "short", None, "Shuffle 3 other cards from your discard pile into your deck. Until your next turn starts, your health can't reach 0 (damage that would put it to less than 1 puts it to 1 instead.)"),
    (3, "Reconsider", "hero", 1, "short", None, "Look at the top 3 cards of your deck, and put them back in any order you choose."),
    (2, "Noble Sacrifice", "hero", 1, "short", None, "As an additional cost to play this card, you must sacrifice a long card. Look at your opponent's hand and discard a card from it."),
]
"""The card data format is: quantity, name, owner, power_cost, card_type, health, card text"""

monster_card_data = [
    (3, "Monster's Pawn", "monster", 3, "long", 3, "Your first short card each turn costs no power to play."),
    (1, "Power Trip", "monster", 0, "short", None, "Gain +2 power (for this turn only)."),
    (3, "Go All In", "monster", 3, "short", None, "Choose a player. They draw 3 cards and lose 5 health."),
    (1, "Fold", "monster", 0, "short", None, "Choose a player. They gain 4 health and discard the top 2 cards of their deck."),
    (3, "Poker Face", "monster", 2, "short", None, "Deal 4 damage (to any player or long card)."),
    (3, "Cheap Shot", "monster", 2, "short", None, "Deal 2 damage (to any player or long card). Draw a card."),
    (1, "The 'Ol Switcheroo", "monster", 3, "short", None, "The Hero and the Monster switch health."),
    (2, "Ultimatum", "monster", 1, "short", None, "Search your deck for any two cards you want with different names and reveal them. Your opponent chooses one of them. Put the chosen card into your hand, and shuffle the other back into your deck."),
    (3, "Peek", "monster", 1, "short", None, "Look at the top 2 cards of your deck and put one into your hand, and the other on the bottom of your deck."),
]

card_data = hero_card_data + monster_card_data

num_cards = 0

long_cards_vector = []  # List of the uids of just long cards

uid = 0  # This thing is to get num_cards and the long_cards_vector. (Overlaps with build_deck in utils, but that's ok for now.)
for i in range(len(card_data)):
    quantity, name, owner, power_cost, card_type, health, card_text = card_data[i]
    for j in range(quantity):
        num_cards += 1
        if card_type == "long":
            long_cards_vector.append(uid)
        uid += 1

num_actions = num_cards + 2  # +2 for end turn and cancel, computers can't cancel so for them it's num_cards + 1

# Game phases. The strings can be edited to say anything and the game will run the same.
PHASE_AWAITING_INPUT = "Awaiting input."
PHASE_PLAYING_SELECTED_CARD = "Choosing how to play selected card."
PHASE_VIEWING_CARD_INFO = "Viewing card info."
PHASE_SELECTING_GRAVEYARD_CARD = "Choosing card from graveyard."
PHASE_REORDERING_DECK_TOP3 = "Reordering top 3 cards of deck. First card chosen goes on top, second card below that, and third is last."
PHASE_SACRIFICING_LONG_CARD = "Choosing a Noble Sacrifice."
PHASE_DISCARDING_CARD_FROM_OPP_HAND = "Looking at opponent's hand and discarding a card from it."
PHASE_CHOOSING_GO_ALL_IN_TARGET = "Choosing Go All In target."
PHASE_CHOOSING_FOLD_TARGET = "Choosing Fold target."
PHASE_CHOOSING_POKER_FACE_TARGET = "Choosing Poker Face target."
PHASE_CHOOSING_CHEAP_SHOT_TARGET = "Choosing Cheap Shot target."
PHASE_CHOOSING_ULTIMATUM_CARD = "Choosing an Ultimatum card from deck."
PHASE_OPP_CHOOSING_FROM_ULTIMATUM = "Choosing from Ultimatum. Chosen card goes into opp hand, other is shuffled into their deck."
PHASE_CHOOSING_FROM_DECK_TOP2 = "Choosing from Peek."
PHASE_HAND_FULL_DISCARDING_CARD = "Hand full, discarding card."

game_phases = [
    PHASE_AWAITING_INPUT,
    PHASE_PLAYING_SELECTED_CARD,
    PHASE_VIEWING_CARD_INFO,
    PHASE_SELECTING_GRAVEYARD_CARD,
    PHASE_REORDERING_DECK_TOP3,
    PHASE_SACRIFICING_LONG_CARD,
    PHASE_DISCARDING_CARD_FROM_OPP_HAND,
    PHASE_CHOOSING_GO_ALL_IN_TARGET,
    PHASE_CHOOSING_FOLD_TARGET,
    PHASE_CHOOSING_POKER_FACE_TARGET,
    PHASE_CHOOSING_CHEAP_SHOT_TARGET,
    PHASE_CHOOSING_ULTIMATUM_CARD,
    PHASE_OPP_CHOOSING_FROM_ULTIMATUM,
    PHASE_CHOOSING_FROM_DECK_TOP2,
    PHASE_HAND_FULL_DISCARDING_CARD
]

ERROR_ENEMY_HAS_THE_SUN = "Enemy has The Sun, you can't play more than one card per turn."
ERROR_ENEMY_HAS_THE_MOON = "Enemy has The Moon, you can't play power cards."
ERROR_INVALID_SELECTION = "Invalid selection."
ERROR_CANT_PLAY_ANOTHER_POWER_CARD = "You can't play another power card this turn."
ERROR_NOT_ENOUGH_POWER = "Not enough power to play this card."
ERROR_NO_SACRIFICE = "As an additional cost to play this card, you must sacrifice a long card."
ERROR_MUST_PICK_DIFFERENT_CARD = "Already picked."
ERROR_MUST_HAVE_DIFFERENT_NAME = "Card must have a different name."
ERROR_NO_FURTHER_MOVES = "No further moves available with this card."
ERROR_COMPUTERS_CANT_DO = "Computers can't do this action."  # Canceling and seeing card info are QOL features for people, not computers

def build_decks():
    hero_deck = []
    monster_deck = []
    uid = 0
    card_id = 0
    for quantity, name, owner, power_cost, card_type, health, card_text in card_data:
        for i in range(quantity):
            if owner == "hero":
                card = create_card(name, card_id, uid, owner, card_type, power_cost, health, card_text)
                uid += 1  # Increment uid for each unique card
                hero_deck.append(card)
            elif owner == "monster":
                card = create_card(name, card_id, uid, owner, card_type, power_cost, health, card_text)
                uid += 1
                monster_deck.append(card)
        card_id += 1  # Increment card_id for each new card name
    return hero_deck, monster_deck

num_game_phases = len(game_phases)
# Create Matrix of size [i][j], where i = num_actions and j = num_game_phases
ACTION_MAP = [[None for j in range(num_game_phases)] for i in range(num_actions)]

def action_map_helper(game_phase, SelectFromClass=None, choosing_up_down=False, can_target_players=False, can_end_turn=False, can_get_card_info=False, can_cancel=False):
    """Based on a set of parameters, fills in the ACTION_MAP for a specific game phase."""
    j = game_phases.index(game_phase)  # phase_id
    if choosing_up_down:
        ACTION_MAP[0][j] = PlayFaceUp
        ACTION_MAP[1][j] = PlayFaceDown
    if SelectFromClass:
        for i in range(num_actions):
            ACTION_MAP[i][j] = SelectFromClass
    if can_target_players:  # Might need to give these their own slots in order to not confuse AI
        ACTION_MAP[-4][j] = TargetHero
        ACTION_MAP[-3][j] = TargetMonster
    if can_end_turn:
        ACTION_MAP[-2][j] = EndTurn
    if can_get_card_info:
         ACTION_MAP[-2][j] = GetCardInfo  # Same spot as EndTurn, but doesn't matter
    if can_cancel:
        ACTION_MAP[-1][j] = Cancel

# Now fill in the action map for each game phase using the action_map_helper.
action_map_helper(PHASE_AWAITING_INPUT, SelectFromHand, can_end_turn=True)
action_map_helper(PHASE_VIEWING_CARD_INFO, can_cancel=True)
action_map_helper(PHASE_PLAYING_SELECTED_CARD, choosing_up_down=True, can_get_card_info=True, can_cancel=True)  # Can only get card info from this menu location
action_map_helper(PHASE_REORDERING_DECK_TOP3, SelectFromDeckTop3)  # Can't cancel due to revealed info
action_map_helper(PHASE_SACRIFICING_LONG_CARD, SelectFromOwnBattlefield, can_cancel=True)
action_map_helper(PHASE_DISCARDING_CARD_FROM_OPP_HAND, SelectFromOppHand)  # Can't cancel since important info is revealed
action_map_helper(PHASE_SELECTING_GRAVEYARD_CARD, SelectFromGraveyard, can_cancel=True)
action_map_helper(PHASE_CHOOSING_GO_ALL_IN_TARGET, can_target_players=True, can_cancel=True)
action_map_helper(PHASE_CHOOSING_FOLD_TARGET, can_target_players=True, can_cancel=True)
action_map_helper(PHASE_CHOOSING_POKER_FACE_TARGET, SelectFromBattlefield, can_target_players=True, can_cancel=True)
action_map_helper(PHASE_CHOOSING_CHEAP_SHOT_TARGET, SelectFromBattlefield, can_target_players=True, can_cancel=True)
action_map_helper(PHASE_CHOOSING_ULTIMATUM_CARD, SelectFromDeck)  # If cancel was here, you would need to shuffle after looking at your deck. That's an extra step to the reset() function
action_map_helper(PHASE_OPP_CHOOSING_FROM_ULTIMATUM, SelectFromUltimatum)  # Can't cancel due to lack of choice for opponent
action_map_helper(PHASE_CHOOSING_FROM_DECK_TOP2, SelectFromDeckTop2)  # Can't cancel due to revealed information
action_map_helper(PHASE_HAND_FULL_DISCARDING_CARD, SelectFromHand, can_cancel=True)
# Fill in the rest of the game phases with appropriate actions

def create_action(gs, action_id):
    phase_id = game_phases.index(gs.game_phase)
    action_class = ACTION_MAP[action_id][phase_id]

    if action_class is None:
        return InvalidAction(gs, action_id)  
    else:
        action = action_class(gs, action_id)
    return action

# Test:
# j = 1
# for i in range(num_actions):
#     print(f"Action {i}: {ACTION_MAP[i][j] if ACTION_MAP[i][j] else 'None'}")

def display_gamestate(gs):
    lines = []
    sort_key = lambda card: card.name  # or uid

    # Red and purple for Monser, green and yellow for hero
    if gs.turn_priority == "monster":
        lines.append(f"{gs.turn_priority.upper()}'s TURN (Turn {gs.turn_number})")
    else:
        lines.append(f"{gs.turn_priority.upper()}'s TURN (Turn {gs.turn_number})")
    lines.append(f"Game Phase: {gs.game_phase}")
    lines.append(f"My Health: {gs.me.health} | My Deck Size: {len(gs.me.deck)} | My Power: {gs.me.power}")
    lines.append(f"Opp Health: {gs.opp.health} | Opp Deck Size: {len(gs.opp.deck)} | Opp Hand Size: {len(gs.opp.hand)} | Opp Power Cards: {len(gs.opp.power_cards)}")

    # These are for extra info not always present
    # Noble Sacrifice hand reveal
    if gs.game_phase == PHASE_DISCARDING_CARD_FROM_OPP_HAND:  
        lines.append(f"Opp hand: {[card.name for card in sorted(gs.opp.hand, key=sort_key)]}")
    # Peek top2 reveal
    if gs.game_phase == "choosing from Peek":
        top2 = gs.me.deck[:2]
        lines.append(f"My deck top 2 cards: {[card.name for card in sorted(top2, key=sort_key)]}")
    # Ultimatum deck reveal
    if gs.game_phase == PHASE_CHOOSING_ULTIMATUM_CARD:  
        lines.append(f"My deck: {[card.name for card in sorted(gs.me.deck, key=sort_key)]}")
    # Ultimatum ultimatum
    if gs.game_phase == PHASE_OPP_CHOOSING_FROM_ULTIMATUM:  
        lines.append(f"Opp Ultimatum: {[card.name for card in sorted(gs.cache[1:3], key=sort_key)]}")
    # Reconsider reveal
    if gs.game_phase == PHASE_REORDERING_DECK_TOP3:
        top2 = gs.me.deck[:3]
        lines.append(f"My deck top 3 cards: {[card.name for card in sorted(top2, key=sort_key)]}")
    # Standard info
    if gs.me.hand:
        lines.append(f"My Hand: {[card.name for card in sorted(gs.me.hand, key=sort_key)]}")
    if gs.me.power_cards:
        lines.append(f"My Power Cards: {[card.name for card in sorted(gs.me.power_cards, key=sort_key)]}")
    if gs.me.battlefield:
        lines.append(f"My Battlefield: {[(card.name, card.health) for card in sorted(gs.me.battlefield, key=sort_key)]}")
    if gs.opp.battlefield:
        lines.append(f"Opp Battlefield: {[(card.name, card.health) for card in sorted(gs.opp.battlefield, key=sort_key)]}")
    if gs.me.graveyard:
        lines.append(f"My Graveyard: {[card.name for card in sorted(gs.me.graveyard, key=sort_key)]}")
    if gs.opp.graveyard:
        lines.append(f"Opp Graveyard: {[card.name for card in sorted(gs.opp.graveyard, key=sort_key)]}")
    if gs.me.monsters_pawn_buff:
        lines.append("Monster's Pawn buff is active")
    if gs.cache:
        lines.append(f"Cache: {[card.name for card in sorted(gs.cache, key=sort_key)]}")

    # Print card info in a basic way. Could be amended to display card art as well.
    if gs.game_phase == PHASE_VIEWING_CARD_INFO:
        lines.append(f"Power Cost: {gs.cache[0].power_cost}\n{gs.cache[0].card_text}")  # cache[0] is the resolving card that we need to access text from

    return "\n".join(lines)

def display_actions(gs):
    lines = []
    lines.append("Available Actions:")
    
    for action_id in range(num_actions):  # Assuming 20 possible actions
        # print("Creating action: ", action_id)
        action = create_action(gs, action_id)
        legal, error = action.is_legal()
        
        if legal:
            extra_info = ""
            action_name = type(action).__name__  # Get the class name
            if action_name in "SelectFromHand":
                extra_info = f": {action.resolving_card.name}" if action.card_list else ""
            if action_name == "SelectFromBattlefield":
                extra_info = f": {action.target.name}" if action.card_list else ""
            if action_name == "SelectFromOwnBattlefield":
                extra_info = f": {action.sacrifice.name}" if action.card_list else ""
            if action_name == "SelectFromOppHand":
                extra_info = f": {action.discard.name}" if action.card_list else ""
            if action_name == "SelectFromDeckTop2":
                extra_info = f": {action.selected_card.name}" if action.card_list else ""
            if action_name == "SelectFromGraveyard":
                extra_info = f": {action.selected_card.name}" if action.card_list else ""
            if action_name == "SelectFromDeck":
                extra_info = f": {action.selected_card.name}" if action.card_list else ""
            if action_name == "SelectFromUltimatum":
                extra_info = f": {action.selected_card.name}" if action.card_list else ""
            if action_name == "SelectFromDeckTop3":
                extra_info = f": {action.selected_card.name}" if action.card_list else ""
            lines.append(f"[{action_id}] {action_name}{extra_info}")
        elif error != ERROR_INVALID_SELECTION:  # QOL, error invalid shows up too often and don't need to see it
            lines.append(f"[{action_id}] (Invalid) {error}")
        
    return "\n".join(lines)

class GameEngine:
    def __init__(self):
        self.gs = None
        self.hero = None
        self.monster = None
        self.num_actions = num_actions

    def reset(self, hero_type="computer", monster_type="computer"):
        # Starts the game from scratch.
        # Reset the important stuff.
        self.gs = None
        self.hero = None
        self.monster = None

        # Build decks and players
        hero_deck, monster_deck = build_decks()
        self.hero = Player("hero", hero_deck, hero_type)
        self.monster = Player("monster", monster_deck, monster_type)

        # coin flip to see who goes first
        coin_flip = randint(0, 1)
        going_first = None
        if coin_flip == 0:  # Heads is for Monster, obviously
            going_first = "monster"
            self.monster.going_first = True
        else:
            going_first = "hero"
            self.hero.going_first = True

        # Initialize the game state, resetting important variables like the game phase and cache
        self.gs = GameState(self.hero, self.monster, going_first, PHASE_AWAITING_INPUT, cache=[])
        self.hero.shuffle()
        self.monster.shuffle()
        self.hero.draw(4)
        self.monster.draw(4)

    def iterate(self, action_id):
        # Uses the action_id to create an action and enacts it, which changes the game state. 
        # Returns True if success
        action = create_action(self.gs, action_id)
        return action.enact()

    def iterate_sequence(self, sequence):
        for action_id in sequence:
            self.iterate(action_id)

    def get_display_text(self):
        # Shows the information needed to play the game.
        gamestate_text = display_gamestate(self.gs)
        actions_text = display_actions(self.gs)
        return gamestate_text, actions_text

    def get_action_text(self, actions_text, action_id):
        # Parses the text block of available actions to find the specific line.
        target_prefix = f"[{action_id}]"
        # Split the text into individual lines
        lines = actions_text.split('\n')
        
        for line in lines:
            # Strip whitespace to handle indentation or trailing spaces
            clean_line = line.strip()
            # Check if this line starts with target "[16]"
            if clean_line.startswith(target_prefix):
                # Remove the prefix and strip any remaining leading whitespace
                return clean_line.removeprefix(target_prefix).lstrip()
        return None

    def get_legal_actions(self, actions_text):
        # Splits the text into individual lines to process them one by one
        lines = actions_text.strip().split("\n")
        valid_ids = []

        for line in lines:
            # Checks if the line contains a bracketed ID and is not marked Invalid
            if line.startswith("[") and "(Invalid)" not in line:
                # Finds where the number ends to slice it out correctly
                end_index = line.find("]")
                if end_index != -1:
                    # Extracts the number substring and converts to integer
                    action_id = int(line[1:end_index])
                    valid_ids.append(action_id)
                    
        return valid_ids

    def get_legal_sequences(self, all_sequences):
        root = {"children": {}, "terminal": []}

        for seq in all_sequences:
            node = root
            for step in seq["steps"]:
                aid = step["action_id"]
                if aid not in node["children"]:
                    node["children"][aid] = {"children": {}, "terminal": []}
                node = node["children"][aid]
            node["terminal"].append(seq)

        legal = []
        original_gs_dict = self.gs.to_dict()

        def dfs(node, gs_dict):
            legal.extend(node["terminal"])

            for action_id, child_node in node["children"].items():
                test_gs = GameState.from_dict(gs_dict)
                action = create_action(test_gs, int(action_id))
                is_legal, reason = action.enact()
                if is_legal:
                    dfs(child_node, test_gs.to_dict())

        dfs(root, original_gs_dict)
        return legal

    def get_results(self):
        # Returns a result if there is one. If game isn't over, returns None.
        if self.gs.winner:
            if self.gs.winner == "hero":
                rewards = {"hero": 1.0, "monster": -1.0}
            elif self.gs.winner == "monster":
                rewards = {"hero": -1.0, "monster": 1.0}
            else:
                rewards = {"hero": 0.0, "monster": 0.0}
            return rewards
        else:
            return None