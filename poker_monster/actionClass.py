from typing import Tuple, Optional

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

# Errors
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
ERROR_ACTION_WITHELD_FROM_AI = "This action is witheld from AIs since it would harm their strategy"  # These are like automatic reflexes for the AI - things not to do that would harm 

class Action(object):
    def __init__(self, gs, action_id):
        # The largest unit of gameplay.
        self.gs = gs
        self.action_id = action_id  # Int from 0 to num_actions - 1. Corresponds to various action types. Each subclass is an action type.

    def is_legal(self) -> Tuple[bool, Optional[str]]:
        # Check if the action is legal. Returns a tuple of (is_legal, reason). If True, reason is None.
        raise NotImplementedError("Subclass must implement is_legal()")

    def execute(self) -> None:
        # Execute the action, changing the gamestate.
        raise NotImplementedError("Subclass must implement is_legal()")

    def enact(self) -> Tuple[bool, Optional[str]]:
        # Enact = if it is legal, update some rewards, and then execute the action
        legal, reason = self.is_legal()
        self.gs.me.action_number += 1
        if legal:
            #print("Action is legal")
            try:
                self.execute()
            except Exception as e:
                print(f"The action had an error while executing: {e}.")
                print(f"Action Class: {type(self).__name__}")
                print(f"Action ID: {self.action_id} - Game Phase: {self.gs.game_phase}")
                print(f"cache: {[card.name for card in self.gs.cache]}")
                raise TypeError
            # Do these after every action
            self.gs.update_pawn_buff()
            self.gs.check_long_card_deaths()
            self.gs.check_game_over()
        else:
            #print("Action is not legal")
            ...
        return legal, reason

    # Reset the cache after resolving a given action.
    def reset(self):
        self.gs.cache = []
        self.gs.game_phase = PHASE_AWAITING_INPUT
        #print("Cleared cache")

# Does not represent a valid action, but is used to handle errors.
class InvalidAction(Action):
    def is_legal(self):
        return False, ERROR_INVALID_SELECTION
    
    def execute(self):
        raise ValueError("Invalid action taken")

# For cards that target the hero directly.
class TargetHero(Action):
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.resolving_card = self.gs.cache[0]

    def is_legal(self):
        return True, None
    
    def execute(self):
        if self.resolving_card.name == "Go All In":
            #print("Playing Go All In")
            self.gs.hero.health -= 5
            self.gs.hero.draw(3)
        
        elif self.resolving_card.name == "Fold":
            #print("Playing Fold")
            self.gs.hero.health += 4
            self.gs.hero.mill(2)
        
        elif self.resolving_card.name == "Poker Face":
            #print("Playing Poker Face")
            self.gs.hero.health -= 4
        
        elif self.resolving_card.name == "Cheap Shot":
            #print("Playing Cheap Shot")
            self.gs.hero.health -= 2
            self.gs.me.draw()
        # For all these cards:
        self.gs.me.play_face_up(self.gs, self.resolving_card, no_effect=True)  # No effect since that was done just above
        self.reset()

# For cards that target the monster directly. Mirrors TargetHero.
class TargetMonster(Action):
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.resolving_card = self.gs.cache[0]

    def is_legal(self):
        return True, None
    
    def execute(self):
        if self.resolving_card.name == "Go All In":
            #print("Playing Go All In")
            self.gs.monster.health -= 5
            self.gs.monster.draw(3)
        
        elif self.resolving_card.name == "Fold":
            #print("Playing Fold")
            self.gs.monster.health += 4
            self.gs.monster.mill(2)
        
        elif self.resolving_card.name == "Poker Face":
            #print("Playing Poker Face")
            self.gs.monster.health -= 4
        
        elif self.resolving_card.name == "Cheap Shot":
            #print("Playing Cheap Shot")
            self.gs.monster.health -= 2
            self.gs.me.draw()
        # For all these cards:
        self.gs.me.play_face_up(self.gs, self.resolving_card, no_effect=True)  # No effect since that was done just above
        self.reset()

class GetCardInfo(Action):
    def is_legal(self):
        if self.gs.me.player_type.startswith("computer"):
            return False, ERROR_COMPUTERS_CANT_DO
        return True, None
    
    def execute(self):
        # Display card info
        self.gs.game_phase = PHASE_VIEWING_CARD_INFO  # You can read card text and cancel

# To cancel the current action and go back.
class Cancel(Action):
    def is_legal(self):
        if self.gs.me.player_type.startswith("computer"):
            return False, ERROR_COMPUTERS_CANT_DO
        return True, None
    
    def execute(self):
        self.reset()  # Just resets the cache. No gamestate changes; all that's lost is information.

# To end the turn. This also starts the next turn for the opponent.
class EndTurn(Action):
    def is_legal(self):
        return True, None  # If the option is available, it is always legal.
    
    def execute(self):        
        if len(self.gs.me.hand) > 5:  # check if hand is over maximum hand size (5)
            self.gs.game_phase = PHASE_HAND_FULL_DISCARDING_CARD
        else:
            self.reset()
            self.gs.turn_transition()

# To select a card from a list (e.g., hand or battlefield).
# After selecting a card from anywhere, it is added to the cache.
# The bottom card in the cache, index [0], is always the one being played--the resolving card.
class SelectFromHand(Action):
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.card_list = self.gs.me.hand # Search this list for a card with a matching uid.
        self.resolving_card = None # Will store selected card here

    def future_moves_available(self, card) -> Tuple:
        # This is just to simplify the game by showing fewer available moves that don't lead anywhere
        # This enables not letting computers cancel their moves

        # Proposed faster code (reverting the change rather than doing a deepcopy)
        original_phase = self.gs.game_phase
        self.gs.game_phase = PHASE_PLAYING_SELECTED_CARD
        self.gs.cache.append(card)
        test_face_up_action = PlayFaceUp(self.gs, 0)  # 0 = action_id; might not matter?
        test_face_down_action = PlayFaceDown(self.gs, 1)
        test_face_up_action_is_legal, reason = test_face_up_action.is_legal()
        test_face_down_action_is_legal, reason = test_face_down_action.is_legal()
        self.gs.cache.pop()
        self.gs.game_phase = original_phase  # Revert the game phase back to the original state

        # If the face up action is not legal and the face down action is not legal, then it should not be an option -> return False, ERROR_NO_FURTHER_MOVES
        if not test_face_up_action_is_legal and not test_face_down_action_is_legal:
            if self.gs.me.player_type.startswith("computer"):  # Could make it an option for person players to enable this in the future
                return False
        else:
            return True
        
    def is_legal(self):
        # Search card list for a card. If there is a match, action is legal.
        for card in self.card_list:
            if card.uid == self.action_id:  # Card's uid must match the action_id
                # If card does not go anywhere, show as not legal
                if self.future_moves_available(card) == False and self.gs.game_phase != PHASE_HAND_FULL_DISCARDING_CARD:  # Need game phase check since this action is also used for end of turn discard
                    return False, ERROR_NO_FURTHER_MOVES
                self.resolving_card = card
                return True, None  # match found
        return False, ERROR_INVALID_SELECTION  # If no match found, action is illegal.

    def execute(self) -> None:
        self.gs.cache.append(self.resolving_card)  # Add selected card to the cache--always the [0] index
        #print(f"Added card '{self.resolving_card.name}' to cache via '{type(self).__name__}'")
        if self.gs.game_phase == PHASE_AWAITING_INPUT:
            self.gs.game_phase = PHASE_PLAYING_SELECTED_CARD
        
        elif self.gs.game_phase == PHASE_HAND_FULL_DISCARDING_CARD:
            self.gs.me.discard(self.resolving_card)
            #print("Discarded card: " + self.resolving_card.name)
            if len(self.gs.me.hand) <= 5:  # 5 = max hand size
                self.reset()
                self.gs.turn_transition()
            # Else, do nothing -- player has to keep discarding cards until they are below the maximum

# This time, the card list being searched is both battlefields.
# For Poker Face and Cheap Shot
class SelectFromBattlefield(Action):
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.card_list = self.gs.me.battlefield + self.gs.opp.battlefield
        self.target = None  # In this case, a target long card
        self.resolving_card = self.gs.cache[0]

    def is_legal(self):
        # Search card list for a card with a matching uid
        for long_card in self.card_list:
            if long_card.uid == self.action_id:
                self.target = long_card
                if self.gs.me.player_type.startswith("computer"):
                    if self.target in self.gs.me.battlefield:
                        return False, ERROR_ACTION_WITHELD_FROM_AI  # The AI should never, ever target their own long card
                return True, None
        return False, ERROR_INVALID_SELECTION  # If no match found
    
    def execute(self) -> None:
        self.gs.cache.append(self.target)  # Add selected target to the cache
        #print(f"Added card '{self.target.name}' to cache via '{type(self).__name__}'")
        self.gs.me.play_face_up(self.gs, self.resolving_card)  
        self.reset()

# For Noble Sacrifice
class SelectFromOwnBattlefield(Action):
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.card_list = self.gs.me.battlefield
        self.sacrifice = None  # long card to be sacrificed as payment for Noble Sacrifice

        self.resolving_card = self.gs.cache[0]  # Noble Sacrifice is on the bottom of the stack

    def is_legal(self):
        # Search battlefield for a sacrifice
        for long_card in self.card_list:
            if long_card.uid == self.action_id:
                self.sacrifice = long_card
                return True, None
        return False, ERROR_INVALID_SELECTION  # If no match found

    def execute(self):
        self.gs.cache.append(self.sacrifice)  # Add selected sacrifice to the cache
        #print(f"Added card '{self.sacrifice.name}' to cache via '{type(self).__name__}'")
        # If enemy hand is empty, card can still be played. Just does nothing.
        if len(self.gs.opp.hand) == 0:
            #print("Nothing to discard")
            self.gs.cache.append(None)  # This will be tackled in card class.
            #print(f"Added None to cache via '{type(self).__name__}'")
            self.gs.me.play_face_up(self.gs, self.resolving_card)
            self.reset()        
        else:
            self.gs.game_phase = PHASE_DISCARDING_CARD_FROM_OPP_HAND

# For Noble Sacrifice
class SelectFromOppHand(Action):
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.card_list = self.gs.opp.hand
        self.discard = None  # enemy card to discard

        self.resolving_card = self.gs.cache[0]  # Noble Sacrifice is on the bottom of the stack/cache

    def is_legal(self):
        # Search opp hand for discard
        for card in self.card_list:
            if card.uid == self.action_id:
                self.discard = card
                return True, None
        return False, ERROR_INVALID_SELECTION  # If no match found
    
    def execute(self):
        self.gs.cache.append(self.discard)  # Append discard to the cache for next step.
        #print(f"Added card '{self.discard.name}' to cache via '{type(self).__name__}'")
        self.gs.me.play_face_up(self.gs, self.resolving_card)  
        self.reset()

# For Peek
class SelectFromDeckTop2(Action):
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.card_list = self.gs.me.deck[:2]
        self.selected_card = None  # card to put into hand

        self.resolving_card = self.gs.cache[0]  # Noble Sacrifice is on the bottom of the stack/cache

    def is_legal(self):
        # choose card from top2
        for card in self.card_list:
            if card.uid == self.action_id:
                self.selected_card = card
                return True, None
        return False, ERROR_INVALID_SELECTION  # If no match found
    
    def execute(self):
        if len(self.gs.me.deck) == 1:  # You can play this with one card left, but
            self.gs.me.draw()  # you just lose
            self.reset()  # Make sure to reset so the GS is still valid after the game
        else:
            self.gs.cache.append(self.selected_card)  # Append selected card to the cache for next step.
            #print(f"Added card '{self.selected_card.name}' to cache via '{type(self).__name__}'")
            self.gs.me.play_face_up(self.gs, self.resolving_card)  
            self.reset()

# For Last Stand
class SelectFromGraveyard(Action):
    # Need to iterate this action class up to three times in order to get information for Last Stand to resolve
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.card_list = self.gs.me.graveyard
        self.selected_card = None  # card to shuffle into deck

        self.resolving_card = self.gs.cache[0]  # In this case, Last Stand
        
        self.previously_selected_cards = self.gs.cache[1:]

    def is_legal(self):
        # choose card from graveyard that *hasn't already been chosen*
        for card in self.card_list:
            if card.uid == self.action_id:  # Test via uid
                if card in self.previously_selected_cards:
                    return False, ERROR_MUST_PICK_DIFFERENT_CARD  # Can't pick the same card twice
                else:
                    self.selected_card = card
                    return True, None
        return False, ERROR_INVALID_SELECTION  # If no match found

    def execute(self):
        # Need escape condition, else this action will loop over and over
        self.gs.cache.append(self.selected_card)  # Append selected card to the cache until cache has 3 cards (not including resolving card)
        #print(f"Added card '{self.selected_card.name}' to cache via '{type(self).__name__}'")
        num_selected_cards = len(self.gs.cache[1:])  # This information is from after the selected card has been added in this step to the cache, unlike the legal check
        if num_selected_cards == 3:  # Full graveyard condition - can nerf to 2
            self.gs.me.play_face_up(self.gs, self.resolving_card)
            self.reset()
        elif num_selected_cards == len(self.gs.me.graveyard):  # Small graveyard condition
            self.gs.me.play_face_up(self.gs, self.resolving_card)
            self.reset()

# For Ultimatum
class SelectFromDeck(Action):
    # choose two cards from deck with *different names*
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.card_list = self.gs.me.deck
        self.selected_card = None  # Will update if found

        self.resolving_card = self.gs.cache[0]  # Ultimatum

        self.previously_selected_card = None
        if len(self.gs.cache) > 1:
            self.previously_selected_card = self.gs.cache[1]  
        # There are only two cards in an ultimatum, so having only one variable for prev selected cards is fine.

        # Need to test if the deck actually has cards with different names
        first_name = self.card_list[0].name
        self.deck_has_different_names = any(card.name != first_name for card in self.card_list[1:])

    def is_legal(self):
        # This is essentially a stricter test in addition to unique id
        for card in self.card_list:
            if card.uid == self.action_id:
                if self.previously_selected_card:
                    if card.name == self.previously_selected_card.name:
                        # Edge case: deck has two cards left, both with the same name. Allow it.
                        if self.deck_has_different_names:
                            return False, ERROR_MUST_HAVE_DIFFERENT_NAME  # Card must have different name, but can be another "Ultimatum"
                self.selected_card = card
                return True, None
        return False, ERROR_INVALID_SELECTION  # If no match found

    def execute(self):
        self.gs.cache.append(self.selected_card)
        #print(f"Added card '{self.selected_card.name}' to cache via '{type(self).__name__}'")
        num_selected_cards = len(self.gs.cache[1:])  # !Again, this must be calculated again after appending the card.
        if num_selected_cards == 2:  # Need two cards for an ultimatum
            self.gs.game_phase = PHASE_OPP_CHOOSING_FROM_ULTIMATUM
            self.gs.pass_priority()  # Pass priority so opponent can choose from established ultimatum (then pass it back)
            #print("Passed priority")

class SelectFromUltimatum(Action):
    # Opp has priority over this step, which is the opposite of every other card in the game.
    # Only with Ultimatum can the opponent make a decision during your turn.
    # Just need to pass back priority after this.
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        ultimatum = self.gs.cache[1:]  # This is the two cards chosen during the previous two game phases
        self.card_list = ultimatum  # Opponent searches the ultimatum
        self.selected_card = None  # Update if found

        self.resolving_card = self.gs.cache[0]

    def is_legal(self):
        # Similar to other searches
        for card in self.card_list:
            if card.uid == self.action_id:  # Test via uid
                self.selected_card = card
                return True, None
        return False, ERROR_INVALID_SELECTION  # If no match found
    
    def execute(self):
        self.gs.cache.append(self.selected_card)  # Add to cache
        #print(f"Added card '{self.selected_card.name}' to cache via '{type(self).__name__}'")
        self.gs.pass_priority()  # Pass it back before the card is handled in the card subclass
        #print("Passed priority back")
        self.gs.me.play_face_up(self.gs, self.resolving_card)
        self.reset()

# Reconsider
class SelectFromDeckTop3(Action):
    # Mimics Last Stand
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.card_list = self.gs.me.deck[:3]  # Top 3 cards of deck, mimics Peek
        self.selected_card = None  # card to rearrange

        self.resolving_card = self.gs.cache[0]  # In this case, Reconsider
        
        self.previously_selected_cards = self.gs.cache[1:]

    def is_legal(self):
        for card in self.card_list:
            if card.uid == self.action_id:  # Test via uid
                if card in self.previously_selected_cards:
                    return False, ERROR_MUST_PICK_DIFFERENT_CARD  # Can't pick the same card twice
                else:
                    self.selected_card = card
                    return True, None
        return False, ERROR_INVALID_SELECTION  # If no match found
    
    def execute(self):
        self.gs.cache.append(self.selected_card)  # Append selected card to the cache until cache has 3 cards (not including resolving card)
        #print(f"Added card '{self.selected_card.name}' to cache via '{type(self).__name__}'")
        num_selected_cards = len(self.gs.cache[1:])  # This information is from after the selected card has been added in this step to the cache, unlike the legal check
        if num_selected_cards == 3:  # Full deck condition
            self.gs.me.play_face_up(self.gs, self.resolving_card)
            self.reset()
        elif num_selected_cards == len(self.gs.me.deck):  # Small deck condition
            self.gs.me.play_face_up(self.gs, self.resolving_card)
            self.reset()

# To play a long or short card face up.
class PlayFaceUp(Action):
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.resolving_card = self.gs.cache[0]

    def is_legal(self) -> Tuple[bool, Optional[str]]:
        has_enough_power = self.resolving_card.power_cost <= self.gs.me.power
        has_free_short_card = self.gs.me.monsters_pawn_buff
        
        if any(long_card.name == "The Sun" for long_card in self.gs.opp.battlefield) and self.gs.card_played_this_turn:
            return False, ERROR_ENEMY_HAS_THE_SUN  # If enemy has the sun, can't play 2 cards in a turn
        
        elif self.resolving_card.card_type == "short":
            if self.resolving_card.name == "Noble Sacrifice":
                if not self.gs.me.battlefield:
                    return False, ERROR_NO_SACRIFICE  # Must have available sacrifice
                if not self.gs.opp.hand and self.gs.me.player_type.startswith("computer"):
                    return False, ERROR_ACTION_WITHELD_FROM_AI  # Helping out the AI
            if self.gs.me.player_type.startswith("computer"):
                if self.resolving_card.name == "The 'Ol Switcheroo" and self.gs.me.health >= self.gs.opp.health:
                    return False, ERROR_ACTION_WITHELD_FROM_AI  # Helping out the AI, you would never play Switcheroo if your opponent would benefit from it
                if self.resolving_card.name == "Awakening" and not any(card for card in self.gs.me.power_cards if card.card_type == "long"):
                    return False, ERROR_ACTION_WITHELD_FROM_AI  # The AI should never play Awakening if there aren't any long cards to flip
            if has_free_short_card:
                #print("Free short card available")
                return True, None  # Can play a free short card thanks to Monster's Pawn
            elif not has_enough_power:
                return False, ERROR_NOT_ENOUGH_POWER
        
        elif self.resolving_card.card_type == "long":
            if not has_enough_power:
                return False, ERROR_NOT_ENOUGH_POWER
        return True, None

    # Different game phases follow depending on the card.
    def execute(self) -> None:
        #print("Gathering info needed to play card face up")
        # These cards require extra info to play, which takes extra game phases
        if self.resolving_card.name == "Last Stand":
            if len(self.gs.me.graveyard) == 0:  # Can still play the card with no graveyard, just does nothing except grant the buff
                # Empty graveyard condition
                self.gs.me.play_face_up(self.gs, self.resolving_card)
                self.reset()
            else:
                self.gs.game_phase = PHASE_SELECTING_GRAVEYARD_CARD
        elif self.resolving_card.name == "Reconsider":
            self.gs.game_phase = PHASE_REORDERING_DECK_TOP3
        elif self.resolving_card.name == "Noble Sacrifice":
            self.gs.game_phase = PHASE_SACRIFICING_LONG_CARD
        elif self.resolving_card.name == "Go All In":
            self.gs.game_phase = PHASE_CHOOSING_GO_ALL_IN_TARGET
        elif self.resolving_card.name == "Fold":
            self.gs.game_phase = PHASE_CHOOSING_FOLD_TARGET
        elif self.resolving_card.name == "Poker Face":
            self.gs.game_phase = PHASE_CHOOSING_POKER_FACE_TARGET
        elif self.resolving_card.name == "Cheap Shot":
            self.gs.game_phase = PHASE_CHOOSING_CHEAP_SHOT_TARGET
        elif self.resolving_card.name == "Ultimatum":
            if len(self.gs.me.deck) == 1:  # You can play this with one card left, but
                self.gs.me.draw()  # you just lose
            else: 
                self.gs.game_phase = PHASE_CHOOSING_ULTIMATUM_CARD
        elif self.resolving_card.name == "Peek":
            self.gs.game_phase = PHASE_CHOOSING_FROM_DECK_TOP2
        # Play the card normally if no extra info needed:
        else:
            #print("No info needed")
            self.gs.me.play_face_up(self.gs, self.resolving_card)  
            self.reset()

# To play a power card.
class PlayFaceDown(Action):
    def __init__(self, gs, action_id):
        super().__init__(gs, action_id)
        self.resolving_card = self.gs.cache[0]
    
    def is_legal(self):
        if any(long_card.name == "The Moon" for long_card in self.gs.opp.battlefield):  # Checking for The Moon (id 3)
            return False, ERROR_ENEMY_HAS_THE_MOON  #  If opponent has the moon, can't play power cards
        elif self.gs.me.power_plays_left < 1:
            return False, ERROR_CANT_PLAY_ANOTHER_POWER_CARD
        if self.gs.me.player_type.startswith("computer"):
            if self.gs.me.name == "monster" and (self.resolving_card.name in ["Go All In" , "The 'Ol Switcheroo"]):  # The AI should never, ever do this
                return False, ERROR_ACTION_WITHELD_FROM_AI
            if self.gs.me.name == "hero" and self.resolving_card.name == "Awakening" and sum(1 for card in self.gs.me.hand if card.name == "Awakening") == 1:
                return False, ERROR_ACTION_WITHELD_FROM_AI  # The AI should never play Awakening face down if they don't have another in their hand they can play
        return True, None
    
    def execute(self):
        #print("playing card face down: " + self.card.name)
        self.gs.me.play_power_card(self.gs, self.resolving_card)
        self.reset()