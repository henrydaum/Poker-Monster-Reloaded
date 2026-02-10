# Game phase
from poker_monster.playerClass import Player
from poker_monster.cardClass import Card

PHASE_AWAITING_INPUT = "Awaiting input."

class GameState:
    def __init__(self, hero, monster, turn_priority=None, game_phase=PHASE_AWAITING_INPUT, cache=None):
        # Initializes the game state. Contains both players.
        self.hero = hero
        self.monster = monster
        self.turn_priority = turn_priority  # Basically whose turn it is to take an action
        self.game_phase = game_phase
        self.cache = [] if cache is None else cache  # The cache, similar to Magic: The Gathering's stack
        self.turn_number = 0
        self.winner = None

        self.card_played_this_turn = False  # Flag to track if any card has been played this turn
        self.short_card_played_this_turn = False

    @property
    def me(self):
        """Returns the current player based on turn priority"""
        return self.hero if self.turn_priority == "hero" else self.monster
    
    @property
    def opp(self):
        """Returns the opponent based on turn priority"""
        return self.monster if self.turn_priority == "hero" else self.hero

    @property
    def uncertainty(self):
        # Uncertainty: This is a useful value for AIs to know - It corresponds to how much hidden information there is.
        return len(self.opp.hand) + len(self.opp.deck) + len(self.opp.power_cards) + len(self.me.deck)
    
    def pass_priority(self):
        """Swaps turn priority between hero and monster."""
        self.turn_priority = "monster" if self.turn_priority == "hero" else "hero"

    def turn_transition(self):
        # Ends one player's turn and starts the other's
        #print(f"Ending {self.me.name}'s turn")
        self.me.end_turn(self)
        # Note that turn priority changes here, so the next player is now the previous opponent.
        self.turn_number += 1
        self.card_played_this_turn = False
        self.short_card_played_this_turn = False  # Make sure this is before start_turn so that Monster's Pawn can trigger. The order here is finnicky.
        #print(f"Starting {self.me.name}'s turn")
        self.me.start_turn(self)

    # Check this after every action
    def check_game_over(self) -> None:
        """Determine winner, if any."""
        monster_win = False
        hero_win = False
    
        # Check Hero loss conditions
        if len(self.hero.deck) == 0:
            monster_win = True
        if self.hero.health < 1:
            if self.hero.last_stand_buff:
                #print("Hero stayed alive using Last Stand buff")
                self.hero.health = 1  # Keeps Hero alive
            else:
                monster_win = True
    
        # Check Monster loss conditions
        if self.monster.health < 1 or len(self.monster.deck) == 0:
            hero_win = True
    
        # Determine final outcome
        if monster_win and hero_win:
            self.winner = "tie"
        elif monster_win:
            self.winner = "monster"
        elif hero_win:
            self.winner = "hero"

    # Check this after every action
    def check_long_card_deaths(self) -> None:
        """Check if any long cards on the battlefield have died."""
        for card in self.hero.battlefield[:]:
            if card.health <= 0:
                self.hero.battlefield.remove(card)
                self.hero.graveyard.append(card)
                card.health = card.starting_health  # Restore to full health after it is in the graveyard
        for card in self.monster.battlefield[:]:
            if card.health <= 0:
                self.monster.battlefield.remove(card)
                self.monster.graveyard.append(card)
                card.health = card.starting_health

    # Update this after every action
    def update_pawn_buff(self):
        for long_card in self.me.battlefield:
            if long_card.name == "Monster's Pawn":
                long_card.effect(self)  # This should trigger the buff

        # Check if all Monster's Pawns have died on a given battlefield, and only then remove the buff
        if not any(long_card.name == "Monster's Pawn" for long_card in self.me.battlefield):
            self.me.monsters_pawn_buff = 0
        if not any(long_card.name == "Monster's Pawn" for long_card in self.opp.battlefield):
            self.opp.monsters_pawn_buff = 0

    def get_legal_actions(self):
        # Returns a list of legal actions, each with fresh gamestates
        legal_actions = []
        seen_types = set()

        for i in range(num_actions - 1):  # This loop is kind of slow but it's the only way.
            # Make an action
            action = create_action(self, i)
            # Check if action is legal. This doesn't alter gs, so no need to make a new state until a legal one is found
            legal, reason = action.is_legal()
            action_type = type(action).__name__
            if legal and action_type not in seen_types:
                seen_types.add(action_type)
                # Need to make sure each action has a fresh gs, this is how.
                new_state = self.__class__.from_dict(self.to_dict())
                legal_action = create_action(new_state, i)
                legal_actions.append(legal_action)
        return legal_actions

    def to_dict(self):
        # Turns the gs into a dictionary that HTML can read
        return {
            # Call the methods you already wrote for Player
            "hero": self.hero.to_dict(),
            "monster": self.monster.to_dict(),
            
            # Save the simple attributes
            "turn_priority": self.turn_priority,
            "game_phase": self.game_phase,
            "turn_number": self.turn_number,
            "winner": self.winner,

            # Flags:
            "card_played_this_turn": self.card_played_this_turn,
            "short_card_played_this_turn": self.short_card_played_this_turn,

            # The cache is a list of cards, so we serialize it like other card lists
            "cache": [card.to_dict() for card in self.cache]
        }

    @classmethod
    def from_dict(cls, data):
        # Rebuild the complex objects first
        hero = Player.from_dict(data["hero"])
        monster = Player.from_dict(data["monster"])
        cache = [Card.from_dict(card_data) for card_data in data["cache"]]

        # Create a new GameState instance with the rebuilt players
        gs = cls(
            hero=hero,
            monster=monster,
            turn_priority=data["turn_priority"],
            game_phase=data["game_phase"],
            cache=cache
        )
        
        # Set any remaining attributes
        gs.turn_number = data["turn_number"]
        gs.game_phase = data["game_phase"]
        gs.winner = data["winner"]
        gs.card_played_this_turn = data["card_played_this_turn"]
        gs.short_card_played_this_turn = data["short_card_played_this_turn"]
        
        return gs