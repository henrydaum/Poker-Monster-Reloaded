from poker_monster.cardClass import Card

class Player:
    def __init__(self, name, deck, player_type="computer_random"):
        # Initializes a player with a name and a deck of cards.
        self.name = name  # "hero " or "monster"
        self.deck = deck
        self.hand = []
        self.battlefield = []
        self.graveyard = []
        self.health = 15  # Starting health
        self.power_cards = []
        self.power = 0  # Used to play cards face up, gained from playing cards face down.
        self.power_plays_left = 1
        self.power_plays_made_this_turn = 0
        self.last_stand_buff = False
        self.monsters_pawn_buff = False
        self.going_first = False
        self.player_type = player_type # "person" or "computer" - being a computer means being unable to cancel actions or view card info
        self.action_number = 0
        
    def start_turn(self, gs):
        self.power_plays_left = 1
        self.power_plays_made_this_turn = 0
        self.power += len(self.power_cards)
        self.draw()
        for long_card in self.battlefield:
            if long_card.name not in ["The Moon", "The Sun"]:
                #print("Trying to trigger effect: " + long_card.name)
                long_card.effect(gs)
        if self.last_stand_buff:
            self.last_stand_buff = False  # Set this to 0 at the start of one's own turn so that it can be active during opponent's turn
            #print("Last Stand buff wore off")

    def end_turn(self, gs):
        self.power = 0
        gs.pass_priority()

    def draw(self, qty=1):
        for i in range(qty):
            # If there are no cards left in the deck, do not draw to prevent an error.
            if self.deck:
                card = self.deck.pop(0)
                self.hand.append(card)

    def mill(self, qty=1):  # Burns cards from the top of the deck
        for i in range(qty):
            if self.deck:
                card = self.deck.pop(0)
                self.graveyard.append(card)

    def shuffle(self):
        from random import shuffle
        shuffle(self.deck)

    def discard(self, card):
        self.hand.remove(card)
        self.graveyard.append(card)

    # Playing a card face down
    def play_power_card(self, gs, card):
        self.hand.remove(card)
        self.power_cards.append(card)
        self.power_plays_left -= 1
        self.power_plays_made_this_turn += 1
        self.power += 1

    def pay_power_cost(self, gs, card):
        #print("Paying power cost")
        if card.card_type == "short":
            if self.monsters_pawn_buff == True:
                #print("Monster's Pawn buff used")
                self.monsters_pawn_buff = False  # Monster's Pawn buff is used up, and don't have to pay power cost.
            else:
                self.power -= card.power_cost
        if card.card_type == "long":
            self.power -= card.power_cost

    # Playing a card face up, either long or short.
    def play_face_up(self, gs, card, no_effect=False):
        # Pass no_effect to play_short_card (and not play_long_card) since only short cards have effects when you play them
        #print("no more info needed")
        #print("playing card face up: " + card.name)
        if card.card_type == "short":
            self.play_short_card(gs, card, no_effect)
            gs.short_card_played_this_turn = True  # For Monster's Pawn
        elif card.card_type == "long":
            self.play_long_card(gs, card)
        gs.card_played_this_turn = True  # For The Sun

    def play_long_card(self, gs, card):
        self.pay_power_cost(gs, card)
        self.hand.remove(card)
        self.battlefield.append(card)

    def play_short_card(self, gs, card, no_effect=False):
        self.pay_power_cost(gs, card)
        self.hand.remove(card)
        if not no_effect:  # This is for TargetHero and TargetMonster actions
            card.effect(gs)
        self.graveyard.append(card)

    def to_dict(self):
        return {
            "name": self.name,
            "health": self.health,
            "power": self.power,
            "power_plays_left": self.power_plays_left,
            "power_plays_made_this_turn": self.power_plays_made_this_turn,
            "last_stand_buff": self.last_stand_buff,
            "monsters_pawn_buff": self.monsters_pawn_buff,
            "going_first": self.going_first,
            "player_type": self.player_type,
            "action_number": self.action_number,
            
            # Create a list of nested card dictionaries for each card zone
            "hand": [card.to_dict() for card in self.hand],
            "deck": [card.to_dict() for card in self.deck],
            "battlefield": [long_card.to_dict() for long_card in self.battlefield],
            "graveyard": [card.to_dict() for card in self.graveyard],
            "power_cards": [power_card.to_dict() for power_card in self.power_cards],
        }

    @classmethod
    def from_dict(cls, data):
        # Create a new Player instance. Note the empty deck for now.
        player = cls(name=data['name'], deck=[], player_type=data['player_type'])
        
        # Set the simple attributes
        player.health = data["health"]
        player.power = data["power"]
        player.power_plays_left = data["power_plays_left"]
        player.power_plays_made_this_turn = data["power_plays_made_this_turn"]
        player.last_stand_buff = data["last_stand_buff"]
        player.monsters_pawn_buff = data["monsters_pawn_buff"]
        player.going_first = data["going_first"]
        player.player_type = data["player_type"]
        player.action_number = data["action_number"]

        # Rebuild the card lists using Card.from_dict()
        player.hand = [Card.from_dict(card_data) for card_data in data["hand"]]
        player.deck = [Card.from_dict(card_data) for card_data in data["deck"]]
        player.battlefield = [Card.from_dict(card_data) for card_data in data["battlefield"]]
        player.graveyard = [Card.from_dict(card_data) for card_data in data["graveyard"]]
        player.power_cards = [Card.from_dict(card_data) for card_data in data["power_cards"]]
        
        return player