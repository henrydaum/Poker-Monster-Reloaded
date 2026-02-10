class Card:
    def __init__(self, name, card_id, uid, owner, card_type, power_cost, health, card_text):
        # The smallest unit of gameplay.
        self.name = name
        self.card_id = card_id
        self.uid = uid  # unique identifier (duplicates have different uids)
        self.owner = owner  # "hero" or "monster"; can change by stealing with A Playful Pixie; this info will need to be shown to an AI
        self.card_type = card_type  # "short" or "long"
        self.power_cost = power_cost
        self.health = health
        self.starting_health = health  # Used to reset health after a card dies so it can be played again.
        self.card_text = card_text  # Will be displayed

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.uid == other.uid

    def effect(self, gs) -> None:
        # Changes the game state based on the card's effect. 
        # Every effect has its own subclass.
        raise NotImplementedError("Subclass must implement effect()")

    def to_dict(self):
        # Encode Card into a dictionary that HTML can use
        image_filename = f"images/{self.name.replace(' ', '_')}.png"

        data = {
            "name": self.name,
            "card_id": self.card_id,
            "uid": self.uid,
            "owner": self.owner,
            "card_type": self.card_type,
            "power_cost": self.power_cost,
            "health": self.health,
            "starting_health": self.starting_health,
            "card_text": self.card_text,
            "image_filename": image_filename  # To display images on the website
        }
        return data

    @classmethod
    def from_dict(cls, data):
        # Take the dictionary and use its values to call create_card (turns HTML into a Card)
        card =  create_card(
            name=data["name"],
            card_id=data["card_id"],
            uid=data["uid"],
            owner=data["owner"],
            card_type=data["card_type"],
            power_cost=data["power_cost"],
            health=data["health"],
            card_text=data["card_text"]
        )
        card.starting_health = data["starting_health"]
        return card

class Awakening(Card):
    def effect(self, gs):
        #print("Playing Awakening")
        flipped_power_cards = gs.me.power_cards
        gs.me.power_cards = []  # reset power cards
        for power_card in flipped_power_cards:
            if power_card.card_type == "short":
                gs.me.hand.append(power_card)
            if power_card.card_type == "long":
                gs.me.battlefield.append(power_card)

class HealthyEating(Card):
    def effect(self, gs):
        #print("Playing Healthy Eating")
        gs.me.draw()
        gs.me.power_plays_left += 1

# No subclasses for The Sun and The Moon

class APLayfulPixie(Card):
    def effect(self, gs):
        #print("A Playful Pixie effect triggered")
        if gs.opp.deck:
            card = gs.opp.deck.pop(0)
            if card.name == "Mind Control":  # Prevent stealing this card?
                ...
            card.owner = gs.me.name
            gs.me.hand.append(card)

class APearlescentDragon(Card):
    def effect(self, gs):
        #print("A Pearlescent Dragon effect triggered")
        gs.opp.health -= 5
        gs.me.health += 5

class LastStand(Card):
    # Needs to work with all graveyard conditions
    def effect(self, gs):
        #print("Playing Last Stand")
        selected_cards = gs.cache[1:]  # If there are no selected cards, gs.cache[1:] returns an empty list
        for card in selected_cards:  # These cards will be taken out of the graveyard and shuffled into the deck
            gs.me.graveyard.remove(card)
            gs.me.deck.append(card)
        gs.me.shuffle()
        gs.me.last_stand_buff = True
        #print("Last Stand buff granted")

class Reconsider(Card):
    def effect(self, gs):
        #print("Playing Reconsider")
        cards_in_new_order = gs.cache [1:]  # If there are fewer than 3 cards left in the deck, this will return that many cards
        # Reinstate deck
        gs.me.deck = gs.me.deck[3:]  # Removes top 3 cards of deck (or all if there's 3 or less cards remaining)
        for card in reversed(cards_in_new_order):
            gs.me.deck.insert(0, card)  # First card chosen goes first, last card chosen goes last

class NobleSacrifice(Card):
    def effect(self, gs):
        #print("Playing Noble Sacrifice")
        sacrifice = gs.cache[1]
        discard = gs.cache[2]  # Can play when opp has no cards in hand, need to code this
        gs.me.battlefield.remove(sacrifice)
        gs.me.graveyard.append(sacrifice)
        if discard:  # If discard exists, discard it.
            gs.opp.discard(discard)

# For the buff to work properly, short_card_played_this_turn must be updated properly
class MonstersPawn(Card):
    def effect(self, gs):
        if not gs.short_card_played_this_turn:
            if gs.me.monsters_pawn_buff != True:
                #print("Monster's Pawn buff granted")  # Don't print this message if you already have the buff
                ...
            gs.me.monsters_pawn_buff = True
        else:
            gs.me.monsters_pawn_buff = False

class PowerTrip(Card):
    def effect(self, gs):
        #print("Playing Power Trip")
        gs.me.power += 2

# No subclasses for Go All In and Fold

class PokerFace(Card):
    def effect(self, gs):
        #print("Playing Poker Face")
        # When Poker Face targets a player, that is handled in the action class.
        target = gs.cache[1]
        target.health -= 4

class CheapShot(Card):
    def effect(self, gs):
        #print("Playing Cheap Shot")
        # When Cheap Shot targets a player, that is handled in the action class.
        target = gs.cache[1]
        target.health -= 2
        gs.me.draw()

class TheOlSwitcheroo(Card):
    def effect(self, gs):
        #print("Playing The 'Ol Switcheroo")
        temp_health = gs.hero.health
        gs.hero.health = gs.monster.health
        gs.monster.health = temp_health

class Ultimatum(Card):
    def effect(self, gs):
        #print("Playing Ultimatum")
        # Note: current turn_priority should be the person playing the card (not the opp)
        ultimatum = gs.cache[1:3]  # Note: this is a list with two cards
        for card in ultimatum:
            if card in gs.me.deck:
                gs.me.deck.remove(card)  # remove ultimatum cards from deck
        opp_selected_card = gs.cache[-1]
        ultimatum.remove(opp_selected_card)
        gs.me.hand.append(opp_selected_card)
        gs.me.deck.append(ultimatum[0])  # Unselected card goes into deck
        gs.me.shuffle()

class Peek(Card):
    def effect(self, gs):
        #print("Playing Peek")
        deck_top2 = gs.me.deck[:2]  # copy top 2 cards of deck
        gs.me.deck = gs.me.deck[2:]  # remove top 2 cards of deck
        selected_card = gs.cache[1]
        gs.me.hand.append(selected_card)  # put selected card into hand
        deck_top2.remove(selected_card)
        gs.me.deck.append(deck_top2[0])  # Put the other card on the bottom.
        # Don't forget to get the index!

def create_card(name, card_id, uid, owner, card_type, power_cost, health, card_text):
    card_name_to_effect = {
        "Awakening": Awakening,
        "Healthy Eating": HealthyEating,
        "A Playful Pixie": APLayfulPixie,
        "A Pearlescent Dragon": APearlescentDragon,
        "Last Stand": LastStand,
        "Reconsider": Reconsider,
        "Noble Sacrifice": NobleSacrifice,
        "Monster's Pawn": MonstersPawn,
        "Power Trip": PowerTrip,
        "Poker Face": PokerFace,
        "Cheap Shot": CheapShot,
        "The 'Ol Switcheroo": TheOlSwitcheroo,
        "Ultimatum": Ultimatum,
        "Peek": Peek,
    }  # Not all cards need their own subclass

    CardClass = card_name_to_effect.get(name, Card)
    return CardClass(name, card_id, uid, owner, card_type, power_cost, health, card_text)