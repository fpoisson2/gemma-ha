"""
Générateur de dataset pour fine-tuner FunctionGemma sur Home Assistant.
Génère des exemples multi-turn avec pattern get_entities -> action.

Le modèle apprend à:
1. D'abord récupérer les entités disponibles (get_entities)
2. Puis appeler la fonction appropriée avec la bonne entité

IMPORTANT - Règles de qualité du dataset:
- Filtrer les entités techniques/système (zigbee, gpio, bridge, etc.)
- Exiger une location spécifique pour éviter l'ambiguïté
- Valider la correspondance entre l'action et l'entité cible
"""

import os
import json
import random
from typing import Optional
from dataclasses import dataclass, field

import yaml
from tqdm import tqdm


# =============================================================================
# FILTRAGE DES ENTITÉS TECHNIQUES/SYSTÈME
# =============================================================================

# Patterns d'entités à EXCLURE du dataset (entités techniques, système, debug)
EXCLUDED_ENTITY_PATTERNS = [
    # Zigbee/Z-Wave/réseau
    "zigbee2mqtt",
    "bridge",
    "permit_join",
    "coordinator",
    "z2m_",
    "zha_",
    # Hardware/GPIO
    "gpio",
    "slzb",
    "esp32",
    "esp8266",
    "tasmota",
    "sonoff_",
    # Système/Debug
    "disable_leds",
    "enable_outdoor_temperature",
    "debug",
    "test_",
    "_test",
    "dummy",
    # Réseau/IP
    "192_168_",
    "10_0_",
    "172_16_",
    "autoriser_les_appels",
    # Technique spécifique
    "relais_distant",
    "affichage",  # Sous-fonction d'un appareil
    "purificateur",  # Sous-fonction d'un appareil
    "updater",
    "update",
    # Intégrations système
    "hacs",
    "supervisor",
    "core_",
    "addon_",
]

# Friendly names à exclure aussi
EXCLUDED_FRIENDLY_NAME_PATTERNS = [
    "Zigbee",
    "Bridge",
    "GPIO",
    "SLZB",
    "Permit Join",
    "Disable LED",
    "Enable Outdoor",
    "Coordinator",
    "Debug",
    "Test ",
]


def is_technical_entity(entity: dict) -> bool:
    """
    Vérifie si une entité est technique/système et doit être exclue du dataset.

    Args:
        entity: Dictionnaire avec 'entity_id' et optionnellement 'attributes.friendly_name'

    Returns:
        True si l'entité doit être exclue
    """
    entity_id = entity.get("entity_id", "").lower()
    friendly_name = entity.get("attributes", {}).get("friendly_name", "").lower()

    # Vérifier les patterns dans l'entity_id
    for pattern in EXCLUDED_ENTITY_PATTERNS:
        if pattern.lower() in entity_id:
            return True

    # Vérifier les patterns dans le friendly_name
    for pattern in EXCLUDED_FRIENDLY_NAME_PATTERNS:
        if pattern.lower() in friendly_name:
            return True

    return False


def filter_entities(entities: list[dict]) -> list[dict]:
    """Filtre les entités techniques du dataset."""
    filtered = [e for e in entities if not is_technical_entity(e)]
    excluded_count = len(entities) - len(filtered)
    if excluded_count > 0:
        print(f"  → {excluded_count} entités techniques exclues")
    return filtered


# Variations de texte pour robustesse aux fautes de frappe
def add_typos(text: str, probability: float = 0.3) -> str:
    """Ajoute des fautes de frappe réalistes au texte."""
    if random.random() > probability:
        return text

    typo_type = random.choice([
        "missing_accent",
        "missing_letter",
        "double_letter",
        "swap_letters",
        "wrong_accent",
        "missing_space",
        "lowercase",
    ])

    if typo_type == "missing_accent":
        # Supprimer les accents
        replacements = [
            ("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
            ("à", "a"), ("â", "a"),
            ("ù", "u"), ("û", "u"),
            ("î", "i"), ("ï", "i"),
            ("ô", "o"), ("ö", "o"),
            ("ç", "c"),
        ]
        for old, new in replacements:
            if old in text and random.random() < 0.5:
                text = text.replace(old, new, 1)
                break

    elif typo_type == "missing_letter" and len(text) > 5:
        # Supprimer une lettre aléatoire
        idx = random.randint(1, len(text) - 2)
        text = text[:idx] + text[idx+1:]

    elif typo_type == "double_letter" and len(text) > 3:
        # Doubler une lettre
        idx = random.randint(1, len(text) - 2)
        if text[idx].isalpha():
            text = text[:idx] + text[idx] + text[idx:]

    elif typo_type == "swap_letters" and len(text) > 3:
        # Échanger deux lettres adjacentes
        idx = random.randint(1, len(text) - 3)
        text = text[:idx] + text[idx+1] + text[idx] + text[idx+2:]

    elif typo_type == "wrong_accent":
        # Mauvais accent
        replacements = [
            ("é", "è"), ("è", "é"),
            ("à", "a"), ("â", "à"),
        ]
        for old, new in replacements:
            if old in text and random.random() < 0.5:
                text = text.replace(old, new, 1)
                break

    elif typo_type == "missing_space":
        # Supprimer un espace
        if " " in text:
            spaces = [i for i, c in enumerate(text) if c == " "]
            if spaces:
                idx = random.choice(spaces)
                text = text[:idx] + text[idx+1:]

    elif typo_type == "lowercase":
        # Tout en minuscules
        text = text.lower()

    return text


# Préfixes de politesse et contexte pour variations
POLITENESS_PREFIXES = [
    "", "", "",  # Plus de chances sans préfixe
    "S'il te plaît, ",
    "Peux-tu ",
    "Est-ce que tu peux ",
    "Je voudrais que tu ",
    "Merci de ",
    "Tu peux ",
    "J'aimerais que tu ",
]

URGENCY_PREFIXES = [
    "", "", "",  # Plus de chances sans préfixe
    "Vite, ",
    "Rapidement, ",
    "Tout de suite, ",
    "Maintenant, ",
    "Immédiatement, ",
]

CONTEXT_SUFFIXES = [
    "", "", "", "",  # Plus de chances sans suffixe
    " s'il te plaît",
    " stp",
    " merci",
    " maintenant",
    " tout de suite",
    " quand tu peux",
]

# Templates de requêtes en français par domaine
TEMPLATES_FR = {
    "light": {
        "turn_on": [
            "Allume la lumière {location}",
            "Allume {entity_name}",
            "Mets la lumière {location}",
            "Éclaire {location}",
            "Peux-tu allumer la lumière {location} ?",
            "Active l'éclairage {location}",
            "Allume les lumières {location}",
            "Je veux de la lumière {location}",
            "Lumière {location} s'il te plaît",
            "Allume tout {location}",
            "Met la lumière {location}",
            "On allume {location}",
            "Allume moi la lumière {location}",
            "Eclaire {location}",
            "Light on {location}",
            "Turn on {entity_name}",
            # Québécois
            "Ouvre la lumière {location}",
            "Ouvre les lumières {location}",
            "Ouvre {entity_name}",
            "Ouvre la light {location}",
            # Nouvelles variations
            "Donne-moi de la lumière {location}",
            "J'ai besoin de lumière {location}",
            "Il fait noir {location}",
            "C'est trop sombre {location}",
            "Rallume {location}",
            "Remets la lumière {location}",
            "Mets-moi la lumière {location}",
            "Active la lumière {location}",
            "Illumine {location}",
            "Fais de la lumière {location}",
        ],
        "turn_off": [
            "Éteins la lumière {location}",
            "Éteins {entity_name}",
            "Coupe la lumière {location}",
            "Désactive l'éclairage {location}",
            "Éteins les lumières {location}",
            "Éteint la lumière {location}",
            "Eteins {location}",
            "Plus de lumière {location}",
            "Coupe {location}",
            "Éteindre {location}",
            "Light off {location}",
            "Turn off {entity_name}",
            # Québécois
            "Ferme la lumière {location}",
            "Ferme les lumières {location}",
            "Ferme {entity_name}",
            "Ferme la light {location}",
            # Nouvelles variations
            "Enlève la lumière {location}",
            "Arrête la lumière {location}",
            "Stop la lumière {location}",
            "Noir {location}",
            "Plus besoin de lumière {location}",
            "Désactive {entity_name}",
            "Coupe tout {location}",
            "Éteins-moi ça {location}",
            "Fais le noir {location}",
        ],
        "set_brightness": [
            "Mets la lumière {location} à {brightness}%",
            "Règle la luminosité {location} à {brightness}%",
            "Tamise {location} à {brightness}%",
            "{entity_name} à {brightness} pourcent",
            "Luminosité {location} {brightness}%",
            "Met {brightness} pourcent {location}",
            "Baisse la lumière {location} à {brightness}%",
            "Monte la lumière {location} à {brightness}%",
            # Nouvelles variations
            "Dimme {location} à {brightness}%",
            "Ajuste la lumière {location} à {brightness}%",
            "Je veux {brightness}% {location}",
            "{brightness} pourcent de luminosité {location}",
            "Lumière {location} {brightness} pour cent",
            "Mets {location} à {brightness}",
        ],
        "get_state": [
            "Est-ce que la lumière {location} est allumée ?",
            "La lumière {location} est allumée ?",
            "Est-ce que {entity_name} est allumée ?",
            "Quel est l'état de la lumière {location} ?",
            "La lumière {location} est éteinte ?",
            "C'est allumé {location} ?",
            "Les lumières {location} sont allumées ?",
            "Est-ce allumé {location} ?",
            # Nouvelles variations
            "La lumière {location} marche ?",
            "Y'a de la lumière {location} ?",
            "C'est éclairé {location} ?",
            "Statut lumière {location}",
            "État de {entity_name} ?",
        ],
    },
    "person": {
        "get_state": [
            "Où est {entity_name} ?",
            "Où se trouve {entity_name} ?",
            "{entity_name} est où ?",
            "Est-ce que {entity_name} est à la maison ?",
            "{entity_name} est à la maison ?",
            "Quelle est la position de {entity_name} ?",
            "{entity_name} est là ?",
            "T'es où {entity_name} ?",
            "{entity_name} est rentré ?",
            "Est-ce que {entity_name} est arrivé ?",
            "{entity_name} est parti ?",
            "Localise {entity_name}",
            "Where is {entity_name}?",
        ],
    },
    "switch": {
        "turn_on": [
            "Allume {entity_name}",
            "Active {entity_name}",
            "Mets {entity_name} en marche",
            "Démarre {entity_name}",
            "Lance {entity_name}",
            "Met {entity_name}",
            "Allume le {entity_name}",
            "Active le {entity_name}",
            "Turn on {entity_name}",
        ],
        "turn_off": [
            "Éteins {entity_name}",
            "Désactive {entity_name}",
            "Arrête {entity_name}",
            "Coupe {entity_name}",
            "Stoppe {entity_name}",
            "Ferme {entity_name}",
            "Éteins le {entity_name}",
            "Turn off {entity_name}",
        ],
        "get_state": [
            "Est-ce que {entity_name} est allumé ?",
            "{entity_name} est activé ?",
            "Quel est l'état de {entity_name} ?",
            "{entity_name} marche ?",
            "C'est allumé {entity_name} ?",
            "{entity_name} est en marche ?",
        ],
    },
    "climate": {
        # IMPORTANT: Tous les templates climate DOIVENT avoir {location} pour éviter l'ambiguïté
        # Les templates sans location sont déplacés vers VAGUE_CLIMATE_TEMPLATES pour générer des clarifications
        "set_temperature": [
            # Templates AVEC location (spécifiques)
            "Mets le chauffage {location} à {temperature} degrés",
            "Règle la température {location} à {temperature}°C",
            "Je veux {temperature} degrés {location}",
            "Température {location} à {temperature} degrés",
            "Chauffe {location} à {temperature}°C",
            "Monte le chauffage {location} à {temperature}",
            "Baisse la température {location} à {temperature}",
            "Met {temperature} degrés {location}",
            "{temperature} degrés {location} s'il te plaît",
            "Thermostat {location} à {temperature}",
            "Chauffage {location} à {temperature}",
            "Met le thermostat {location} à {temperature}",
            "Règle {location} à {temperature}",
            "{temperature}° {location}",
            "Mets-moi {temperature} degrés {location}",
            "On met {temperature} {location}",
            "Change la température {location} à {temperature}",
            "Ajuste le thermostat {location} à {temperature}",
            "Configure {temperature} degrés {location}",
        ],
        "set_hvac_mode": [
            # Templates AVEC location
            "Mets le thermostat {location} en mode {mode}",
            "Passe {location} en mode {mode}",
            "Active le mode {mode} {location}",
            "Mode {mode} {location}",
            "Met {location} en {mode}",
            "Change le mode {location} en {mode}",
            "Thermostat {location} en {mode}",
            "Passe le chauffage {location} en {mode}",
        ],
        "turn_on": [
            # Templates AVEC location (spécifiques)
            "Allume le chauffage {location}",
            "Démarre la climatisation {location}",
            "Active le thermostat {location}",
            "Ouvre le chauffage {location}",
            "Démarre le chauffage {location}",
            "Met le chauffage {location}",
            "Allume la clim {location}",
            "Lance le chauffage {location}",
            "J'ai froid {location}",
            "Chauffe {location}",
            "Refroidis {location}",
            "Active la climatisation {location}",
            "Mets la clim {location}",
        ],
        "turn_off": [
            # Templates AVEC location (spécifiques)
            "Éteins le chauffage {location}",
            "Arrête la climatisation {location}",
            "Coupe le chauffage {location}",
            "Ferme le chauffage {location}",
            "Éteins la clim {location}",
            "Arrête le thermostat {location}",
            "Coupe la clim {location}",
            "Stop le chauffage {location}",
            "Désactive le thermostat {location}",
            "Plus de chauffage {location}",
            "Coupe le thermostat {location}",
        ],
        "get_state": [
            "Quelle est la température {location} ?",
            "Il fait combien {location} ?",
            "Quelle température fait-il {location} ?",
            "Le chauffage {location} est allumé ?",
            "Quel est le mode du thermostat {location} ?",
            "C'est à combien {location} ?",
            "Il fait chaud {location} ?",
            "Il fait froid {location} ?",
            "Température {location} ?",
            "Combien de degrés {location} ?",
            "Statut du chauffage {location}",
            "Le thermostat {location} est à combien ?",
        ],
    },
    "cover": {
        "open_cover": [
            "Ouvre les volets {location}",
            "Ouvre {entity_name}",
            "Lève les stores {location}",
            "Monte les volets {location}",
            # Nouvelles variations
            "Relève les volets {location}",
            "Ouvre les stores {location}",
            "Volets {location} ouverts",
            "Monte les stores {location}",
            "Lève {entity_name}",
            "Je veux voir dehors {location}",
            "Fais entrer la lumière {location}",
            "Ouvre tout {location}",
        ],
        "close_cover": [
            "Ferme les volets {location}",
            "Ferme {entity_name}",
            "Baisse les stores {location}",
            "Descends les volets {location}",
            # Nouvelles variations
            "Abaisse les volets {location}",
            "Ferme les stores {location}",
            "Volets {location} fermés",
            "Descends les stores {location}",
            "Baisse {entity_name}",
            "Cache le soleil {location}",
            "Ferme tout {location}",
            "Bloque la lumière {location}",
        ],
        "set_cover_position": [
            "Mets les volets {location} à {position}%",
            "Ouvre {entity_name} à {position}%",
            # Nouvelles variations
            "Volets {location} à {position}%",
            "Position {position}% {location}",
            "{position} pourcent les volets {location}",
            "Règle les volets {location} à {position}",
        ],
        "get_state": [
            "Les volets {location} sont ouverts ?",
            "Les stores {location} sont fermés ?",
            "État des volets {location}",
            "Position des volets {location} ?",
            "{entity_name} est ouvert ?",
        ],
    },
    "lock": {
        "lock": [
            "Verrouille {entity_name}",
            "Ferme à clé {location}",
            "Verrouille la porte {location}",
            # Nouvelles variations
            "Bloque la porte {location}",
            "Met le verrou {location}",
            "Sécurise {entity_name}",
            "Lock {entity_name}",
            "Ferme la serrure {location}",
            "Active le verrou {location}",
            "Barre la porte {location}",
        ],
        "unlock": [
            "Déverrouille {entity_name}",
            "Ouvre {entity_name}",
            "Débloque la porte {location}",
            # Nouvelles variations
            "Enlève le verrou {location}",
            "Unlock {entity_name}",
            "Ouvre la serrure {location}",
            "Désactive le verrou {location}",
            "Débarre la porte {location}",
            "Ouvre la porte {location}",
        ],
        "get_state": [
            "La porte {location} est verrouillée ?",
            "{entity_name} est fermé ?",
            "État de la serrure {location}",
            "C'est verrouillé {location} ?",
            "La porte est ouverte ?",
        ],
    },
    "scene": {
        "turn_on": [
            "Active la scène {entity_name}",
            "Lance le mode {entity_name}",
            "Mets l'ambiance {entity_name}",
            "Scène {entity_name}",
            # Nouvelles variations
            "Démarre la scène {entity_name}",
            "Exécute {entity_name}",
            "Lance {entity_name}",
            "Mode {entity_name}",
            "Ambiance {entity_name}",
            "Active {entity_name}",
            "Je veux l'ambiance {entity_name}",
            "Mets-moi la scène {entity_name}",
            "Passe en mode {entity_name}",
            "Configure la scène {entity_name}",
        ],
    },
    "fan": {
        "turn_on": [
            "Allume le ventilateur {location}",
            "Démarre {entity_name}",
            "Active la ventilation {location}",
            # Nouvelles variations
            "Lance le ventilo {location}",
            "Mets le ventilateur {location}",
            "J'ai besoin d'air {location}",
            "Ventile {location}",
            "Active {entity_name}",
            "Fais de l'air {location}",
            "Mets de l'air {location}",
        ],
        "turn_off": [
            "Éteins le ventilateur {location}",
            "Arrête {entity_name}",
            "Coupe la ventilation {location}",
            # Nouvelles variations
            "Stop le ventilateur {location}",
            "Arrête le ventilo {location}",
            "Plus de ventilation {location}",
            "Désactive {entity_name}",
            "Coupe le ventilo {location}",
        ],
        "get_state": [
            "Le ventilateur {location} est allumé ?",
            "{entity_name} tourne ?",
            "État du ventilateur {location}",
            "La ventilation {location} marche ?",
        ],
    },
}

# Mapping des noms d'entités vers les locutions françaises
# Clé: partie du nom d'entité (en minuscule), Valeur: forme française
# Templates de requêtes impossibles (pas d'entité correspondante)
NEGATIVE_TEMPLATES = [
    # Pièces inexistantes
    ("Allume la lumière de la piscine", "entity_not_found", "Aucune entité 'light' trouvée pour 'piscine'"),
    ("Éteins les lumières du sous-sol", "entity_not_found", "Aucune entité 'light' trouvée pour 'sous-sol'"),
    ("Ouvre les volets du grenier", "entity_not_found", "Aucune entité 'cover' trouvée pour 'grenier'"),
    ("Mets le chauffage de la véranda à 20 degrés", "entity_not_found", "Aucune entité 'climate' trouvée pour 'véranda'"),
    ("Ferme la lumière de la chambre d'amis", "entity_not_found", "Aucune entité 'light' trouvée pour 'chambre d'amis'"),
    ("Allume le ventilateur de la salle de sport", "entity_not_found", "Aucune entité 'fan' trouvée pour 'salle de sport'"),
    ("Allume la lumière du garage", "entity_not_found", "Aucune entité 'light' trouvée pour 'garage'"),
    ("Éteins le plafonnier de la cave", "entity_not_found", "Aucune entité 'light' trouvée pour 'cave'"),
    ("Ouvre les stores de la mezzanine", "entity_not_found", "Aucune entité 'cover' trouvée pour 'mezzanine'"),
    ("Lumière du vestibule", "entity_not_found", "Aucune entité 'light' trouvée pour 'vestibule'"),
    ("Éclaire la bibliothèque", "entity_not_found", "Aucune entité 'light' trouvée pour 'bibliothèque'"),
    ("Allume le cellier", "entity_not_found", "Aucune entité 'light' trouvée pour 'cellier'"),
    ("Volets de l'atelier", "entity_not_found", "Aucune entité 'cover' trouvée pour 'atelier'"),
    ("Chauffage du bureau de papa", "entity_not_found", "Aucune entité 'climate' trouvée pour 'bureau de papa'"),
    # Personnes inexistantes
    ("Où est Marie ?", "entity_not_found", "Aucune entité 'person' trouvée pour 'Marie'"),
    ("Est-ce que Pierre est à la maison ?", "entity_not_found", "Aucune entité 'person' trouvée pour 'Pierre'"),
    ("Localise le chien", "entity_not_found", "Aucune entité 'person' ou 'device_tracker' trouvée pour 'chien'"),
    ("Où se trouve maman ?", "entity_not_found", "Aucune entité 'person' trouvée pour 'maman'"),
    ("Papa est rentré ?", "entity_not_found", "Aucune entité 'person' trouvée pour 'papa'"),
    ("Julie est où ?", "entity_not_found", "Aucune entité 'person' trouvée pour 'Julie'"),
    ("Le chat est dehors ?", "entity_not_found", "Aucune entité trouvée pour 'chat'"),
    ("Où sont les enfants ?", "entity_not_found", "Aucune entité 'person' trouvée pour 'enfants'"),
    ("Grand-mère est arrivée ?", "entity_not_found", "Aucune entité 'person' trouvée pour 'grand-mère'"),
    # Appareils inexistants
    ("Allume le lave-vaisselle", "entity_not_found", "Aucune entité trouvée pour 'lave-vaisselle'"),
    ("Démarre la machine à laver", "entity_not_found", "Aucune entité trouvée pour 'machine à laver'"),
    ("Éteins le four", "entity_not_found", "Aucune entité trouvée pour 'four'"),
    ("Ouvre le portail", "entity_not_found", "Aucune entité 'cover' ou 'lock' trouvée pour 'portail'"),
    ("Ferme les rideaux", "entity_not_found", "Aucune entité 'cover' trouvée pour 'rideaux'"),
    ("Allume le micro-ondes", "entity_not_found", "Aucune entité trouvée pour 'micro-ondes'"),
    ("Éteins le téléviseur de la salle de jeu", "entity_not_found", "Aucune entité 'media_player' trouvée pour 'salle de jeu'"),
    ("Démarre le robot aspirateur", "entity_not_found", "Aucune entité 'vacuum' trouvée"),
    ("Ouvre le garage", "entity_not_found", "Aucune entité 'cover' trouvée pour 'garage'"),
    ("Allume la cafetière", "entity_not_found", "Aucune entité trouvée pour 'cafetière'"),
    ("Éteins l'imprimante", "entity_not_found", "Aucune entité trouvée pour 'imprimante'"),
    ("Démarre le sèche-linge", "entity_not_found", "Aucune entité trouvée pour 'sèche-linge'"),
    ("Mets la musique", "entity_not_found", "Aucune entité 'media_player' configurée"),
    ("Allume la télé", "entity_not_found", "Aucune entité 'media_player' trouvée pour 'télé'"),
    ("Ouvre le frigo", "entity_not_found", "Aucune entité trouvée pour 'frigo'"),
    # Scènes inexistantes
    ("Active la scène romantique", "entity_not_found", "Aucune scène 'romantique' trouvée"),
    ("Lance le mode fête", "entity_not_found", "Aucune scène 'fête' trouvée"),
    ("Ambiance détente", "entity_not_found", "Aucune scène 'détente' trouvée"),
    ("Mode nuit", "entity_not_found", "Aucune scène 'nuit' trouvée"),
    ("Scène lecture", "entity_not_found", "Aucune scène 'lecture' trouvée"),
]

# Templates de requêtes ambiguës ou incomplètes
AMBIGUOUS_TEMPLATES = [
    # Requêtes trop vagues
    ("Allume", "clarification_needed", "Précisez ce que vous voulez allumer"),
    ("Éteins", "clarification_needed", "Précisez ce que vous voulez éteindre"),
    ("Allume tout", "clarification_needed", "Précisez quelles lumières vous voulez allumer"),
    ("Éteins tout", "clarification_needed", "Précisez quelles lumières vous voulez éteindre"),
    ("Ouvre", "clarification_needed", "Précisez ce que vous voulez ouvrir"),
    ("Ferme", "clarification_needed", "Précisez ce que vous voulez fermer"),
    ("Mets le chauffage", "clarification_needed", "Précisez la pièce et la température souhaitée"),
    ("Monte le chauffage", "clarification_needed", "Précisez la pièce et la température souhaitée"),
    ("Baisse", "clarification_needed", "Précisez ce que vous voulez baisser"),
    ("Active", "clarification_needed", "Précisez ce que vous voulez activer"),
    ("Démarre", "clarification_needed", "Précisez ce que vous voulez démarrer"),
    ("Change", "clarification_needed", "Précisez ce que vous voulez changer"),
    ("Règle", "clarification_needed", "Précisez ce que vous voulez régler"),
    ("Met", "clarification_needed", "Précisez ce que vous voulez mettre"),
    ("Stop", "clarification_needed", "Précisez ce que vous voulez arrêter"),

    # Requêtes incomplètes coupées
    ("Allume la", "clarification_needed", "Requête incomplète"),
    ("Éteins le", "clarification_needed", "Requête incomplète"),
    ("Mets à", "clarification_needed", "Requête incomplète"),
    ("Je veux", "clarification_needed", "Requête incomplète"),
    ("Peux-tu", "clarification_needed", "Requête incomplète"),
    ("Est-ce que", "clarification_needed", "Requête incomplète"),
    ("Où est", "clarification_needed", "Requête incomplète"),
    ("La lumière", "clarification_needed", "Précisez l'action souhaitée"),
    ("Le chauffage", "clarification_needed", "Précisez l'action souhaitée"),
    ("Les volets", "clarification_needed", "Précisez l'action souhaitée"),
    ("Il fait", "clarification_needed", "Requête incomplète"),
    ("Je voudrais", "clarification_needed", "Requête incomplète"),
    ("S'il te plaît", "clarification_needed", "Précisez votre demande"),
    ("Tu peux", "clarification_needed", "Requête incomplète"),
    # Hors sujet
    ("Quel temps fait-il ?", "out_of_scope", "Je ne peux que contrôler les appareils domotiques"),
    ("Quelle heure est-il ?", "out_of_scope", "Je ne peux que contrôler les appareils domotiques"),
    ("Raconte-moi une blague", "out_of_scope", "Je ne peux que contrôler les appareils domotiques"),
    ("Bonjour", "out_of_scope", "Je suis un assistant domotique. Comment puis-je vous aider ?"),
    ("Merci", "out_of_scope", "Je suis un assistant domotique. Comment puis-je vous aider ?"),
    ("C'est quoi Home Assistant ?", "out_of_scope", "Je ne peux que contrôler les appareils domotiques"),
    ("Comment ça marche ?", "out_of_scope", "Je ne peux que contrôler les appareils domotiques"),
    ("Salut", "out_of_scope", "Je suis un assistant domotique. Comment puis-je vous aider ?"),
    ("Hello", "out_of_scope", "Je suis un assistant domotique. Comment puis-je vous aider ?"),
    ("Bonsoir", "out_of_scope", "Je suis un assistant domotique. Comment puis-je vous aider ?"),
    ("Aide-moi", "out_of_scope", "Dites-moi quelle action domotique vous souhaitez effectuer"),
    ("Help", "out_of_scope", "Dites-moi quelle action domotique vous souhaitez effectuer"),
    ("Qu'est-ce que tu sais faire ?", "out_of_scope", "Je peux contrôler vos lumières, chauffage, volets et autres appareils"),
    ("T'es qui ?", "out_of_scope", "Je suis un assistant domotique pour Home Assistant"),
    ("Ça va ?", "out_of_scope", "Je suis un assistant domotique. Comment puis-je vous aider ?"),
    ("Quoi de neuf ?", "out_of_scope", "Je suis un assistant domotique. Comment puis-je vous aider ?"),
    ("C'est nul", "out_of_scope", "Je suis un assistant domotique. Comment puis-je vous aider ?"),
    ("Tu es bête", "out_of_scope", "Je suis un assistant domotique. Comment puis-je vous aider ?"),
    # Texte incompréhensible
    ("asdfjkl", "clarification_needed", "Je n'ai pas compris votre demande"),
    ("???", "clarification_needed", "Je n'ai pas compris votre demande"),
    ("...", "clarification_needed", "Je n'ai pas compris votre demande"),
    ("lum salon", "clarification_needed", "Précisez l'action souhaitée pour la lumière du salon"),
    ("chauf 20", "clarification_needed", "Précisez quelle action effectuer"),
    ("sal", "clarification_needed", "Je n'ai pas compris votre demande"),
    ("!!", "clarification_needed", "Je n'ai pas compris votre demande"),
    ("ok", "clarification_needed", "Je n'ai pas compris votre demande"),
    ("oui", "clarification_needed", "Précisez votre demande"),
    ("non", "clarification_needed", "Précisez votre demande"),
    ("lumiere", "clarification_needed", "Précisez l'action et la pièce"),
    ("temp", "clarification_needed", "Précisez la température souhaitée"),
    ("volet", "clarification_needed", "Précisez l'action (ouvrir/fermer) et la pièce"),
    ("20 degrés", "clarification_needed", "Précisez quel thermostat régler"),
    ("50%", "clarification_needed", "Précisez quel appareil régler"),
    # Valeurs invalides
    ("Mets le chauffage à 50 degrés", "invalid_value", "Température invalide. Plage acceptée: 15-30°C"),
    ("Mets le chauffage à -5 degrés", "invalid_value", "Température invalide. Plage acceptée: 15-30°C"),
    ("Mets la lumière à 150%", "invalid_value", "Luminosité invalide. Plage acceptée: 0-100%"),
    ("Mets les volets à 200%", "invalid_value", "Position invalide. Plage acceptée: 0-100%"),
    ("Température à 0", "invalid_value", "Température invalide. Plage acceptée: 15-30°C"),
]

# =============================================================================
# TEMPLATES SANS ACTION - Requêtes vagues où le modèle ne doit PAS agir
# =============================================================================

# Ces requêtes sont des commandes "valides" mais manquent de spécificité (pas de location).
# Le modèle doit apprendre à NE RIEN FAIRE pour ces requêtes.
# L'utilisateur devra reformuler avec une location précise.

NO_ACTION_TEMPLATES = [
    # ===== Climate sans location =====
    "Allume le chauffage",
    "Éteins le chauffage",
    "Mets le chauffage",
    "Démarre la climatisation",
    "Arrête la climatisation",
    "Allume la clim",
    "Éteins la clim",
    "Coupe le chauffage",
    "Lance le chauffage",
    "Active le thermostat",
    "Désactive le thermostat",
    "J'ai froid",
    "J'ai chaud",
    "Il fait froid",
    "Il fait chaud",
    "Monte le chauffage",
    "Baisse le chauffage",
    "Mets 20 degrés",
    "Mets 22 degrés",
    "Température à 21",
    "Chauffage à 20",
    "21 degrés s'il te plaît",
    "Met le thermostat à 20",
    "Règle la température à 22",
    "Je veux 20 degrés",
    "Passe en mode chauffage",
    "Mode climatisation",
    "Mode éco",
    # Variations avec politesse
    "S'il te plaît, allume le chauffage",
    "Peux-tu allumer le chauffage ?",
    "Tu peux mettre le chauffage ?",
    "J'aimerais que tu allumes la clim",
    "Est-ce que tu peux éteindre le chauffage ?",
    "Merci de mettre le chauffage",
    # Variations Québécoises
    "Ouvre le chauffage",
    "Ferme le chauffage",
    "Ouvre la clim",
    "Ferme la clim",
    # ===== Lumières sans location spécifique =====
    "Allume la lumière",
    "Éteins la lumière",
    "Allume les lumières",
    "Éteins les lumières",
    "Mets la lumière",
    "Coupe la lumière",
    "Lumière s'il te plaît",
    "Plus de lumière",
    "Pas de lumière",
    "Je veux de la lumière",
    "Éclaire",
    "Fais de la lumière",
    "Rallume",
    "Light on",
    "Light off",
    # Québécois
    "Ouvre la lumière",
    "Ferme la lumière",
    "Ouvre les lumières",
    "Ferme les lumières",
    # ===== Volets sans location =====
    "Ouvre les volets",
    "Ferme les volets",
    "Ouvre les stores",
    "Ferme les stores",
    "Monte les volets",
    "Baisse les volets",
    "Lève les volets",
    "Descends les volets",
    "Volets ouverts",
    "Volets fermés",
    # ===== Ventilateur sans location =====
    "Allume le ventilateur",
    "Éteins le ventilateur",
    "Mets le ventilo",
    "Coupe le ventilo",
    "Active la ventilation",
    "Arrête la ventilation",
    "Lance le ventilateur",
    "J'ai besoin d'air",
    "Fais de l'air",
]


ENTITY_TO_LOCATION_FR = {
    # Pièces principales
    "salon": "du salon",
    "chambre": "de la chambre",
    "cuisine": "de la cuisine",
    "salle_de_bain": "de la salle de bain",
    "salle_a_manger": "de la salle à manger",
    "bureau": "du bureau",
    "couloir": "du couloir",
    "entree": "de l'entrée",
    "garage": "du garage",
    "jardin": "du jardin",
    "terrasse": "de la terrasse",
    "buanderie": "de la buanderie",
    "cave": "de la cave",
    "grenier": "du grenier",
    "balcon": "du balcon",
    "salle_de_jeu": "de la salle de jeu",
    # Chambres spécifiques
    "chambre_francis": "de la chambre de Francis",
    "chambre_noemie": "de la chambre de Noémie",
    "chambre_laura": "de la chambre de Laura",
    "chambre_francis_et_noemie": "de la chambre de Francis et Noémie",
    # Extérieur
    "outdoor": "extérieure",
    "balcon_avant": "du balcon avant",
    "balcon_arriere": "du balcon arrière",
    # Autres
    "armoire": "de l'armoire",
    "armoire_cuisine": "de l'armoire de cuisine",
}

# Modes HVAC
HVAC_MODES_FR = {
    "chauffage": "heat",
    "climatisation": "cool",
    "auto": "auto",
    "éco": "eco",
    "absent": "off",
}


def extract_location_from_entity(entity_id: str) -> Optional[str]:
    """
    Extrait la localisation française depuis un entity_id.

    Exemple: light.salon → "du salon"
             light.salle_de_bain → "de la salle de bain"
             light.lumiere_cuisine → "de la cuisine"

    Retourne None si aucune localisation n'est trouvée.
    """
    # Extraire la partie après le domaine (ex: "salon" de "light.salon")
    entity_name = entity_id.split(".")[-1].lower()

    # Chercher la correspondance la plus longue d'abord (pour éviter que "chambre"
    # matche avant "chambre_laura")
    sorted_keys = sorted(ENTITY_TO_LOCATION_FR.keys(), key=len, reverse=True)

    for key in sorted_keys:
        if key in entity_name:
            return ENTITY_TO_LOCATION_FR[key]

    return None


def escape_param(value: str) -> str:
    """Échappe une valeur de paramètre."""
    return f"<escape>{value}<escape>"


def format_function_call(func_name: str, params: dict) -> str:
    """Formate un appel de fonction FunctionGemma."""
    params_str = ",".join(
        f"{k}:{escape_param(v) if isinstance(v, str) else v}"
        for k, v in params.items()
    )
    return f"<start_function_call>call:{func_name}{{{params_str}}}<end_function_call>"


@dataclass
class MultiTurnExample:
    """Un exemple d'entraînement multi-turn."""
    user_query: str
    domain: str
    available_entities: list[str]  # Liste des entity_ids disponibles pour ce domaine
    target_entity: str  # L'entité choisie
    action: str  # ex: "turn_on", "set_temperature"
    action_params: dict  # Paramètres additionnels (brightness, temperature, etc.)
    all_entities_by_domain: dict = field(default_factory=dict)  # Toutes les entités par domaine pour one-step

    def _format_state_response(self) -> str:
        """Génère une réponse d'état simulée pour l'entraînement."""
        entity_name = self.target_entity.split(".")[-1].replace("_", " ").title()

        if self.domain == "person":
            locations = ["home", "away", "work", "not_home"]
            loc = random.choice(locations)
            return f"{entity_name}: {loc}"
        elif self.domain == "light":
            states = ["on (75%)", "off", "on (100%)", "on (50%)"]
            return f"{entity_name}: {random.choice(states)}"
        elif self.domain == "climate":
            temp = random.randint(18, 24)
            modes = ["heat", "cool", "auto", "off"]
            return f"{entity_name}: {temp}°C, mode {random.choice(modes)}"
        elif self.domain == "cover":
            positions = ["open (100%)", "closed (0%)", "open (50%)"]
            return f"{entity_name}: {random.choice(positions)}"
        elif self.domain == "lock":
            states = ["locked", "unlocked"]
            return f"{entity_name}: {random.choice(states)}"
        elif self.domain == "switch":
            states = ["on", "off"]
            return f"{entity_name}: {random.choice(states)}"
        elif self.domain == "fan":
            states = ["on", "off"]
            return f"{entity_name}: {random.choice(states)}"
        else:
            return f"{entity_name}: unknown"

    def to_training_format(self) -> dict:
        """
        Convertit en format d'entraînement multi-turn.

        Pattern:
        1. User demande une action
        2. Model appelle get_entities pour le domaine
        3. Tool retourne les entités disponibles
        4. Model appelle l'action avec la bonne entité
        """
        # Appel get_entities
        get_entities_call = format_function_call(
            "get_entities",
            {"domain": self.domain}
        )

        # Réponse du tool avec les entités disponibles
        entities_list = ", ".join(self.available_entities[:10])  # Limiter à 10
        tool_response = f"Entités {self.domain} disponibles: {entities_list}"

        # Appel de l'action finale
        action_params = {"entity_id": self.target_entity}
        action_params.update(self.action_params)

        # Pour get_state, utiliser ha.get_states (tool MCP) sans paramètres
        if self.action == "get_state":
            action_call = "<start_function_call>call:ha.get_states{}<end_function_call>"
            # Simuler la réponse du tool avec les états
            states_response = self._format_state_response()
            text = (
                f"<start_of_turn>user\n{self.user_query}<end_of_turn>\n"
                f"<start_of_turn>model\n{get_entities_call}<end_of_turn>\n"
                f"<start_of_turn>tool\n{tool_response}<end_of_turn>\n"
                f"<start_of_turn>model\n{action_call}<end_of_turn>\n"
                f"<start_of_turn>tool\n{states_response}<end_of_turn>"
            )
        else:
            action_call = format_function_call(
                f"{self.domain}.{self.action}",
                action_params
            )
            # Format texte pour l'entraînement
            text = (
                f"<start_of_turn>user\n{self.user_query}<end_of_turn>\n"
                f"<start_of_turn>model\n{get_entities_call}<end_of_turn>\n"
                f"<start_of_turn>tool\n{tool_response}<end_of_turn>\n"
                f"<start_of_turn>model\n{action_call}<end_of_turn>"
            )

        return {"text": text}

    def to_single_turn_format(self) -> dict:
        """
        Convertit en format single-turn (sans get_entities).

        Pattern simplifié:
        1. User demande une action
        2. Model appelle directement l'action avec la bonne entité
        """
        # Appel de l'action directe
        action_params = {"entity_id": self.target_entity}
        action_params.update(self.action_params)

        # Pour get_state, utiliser ha.get_states (tool MCP) sans paramètres
        if self.action == "get_state":
            action_call = "<start_function_call>call:ha.get_states{}<end_function_call>"
            states_response = self._format_state_response()
            text = (
                f"<start_of_turn>user\n{self.user_query}<end_of_turn>\n"
                f"<start_of_turn>model\n{action_call}<end_of_turn>\n"
                f"<start_of_turn>tool\n{states_response}<end_of_turn>"
            )
        else:
            action_call = format_function_call(
                f"{self.domain}.{self.action}",
                action_params
            )
            text = (
                f"<start_of_turn>user\n{self.user_query}<end_of_turn>\n"
                f"<start_of_turn>model\n{action_call}<end_of_turn>"
            )

        return {"text": text}

    def to_one_step_format(self) -> dict:
        """
        Convertit en format d'entraînement one-step.

        Pattern simplifié:
        1. User demande une action + liste des entités disponibles (TOUS les domaines)
        2. Model appelle directement l'action avec la bonne entité

        Ce format inclut TOUTES les entités de tous les domaines pour forcer le modèle
        à choisir la bonne entité parmi toutes les options disponibles.
        """
        # Construire la liste de toutes les entités par domaine
        entities_sections = []
        if self.all_entities_by_domain:
            # Utiliser toutes les entités de tous les domaines
            for domain in ["light", "switch", "climate", "scene", "person", "cover", "lock", "fan"]:
                if domain in self.all_entities_by_domain:
                    domain_entities = self.all_entities_by_domain[domain][:12]  # Limiter à 12 par domaine
                    if domain_entities:
                        entities_list = ", ".join(domain_entities)
                        entities_sections.append(f"Entités {domain} disponibles: {entities_list}")
        else:
            # Fallback: utiliser seulement les entités du domaine actuel
            entities_list = ", ".join(self.available_entities[:10])
            entities_sections.append(f"Entités {self.domain} disponibles: {entities_list}")

        entities_block = "\n".join(entities_sections)
        user_prompt = f"{self.user_query}\n\n{entities_block}"

        # Appel de l'action directe
        action_params = {"entity_id": self.target_entity}
        action_params.update(self.action_params)

        # Pour get_state, utiliser ha.get_states (tool MCP) sans paramètres
        if self.action == "get_state":
            action_call = "<start_function_call>call:ha.get_states{}<end_function_call>"
            states_response = self._format_state_response()
            text = (
                f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n{action_call}<end_of_turn>\n"
                f"<start_of_turn>tool\n{states_response}<end_of_turn>"
            )
        else:
            action_call = format_function_call(
                f"{self.domain}.{self.action}",
                action_params
            )
            text = (
                f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n{action_call}<end_of_turn>"
            )

        return {"text": text}


@dataclass
class NegativeExample:
    """Un exemple négatif (entité non trouvée, requête ambiguë, etc.)."""
    user_query: str
    error_type: str  # "entity_not_found", "clarification_needed", "out_of_scope"
    error_message: str

    def to_training_format(self) -> dict:
        """
        Convertit en format d'entraînement avec appel de fonction erreur.
        """
        # Appel de fonction erreur
        error_call = format_function_call(
            f"error.{self.error_type}",
            {"message": self.error_message}
        )

        # Format texte pour l'entraînement
        text = (
            f"<start_of_turn>user\n{self.user_query}<end_of_turn>\n"
            f"<start_of_turn>model\n{error_call}<end_of_turn>"
        )

        return {"text": text}


@dataclass
class NoActionExample:
    """
    Un exemple où le modèle ne doit PAS exécuter de commande.

    Pour les requêtes vagues ou ambiguës (comme "Allume le chauffage" sans location),
    le modèle doit apprendre à ne rien faire - pas de function call.
    L'utilisateur devra reformuler sa demande avec plus de précision.
    """
    user_query: str
    available_entities: list[str] = field(default_factory=list)
    all_entities_by_domain: dict = field(default_factory=dict)

    def to_training_format(self) -> dict:
        """
        Convertit en format d'entraînement SANS appel de fonction.
        Le modèle apprend à ne rien répondre pour ces requêtes.
        """
        # Format texte avec réponse vide du modèle
        text = (
            f"<start_of_turn>user\n{self.user_query}<end_of_turn>\n"
            f"<start_of_turn>model\n<end_of_turn>"
        )
        return {"text": text}

    def to_one_step_format(self) -> dict:
        """
        Format one-step avec toutes les entités listées mais SANS action.

        Même avec les entités disponibles, si la requête est vague,
        le modèle doit apprendre à ne pas agir.
        """
        # Construire la liste de toutes les entités par domaine
        entities_sections = []
        if self.all_entities_by_domain:
            for domain in ["light", "switch", "climate", "scene", "person", "cover", "lock", "fan"]:
                if domain in self.all_entities_by_domain:
                    domain_entities = self.all_entities_by_domain[domain][:12]
                    if domain_entities:
                        entities_list = ", ".join(domain_entities)
                        entities_sections.append(f"Entités {domain} disponibles: {entities_list}")

        entities_block = "\n".join(entities_sections)
        user_prompt = f"{self.user_query}\n\n{entities_block}" if entities_block else self.user_query

        # Pas de function call - réponse vide
        text = (
            f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n<end_of_turn>"
        )
        return {"text": text}


class DatasetGenerator:
    """Génère un dataset de fine-tuning multi-turn pour FunctionGemma."""

    def __init__(
        self,
        entities: list[dict],
        examples_per_action: int = 20,
        examples_per_domain: int = 100,  # Limite par domaine pour équilibrer
        negative_examples_multiplier: int = 3,  # Multiplier pour générer plus de négatifs
        seed: int = 42,
        filter_technical: bool = True,  # Filtrer les entités techniques
    ):
        self.examples_per_action = examples_per_action
        self.examples_per_domain = examples_per_domain
        self.negative_examples_multiplier = negative_examples_multiplier
        self.examples: list[MultiTurnExample] = []
        self.negative_examples: list[NegativeExample] = []
        self.no_action_examples: list[NoActionExample] = []

        random.seed(seed)

        # Filtrer les entités techniques si demandé
        if filter_technical:
            print("Filtrage des entités techniques...")
            entities = filter_entities(entities)

        self.entities = entities

        # Indexer les entités par domaine
        self.entities_by_domain: dict[str, list[dict]] = {}
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else ""
            if domain:
                if domain not in self.entities_by_domain:
                    self.entities_by_domain[domain] = []
                self.entities_by_domain[domain].append(entity)

        # Afficher le résumé par domaine
        print("Entités par domaine:")
        for domain, domain_entities in sorted(self.entities_by_domain.items()):
            print(f"  {domain}: {len(domain_entities)}")

    def _get_entity_name(self, entity: dict) -> str:
        """Extrait un nom lisible pour une entité."""
        attrs = entity.get("attributes", {})
        friendly_name = attrs.get("friendly_name", "")
        if friendly_name:
            return friendly_name
        return entity.get("entity_id", "").split(".")[-1].replace("_", " ")

    def _get_entity_ids(self, domain: str) -> list[str]:
        """Retourne la liste des entity_ids pour un domaine."""
        return [e["entity_id"] for e in self.entities_by_domain.get(domain, [])]

    def _get_all_entity_ids_by_domain(self) -> dict[str, list[str]]:
        """Retourne un dictionnaire de tous les entity_ids par domaine."""
        return {
            domain: [e["entity_id"] for e in entities]
            for domain, entities in self.entities_by_domain.items()
        }

    def _generate_domain_examples(self, domain: str) -> list[MultiTurnExample]:
        """Génère des exemples pour un domaine (limité pour équilibrage)."""
        examples = []
        domain_entities = self.entities_by_domain.get(domain, [])

        if not domain_entities:
            return examples

        templates = TEMPLATES_FR.get(domain, {})
        available_entity_ids = self._get_entity_ids(domain)

        # Récupérer toutes les entités de tous les domaines pour le format one-step
        all_entities_by_domain = self._get_all_entity_ids_by_domain()

        for action, action_templates in templates.items():
            # Pour chaque entité, générer plusieurs exemples avec différents templates
            for entity in domain_entities[:self.examples_per_action]:
                entity_id = entity["entity_id"]
                entity_name = self._get_entity_name(entity)

                # Extraire la vraie localisation depuis le nom de l'entité
                entity_location = extract_location_from_entity(entity_id)

                # Utiliser TOUS les templates pour plus de variété
                for template in action_templates:
                    action_params = {}

                    # Pour les templates avec {location}, n'utiliser que si on a une vraie location
                    if "{location}" in template:
                        if entity_location is None:
                            # Pas de location connue, skip ce template
                            continue
                        location = entity_location
                    else:
                        location = ""  # Non utilisé

                    if "{brightness}" in template:
                        brightness = random.choice([10, 25, 50, 75, 100])
                        query = template.format(
                            entity_name=entity_name,
                            location=location,
                            brightness=brightness
                        )
                        action_params["brightness_pct"] = brightness
                        actual_action = "turn_on"
                    elif "{temperature}" in template:
                        temperature = random.choice([18, 19, 20, 21, 22, 23, 24])
                        query = template.format(
                            location=location,
                            temperature=temperature
                        )
                        action_params["temperature"] = temperature
                        actual_action = action
                    elif "{mode}" in template:
                        mode_fr = random.choice(list(HVAC_MODES_FR.keys()))
                        query = template.format(mode=mode_fr, location=location)
                        action_params["hvac_mode"] = HVAC_MODES_FR[mode_fr]
                        actual_action = action
                    elif "{position}" in template:
                        position = random.choice([25, 50, 75])
                        query = template.format(
                            entity_name=entity_name,
                            location=location,
                            position=position
                        )
                        action_params["position"] = position
                        actual_action = action
                    else:
                        query = template.format(
                            entity_name=entity_name,
                            location=location
                        )
                        actual_action = action

                    # Version normale
                    examples.append(MultiTurnExample(
                        user_query=query,
                        domain=domain,
                        available_entities=available_entity_ids,
                        target_entity=entity_id,
                        action=actual_action,
                        action_params=action_params.copy(),
                        all_entities_by_domain=all_entities_by_domain,
                    ))

                    # Version avec fautes de frappe (50% du temps)
                    if random.random() < 0.5:
                        typo_query = add_typos(query, probability=1.0)
                        if typo_query != query:  # Seulement si différent
                            examples.append(MultiTurnExample(
                                user_query=typo_query,
                                domain=domain,
                                available_entities=available_entity_ids,
                                target_entity=entity_id,
                                action=actual_action,
                                action_params=action_params.copy(),
                                all_entities_by_domain=all_entities_by_domain,
                            ))

                    # Version avec préfixe de politesse (30% du temps)
                    if random.random() < 0.3:
                        prefix = random.choice(POLITENESS_PREFIXES)
                        if prefix:
                            polite_query = prefix + query[0].lower() + query[1:]
                            examples.append(MultiTurnExample(
                                user_query=polite_query,
                                domain=domain,
                                available_entities=available_entity_ids,
                                target_entity=entity_id,
                                action=actual_action,
                                action_params=action_params.copy(),
                                all_entities_by_domain=all_entities_by_domain,
                            ))

                    # Version avec suffixe contextuel (20% du temps)
                    if random.random() < 0.2:
                        suffix = random.choice(CONTEXT_SUFFIXES)
                        if suffix:
                            # Enlever le point ou ? à la fin si présent
                            base_query = query.rstrip("?.! ")
                            context_query = base_query + suffix
                            examples.append(MultiTurnExample(
                                user_query=context_query,
                                domain=domain,
                                available_entities=available_entity_ids,
                                target_entity=entity_id,
                                action=actual_action,
                                action_params=action_params.copy(),
                                all_entities_by_domain=all_entities_by_domain,
                            ))

        # Générer des exemples de confusion (entités similaires)
        confusion_examples = self._generate_confusion_examples(domain, available_entity_ids, all_entities_by_domain)
        examples.extend(confusion_examples)

        # Mélanger et limiter pour équilibrer les domaines
        random.shuffle(examples)
        if len(examples) > self.examples_per_domain:
            examples = examples[:self.examples_per_domain]

        return examples

    def _generate_confusion_examples(
        self,
        domain: str,
        entity_ids: list[str],
        all_entities_by_domain: dict[str, list[str]]
    ) -> list[MultiTurnExample]:
        """
        Génère des exemples avec des entités similaires pour forcer le modèle à bien distinguer.
        Ex: "lumière du salon" vs "lumière de la salle à manger"
        """
        examples = []

        if len(entity_ids) < 2:
            return examples

        # Trouver des paires d'entités similaires (même préfixe ou contenant des mots similaires)
        similar_pairs = []
        for i, e1 in enumerate(entity_ids):
            for e2 in entity_ids[i+1:]:
                name1 = e1.split(".")[-1].lower()
                name2 = e2.split(".")[-1].lower()

                # Vérifier si les noms partagent des mots
                words1 = set(name1.replace("_", " ").split())
                words2 = set(name2.replace("_", " ").split())
                common_words = words1 & words2

                if common_words and len(common_words) < min(len(words1), len(words2)):
                    similar_pairs.append((e1, e2))

        # Générer des exemples de confusion pour chaque paire
        confusion_templates = {
            "light": [
                "Allume la lumière {location}, pas {other_location}",
                "C'est {location} que je veux allumer, pas {other_location}",
                "Éteins seulement {location}",
                "Juste la lumière {location}",
            ],
            "climate": [
                "Mets le chauffage {location} à {temperature} degrés",
                "Change la température {location}, pas {other_location}",
            ],
            "cover": [
                "Ouvre les volets {location}, pas {other_location}",
                "Ferme seulement les volets {location}",
            ],
        }

        templates = confusion_templates.get(domain, [])
        if not templates:
            return examples

        for e1, e2 in similar_pairs[:10]:  # Limiter à 10 paires
            loc1 = extract_location_from_entity(e1)
            loc2 = extract_location_from_entity(e2)

            if not loc1 or not loc2:
                continue

            for template in templates:
                if "{temperature}" in template:
                    temp = random.choice([19, 20, 21, 22])
                    query = template.format(
                        location=loc1,
                        other_location=loc2,
                        temperature=temp
                    )
                    action_params = {"temperature": temp}
                    action = "set_temperature"
                else:
                    query = template.format(location=loc1, other_location=loc2)
                    action_params = {}
                    action = "turn_on" if "allume" in template.lower() or "ouvre" in template.lower() else "turn_off"
                    if "ouvre" in template.lower():
                        action = "open_cover"
                    elif "ferme" in template.lower():
                        action = "close_cover"

                examples.append(MultiTurnExample(
                    user_query=query,
                    domain=domain,
                    available_entities=[e1, e2] + entity_ids[:5],  # Inclure les deux entités
                    target_entity=e1,  # La première est la cible
                    action=action,
                    action_params=action_params,
                    all_entities_by_domain=all_entities_by_domain,
                ))

        return examples

    def _generate_negative_examples(self) -> list[NegativeExample]:
        """Génère des exemples négatifs (entités non trouvées, requêtes ambiguës)."""
        examples = []

        # Générer plusieurs instances de chaque template négatif
        for _ in range(self.negative_examples_multiplier):
            # Entités non trouvées
            for query, error_type, message in NEGATIVE_TEMPLATES:
                # Version normale
                examples.append(NegativeExample(
                    user_query=query,
                    error_type=error_type,
                    error_message=message,
                ))

                # Version avec fautes de frappe (50% du temps)
                if random.random() < 0.5:
                    typo_query = add_typos(query, probability=1.0)
                    if typo_query != query:
                        examples.append(NegativeExample(
                            user_query=typo_query,
                            error_type=error_type,
                            error_message=message,
                        ))

            # Requêtes ambiguës/incomplètes
            for query, error_type, message in AMBIGUOUS_TEMPLATES:
                examples.append(NegativeExample(
                    user_query=query,
                    error_type=error_type,
                    error_message=message,
                ))

                # Version minuscule pour les requêtes courtes
                if len(query) < 20 and random.random() < 0.5:
                    examples.append(NegativeExample(
                        user_query=query.lower(),
                        error_type=error_type,
                        error_message=message,
                    ))

        random.shuffle(examples)
        return examples

    def _generate_no_action_examples(self) -> list[NoActionExample]:
        """
        Génère des exemples sans action pour les requêtes vagues.

        Ces exemples enseignent au modèle à ne PAS exécuter de commande
        quand la requête manque de spécificité (pas de location).
        """
        examples = []
        all_entities_by_domain = self._get_all_entity_ids_by_domain()

        # Générer plusieurs instances de chaque template
        for _ in range(self.negative_examples_multiplier):
            for query in NO_ACTION_TEMPLATES:
                # Version normale
                examples.append(NoActionExample(
                    user_query=query,
                    all_entities_by_domain=all_entities_by_domain,
                ))

                # Version avec fautes de frappe (50% du temps)
                if random.random() < 0.5:
                    typo_query = add_typos(query, probability=1.0)
                    if typo_query != query:
                        examples.append(NoActionExample(
                            user_query=typo_query,
                            all_entities_by_domain=all_entities_by_domain,
                        ))

                # Version minuscule (30% du temps)
                if random.random() < 0.3:
                    examples.append(NoActionExample(
                        user_query=query.lower(),
                        all_entities_by_domain=all_entities_by_domain,
                    ))

        random.shuffle(examples)
        return examples

    def generate_all(self) -> list[MultiTurnExample]:
        """Génère tous les exemples d'entraînement."""
        print("Génération du dataset multi-turn...")

        all_examples = []

        domains = ["light", "switch", "climate", "cover", "lock", "scene", "fan", "person"]

        for domain in tqdm(domains, desc="Domaines"):
            examples = self._generate_domain_examples(domain)
            print(f"  {domain}: {len(examples)} exemples")
            all_examples.extend(examples)

        # Mélanger
        random.shuffle(all_examples)

        self.examples = all_examples
        print(f"\nTotal exemples positifs: {len(all_examples)}")

        # Générer les exemples négatifs
        print("\nGénération des exemples négatifs...")
        self.negative_examples = self._generate_negative_examples()
        print(f"Total exemples négatifs: {len(self.negative_examples)}")

        # Générer les exemples sans action (requêtes vagues)
        print("\nGénération des exemples sans action (requêtes vagues)...")
        self.no_action_examples = self._generate_no_action_examples()
        print(f"Total exemples sans action: {len(self.no_action_examples)}")

        total = len(all_examples) + len(self.negative_examples) + len(self.no_action_examples)
        print(f"\nTotal général: {total} exemples")

        return all_examples

    def save_dataset(self, output_dir: str, val_split: float = 0.1, include_one_step: bool = True):
        """Sauvegarde le dataset au format JSON Lines."""
        os.makedirs(output_dir, exist_ok=True)

        # Split train/val pour les exemples positifs
        n_val = int(len(self.examples) * val_split)
        val_examples = self.examples[:n_val]
        train_examples = self.examples[n_val:]

        # Split train/val pour les exemples négatifs
        n_neg_val = int(len(self.negative_examples) * val_split)
        neg_val_examples = self.negative_examples[:n_neg_val]
        neg_train_examples = self.negative_examples[n_neg_val:]

        # Split train/val pour les exemples sans action
        n_noact_val = int(len(self.no_action_examples) * val_split)
        noact_val_examples = self.no_action_examples[:n_noact_val]
        noact_train_examples = self.no_action_examples[n_noact_val:]

        # Sauvegarder
        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")

        train_count = 0
        val_count = 0

        with open(train_path, "w", encoding="utf-8") as f:
            # Exemples positifs
            for example in train_examples:
                # Format multi-turn (get_entities → action)
                data = example.to_training_format()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                train_count += 1

                # Format one-step (entités dans le prompt → action directe)
                if include_one_step:
                    data_one_step = example.to_one_step_format()
                    f.write(json.dumps(data_one_step, ensure_ascii=False) + "\n")
                    train_count += 1

                # Format single-turn (sans get_entities, direct)
                data_single = example.to_single_turn_format()
                f.write(json.dumps(data_single, ensure_ascii=False) + "\n")
                train_count += 1

            # Exemples négatifs
            for neg_example in neg_train_examples:
                data = neg_example.to_training_format()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                train_count += 1

            # Exemples sans action (requêtes vagues → pas de function call)
            for noact_example in noact_train_examples:
                # Format simple (sans entités)
                data = noact_example.to_training_format()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                train_count += 1

                # Format one-step (avec entités listées mais toujours pas d'action)
                if include_one_step:
                    data_one_step = noact_example.to_one_step_format()
                    f.write(json.dumps(data_one_step, ensure_ascii=False) + "\n")
                    train_count += 1

        with open(val_path, "w", encoding="utf-8") as f:
            # Exemples positifs
            for example in val_examples:
                # Format multi-turn
                data = example.to_training_format()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                val_count += 1

                # Format one-step
                if include_one_step:
                    data_one_step = example.to_one_step_format()
                    f.write(json.dumps(data_one_step, ensure_ascii=False) + "\n")
                    val_count += 1

                # Format single-turn
                data_single = example.to_single_turn_format()
                f.write(json.dumps(data_single, ensure_ascii=False) + "\n")
                val_count += 1

            # Exemples négatifs
            for neg_example in neg_val_examples:
                data = neg_example.to_training_format()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                val_count += 1

            # Exemples sans action
            for noact_example in noact_val_examples:
                data = noact_example.to_training_format()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                val_count += 1

                if include_one_step:
                    data_one_step = noact_example.to_one_step_format()
                    f.write(json.dumps(data_one_step, ensure_ascii=False) + "\n")
                    val_count += 1

        print(f"Dataset sauvegardé:")
        print(f"  Train: {train_path} ({train_count} exemples)")
        print(f"  Val: {val_path} ({val_count} exemples)")
        print(f"  (inclut: multi-turn, one-step, single-turn, négatifs, et sans-action)")

        # Afficher des exemples
        if train_examples:
            print(f"\nExemple positif (single-turn):")
            sample = train_examples[0].to_single_turn_format()
            print(sample["text"])

        if neg_train_examples:
            print(f"\nExemple négatif:")
            sample_neg = neg_train_examples[0].to_training_format()
            print(sample_neg["text"])

        if noact_train_examples:
            print(f"\nExemple sans action (requête vague):")
            sample_noact = noact_train_examples[0].to_training_format()
            print(sample_noact["text"])


async def main():
    """Génère le dataset depuis Home Assistant."""
    from ha_client import HomeAssistantClient

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Connexion à Home Assistant
    client = HomeAssistantClient.from_env(config["home_assistant"]["url"])

    print("Récupération des données de Home Assistant...")
    await client.build_function_schemas()
    entities = client.entities

    print(f"  Entités: {len(entities)}")

    # Générer le dataset
    generator = DatasetGenerator(
        entities=entities,
        examples_per_action=config["dataset"].get("examples_per_function", 20),
        examples_per_domain=config["dataset"].get("examples_per_domain", 100),
        seed=config["dataset"]["seed"]
    )

    generator.generate_all()

    # Sauvegarder
    generator.save_dataset(
        output_dir=config["dataset"]["output_dir"],
        val_split=config["evaluation"]["val_split"]
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
