"""
Générateur de dataset pour fine-tuner FunctionGemma sur Home Assistant.
Génère des exemples multi-turn avec pattern get_entities -> action.

Le modèle apprend à:
1. D'abord récupérer les entités disponibles (get_entities)
2. Puis appeler la fonction appropriée avec la bonne entité
"""

import os
import json
import random
from typing import Optional
from dataclasses import dataclass, field

import yaml
from tqdm import tqdm


# ============================================================================
# FILTRAGE DES ENTITÉS - Garder uniquement les entités utiles
# ============================================================================

# Domaines utiles pour la domotique vocale
USEFUL_DOMAINS = ['light', 'switch', 'climate', 'scene', 'cover', 'fan', 'lock', 'person']

# Patterns à exclure (entités système, diagnostics, etc.)
ENTITY_EXCLUSIONS = [
    # Updates et OTA
    '_update', '_auto_update', '_ota', '_prerelease', '_firmware',
    # Devices ESPHome/ESP internes
    'espresense_', 'esphome_', '_esp_', '_esphome',
    # AdGuard et autres services
    'adguard_', 'hacs_', 'supervisor_',
    # Diagnostics et capteurs système
    '_battery', '_signal', '_linkquality', '_rssi', '_voltage',
    '_power_on_behavior', '_do_not_disturb', '_led_',
    # Autres patterns système
    '_restart', '_identify', '_debug', '_test',
    '_unavailable', '_unknown',
]


def filter_entities(entities: list[dict]) -> list[dict]:
    """
    Filtre les entités pour ne garder que celles utiles à l'entraînement.

    Args:
        entities: Liste des entités depuis Home Assistant

    Returns:
        Liste filtrée des entités utiles
    """
    filtered = []

    for entity in entities:
        entity_id = entity.get("entity_id", "").lower()

        # Extraire le domaine
        if "." not in entity_id:
            continue
        domain = entity_id.split(".")[0]

        # Vérifier si le domaine est utile
        if domain not in USEFUL_DOMAINS:
            continue

        # Vérifier les exclusions
        if any(excl in entity_id for excl in ENTITY_EXCLUSIONS):
            continue

        filtered.append(entity)

    return filtered


def print_entity_summary(entities: list[dict], filtered: list[dict]) -> None:
    """Affiche un résumé du filtrage des entités."""
    print(f"\n📊 Filtrage des entités:")
    print(f"  Avant: {len(entities)} entités")
    print(f"  Après: {len(filtered)} entités")
    print(f"  Exclues: {len(entities) - len(filtered)}")

    # Compter par domaine
    domain_counts = {}
    for entity in filtered:
        domain = entity.get("entity_id", "").split(".")[0]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print(f"\n  Par domaine:")
    for domain in sorted(domain_counts.keys()):
        print(f"    {domain}: {domain_counts[domain]}")


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
            # Québécois - avec "lumière" explicite pour éviter confusion avec volets
            "Ferme la lumière {location}",
            "Ferme les lumières {location}",
            "Ferme la light {location}",
            # Nouvelles variations
            "Enlève la lumière {location}",
            "Arrête la lumière {location}",
            "Stop la lumière {location}",
            "Noir {location}",
            "Plus besoin de lumière {location}",
            "Désactive la lumière {location}",
            "Coupe la lumière {location}",
            "Éteins-moi la lumière {location}",
            "Fais le noir {location}",
            "Lumière {location} éteinte",
            "Éteins l'éclairage {location}",
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
            "Allume la {entity_name}",
            "Active le {entity_name}",
            "Active la {entity_name}",
            "Turn on {entity_name}",
            "Enclenche {entity_name}",
            "Branche {entity_name}",
        ],
        "turn_off": [
            "Éteins {entity_name}",
            "Désactive {entity_name}",
            "Arrête {entity_name}",
            "Coupe {entity_name}",
            "Stoppe {entity_name}",
            "Éteins le {entity_name}",
            "Éteins la {entity_name}",
            "Turn off {entity_name}",
            "Débranche {entity_name}",
            "Coupe le {entity_name}",
            "Coupe la {entity_name}",
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
        "set_temperature": [
            "Mets le chauffage à {temperature} degrés",
            "Règle la température à {temperature}°C",
            "Je veux {temperature} degrés",
            "Température à {temperature} degrés",
            "Chauffe à {temperature}°C",
            "Monte le chauffage à {temperature}",
            "Baisse la température à {temperature}",
            "Met {temperature} degrés",
            "{temperature} degrés s'il te plaît",
            "Thermostat à {temperature}",
            "Chauffage à {temperature}",
            "Set temperature to {temperature}",
            "Met le à {temperature}",
            "Monte à {temperature}",
            "Descend à {temperature}",
            # Nouvelles variations
            "Je veux qu'il fasse {temperature} degrés",
            "Augmente à {temperature}",
            "Diminue à {temperature}",
            "Règle à {temperature}",
            "{temperature}° dans la maison",
            "Chauffe la maison à {temperature}",
            "Refroidis à {temperature} degrés",
            "Mets-moi {temperature} degrés",
            "On met {temperature}",
            "Change la température à {temperature}",
            "Ajuste le thermostat à {temperature}",
            "Configure {temperature} degrés",
        ],
        "set_hvac_mode": [
            "Mets le thermostat en mode {mode}",
            "Passe en mode {mode}",
            "Active le mode {mode}",
            "Mode {mode}",
            "Met en {mode}",
            # Nouvelles variations
            "Change le mode en {mode}",
            "Bascule en {mode}",
            "Je veux le mode {mode}",
            "Thermostat en {mode}",
            "Passe le chauffage en {mode}",
        ],
        "turn_on": [
            "Allume le chauffage",
            "Démarre la climatisation",
            "Active le thermostat",
            "Ouvre le chauffage",
            "Part le chauffage",
            "Démarre le chauffage",
            "Met le chauffage",
            "Allume la clim",
            "Ouvre la clim",
            # Nouvelles variations
            "Lance le chauffage",
            "J'ai froid",
            "J'ai chaud",
            "Chauffe",
            "Refroidis",
            "Active la climatisation",
            "Mets la clim",
            "Je gèle",
            "On crève de chaud",
        ],
        "turn_off": [
            "Éteins le chauffage",
            "Éteins la climatisation",
            "Éteins la clim",
            "Éteins le thermostat",
            "Arrête le chauffage",
            "Arrête la climatisation",
            "Arrête la clim",
            "Arrête le thermostat",
            "Coupe le chauffage",
            "Coupe la climatisation",
            "Coupe la clim",
            "Coupe le thermostat",
            # Nouvelles variations
            "Stop le chauffage",
            "Stop la clim",
            "Désactive le thermostat",
            "Désactive le chauffage",
            "Désactive la climatisation",
            "Plus de chauffage",
            "Plus de clim",
            "Arrête de chauffer",
            "Arrête de refroidir",
        ],
        "get_state": [
            "Quelle est la température {location} ?",
            "Il fait combien {location} ?",
            "Quelle température fait-il {location} ?",
            "Le chauffage est allumé ?",
            "Quel est le mode du thermostat ?",
            "C'est à combien {location} ?",
            "Il fait chaud {location} ?",
            "Il fait froid {location} ?",
            "Température {location} ?",
            "What's the temperature?",
            # Nouvelles variations
            "Combien de degrés {location} ?",
            "La température actuelle ?",
            "Quel temps fait-il à l'intérieur ?",
            "Statut du chauffage",
            "Le thermostat est à combien ?",
            "État de la climatisation",
        ],
    },
    "cover": {
        "open_cover": [
            # Templates avec "volets" - mot-clé discriminant
            "Ouvre les volets {location}",
            "Ouvre les volets",
            "Lève les volets {location}",
            "Monte les volets {location}",
            "Relève les volets {location}",
            "Remonte les volets {location}",
            "Volets {location} ouverts",
            "Volets ouverts {location}",
            "Ouvre-moi les volets {location}",
            "Peux-tu ouvrir les volets {location}",
            # Templates avec "stores" - mot-clé discriminant
            "Ouvre les stores {location}",
            "Lève les stores {location}",
            "Monte les stores {location}",
            "Remonte les stores {location}",
            "Stores {location} ouverts",
            # Templates avec "persiennes"
            "Ouvre les persiennes {location}",
            "Lève les persiennes {location}",
            # Templates génériques (moins prioritaires)
            "Ouvre {entity_name}",
            "Lève {entity_name}",
            "Monte {entity_name}",
            "Je veux voir dehors {location}",
            "Fais entrer le soleil {location}",
            "Laisse entrer la lumière du jour {location}",
        ],
        "close_cover": [
            # Templates avec "volets" - mot-clé discriminant
            "Ferme les volets {location}",
            "Ferme les volets",
            "Baisse les volets {location}",
            "Descends les volets {location}",
            "Abaisse les volets {location}",
            "Volets {location} fermés",
            "Volets fermés {location}",
            "Ferme-moi les volets {location}",
            "Peux-tu fermer les volets {location}",
            # Templates avec "stores" - mot-clé discriminant
            "Ferme les stores {location}",
            "Baisse les stores {location}",
            "Descends les stores {location}",
            "Abaisse les stores {location}",
            "Stores {location} fermés",
            # Templates avec "persiennes"
            "Ferme les persiennes {location}",
            "Baisse les persiennes {location}",
            # Templates génériques
            "Ferme {entity_name}",
            "Baisse {entity_name}",
            "Descends {entity_name}",
            "Cache le soleil {location}",
            "Bloque la lumière du soleil {location}",
            "Il y a trop de soleil {location}",
        ],
        "set_cover_position": [
            "Mets les volets {location} à {position}%",
            "Volets {location} à {position}%",
            "Volets {location} à {position} pourcent",
            "Position des volets {location} à {position}%",
            "Règle les volets {location} à {position}%",
            "Ouvre les volets {location} à {position}%",
            "Ferme les volets {location} à {position}%",
            "Mets les stores {location} à {position}%",
            "Stores {location} à {position}%",
            "Ouvre {entity_name} à {position}%",
            "{position} pourcent les volets {location}",
            "{position}% les volets {location}",
        ],
        "get_state": [
            "Les volets {location} sont ouverts ?",
            "Les volets {location} sont fermés ?",
            "Est-ce que les volets {location} sont ouverts ?",
            "Les stores {location} sont ouverts ?",
            "Les stores {location} sont fermés ?",
            "État des volets {location}",
            "Position des volets {location} ?",
            "Les volets sont ouverts ou fermés {location} ?",
            "{entity_name} est ouvert ?",
            "{entity_name} est fermé ?",
            "À combien sont les volets {location} ?",
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
            # Templates avec "scène" - mot-clé discriminant principal
            "Active la scène {entity_name}",
            "Lance la scène {entity_name}",
            "Démarre la scène {entity_name}",
            "Exécute la scène {entity_name}",
            "Mets la scène {entity_name}",
            "Scène {entity_name}",
            "Scène {entity_name} s'il te plaît",
            "Je veux la scène {entity_name}",
            "Mets-moi la scène {entity_name}",
            "Applique la scène {entity_name}",
            "Charge la scène {entity_name}",
            "Configure la scène {entity_name}",
            # Templates avec "ambiance" - mot-clé discriminant secondaire
            "Mets l'ambiance {entity_name}",
            "Ambiance {entity_name}",
            "Je veux l'ambiance {entity_name}",
            "Ambiance {entity_name} s'il te plaît",
            "Active l'ambiance {entity_name}",
            "Lance l'ambiance {entity_name}",
            # Templates avec "mode" pour les scènes
            "Passe en mode {entity_name}",
            "Mode {entity_name}",
            "Active le mode {entity_name}",
            "Mets le mode {entity_name}",
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
    ("Allume", "clarification_needed", "Précisez ce que vous voulez allumer (lumière, appareil, etc.)"),
    ("Éteins", "clarification_needed", "Précisez ce que vous voulez éteindre (lumière, appareil, etc.)"),
    ("Allume tout", "clarification_needed", "Précisez quelles lumières vous voulez allumer"),
    ("Éteins tout", "clarification_needed", "Précisez quelles lumières vous voulez éteindre"),
    ("Ouvre", "clarification_needed", "Précisez ce que vous voulez ouvrir (volets, serrure, etc.)"),
    ("Ferme", "clarification_needed", "Précisez ce que vous voulez fermer (volets, serrure, etc.)"),
    ("Ferme tout", "clarification_needed", "Précisez ce que vous voulez fermer (volets, lumières, etc.)"),
    ("Mets le chauffage", "clarification_needed", "Précisez la température souhaitée"),
    ("Monte le chauffage", "clarification_needed", "Précisez la température souhaitée"),
    ("Baisse", "clarification_needed", "Précisez ce que vous voulez baisser (volets, température, lumière)"),
    ("Active", "clarification_needed", "Précisez ce que vous voulez activer (scène, appareil, etc.)"),
    ("Démarre", "clarification_needed", "Précisez ce que vous voulez démarrer"),
    ("Change", "clarification_needed", "Précisez ce que vous voulez changer"),
    ("Règle", "clarification_needed", "Précisez ce que vous voulez régler"),
    ("Met", "clarification_needed", "Précisez ce que vous voulez mettre"),
    ("Stop", "clarification_needed", "Précisez ce que vous voulez arrêter"),
    ("Lance", "clarification_needed", "Précisez ce que vous voulez lancer (scène, appareil, etc.)"),
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
    available_entities: list[str]  # Liste des entity_ids disponibles
    target_entity: str  # L'entité choisie
    action: str  # ex: "turn_on", "set_temperature"
    action_params: dict  # Paramètres additionnels (brightness, temperature, etc.)

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
        1. User demande une action + liste des entités disponibles
        2. Model appelle directement l'action avec la bonne entité
        """
        # Liste des entités disponibles dans le prompt
        entities_list = ", ".join(self.available_entities[:10])
        user_prompt = f"{self.user_query}\n\nEntités {self.domain} disponibles: {entities_list}"

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


class DatasetGenerator:
    """Génère un dataset de fine-tuning multi-turn pour FunctionGemma."""

    def __init__(
        self,
        entities: list[dict],
        examples_per_action: int = 20,
        examples_per_domain: int = 100,  # Limite par domaine pour équilibrer
        negative_examples_multiplier: int = 3,  # Multiplier pour générer plus de négatifs
        seed: int = 42
    ):
        self.entities = entities
        self.examples_per_action = examples_per_action
        self.examples_per_domain = examples_per_domain
        self.negative_examples_multiplier = negative_examples_multiplier
        self.examples: list[MultiTurnExample] = []
        self.negative_examples: list[NegativeExample] = []

        random.seed(seed)

        # Indexer les entités par domaine
        self.entities_by_domain: dict[str, list[dict]] = {}
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else ""
            if domain:
                if domain not in self.entities_by_domain:
                    self.entities_by_domain[domain] = []
                self.entities_by_domain[domain].append(entity)

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

    def _generate_domain_examples(self, domain: str) -> list[MultiTurnExample]:
        """Génère des exemples pour un domaine (limité pour équilibrage)."""
        examples = []
        domain_entities = self.entities_by_domain.get(domain, [])

        if not domain_entities:
            return examples

        templates = TEMPLATES_FR.get(domain, {})
        available_entity_ids = self._get_entity_ids(domain)

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
                        query = template.format(mode=mode_fr)
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
                            ))

        # Générer des exemples de confusion (entités similaires)
        confusion_examples = self._generate_confusion_examples(domain, available_entity_ids)
        examples.extend(confusion_examples)

        # Mélanger et limiter pour équilibrer les domaines
        random.shuffle(examples)
        if len(examples) > self.examples_per_domain:
            examples = examples[:self.examples_per_domain]

        return examples

    def _generate_confusion_examples(self, domain: str, entity_ids: list[str]) -> list[MultiTurnExample]:
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

        print(f"\nTotal général: {len(all_examples) + len(self.negative_examples)} exemples")

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

        print(f"Dataset sauvegardé:")
        print(f"  Train: {train_path} ({train_count} exemples)")
        print(f"  Val: {val_path} ({val_count} exemples)")
        print(f"  (inclut: multi-turn, one-step, single-turn, et négatifs)")

        # Afficher des exemples
        if train_examples:
            print(f"\nExemple positif (single-turn):")
            sample = train_examples[0].to_single_turn_format()
            print(sample["text"])

        if neg_train_examples:
            print(f"\nExemple négatif:")
            sample_neg = neg_train_examples[0].to_training_format()
            print(sample_neg["text"])


async def main():
    """Génère le dataset depuis Home Assistant."""
    from ha_client import HomeAssistantClient

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Connexion à Home Assistant
    client = HomeAssistantClient.from_env(config["home_assistant"]["url"])

    print("Récupération des données de Home Assistant...")
    await client.build_function_schemas()
    raw_entities = client.entities

    # Filtrer les entités pour ne garder que celles utiles
    entities = filter_entities(raw_entities)
    print_entity_summary(raw_entities, entities)

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
