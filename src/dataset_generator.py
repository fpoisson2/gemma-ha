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
        ],
        "set_hvac_mode": [
            "Mets le thermostat en mode {mode}",
            "Passe en mode {mode}",
            "Active le mode {mode}",
            "Mode {mode}",
            "Met en {mode}",
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
        ],
        "turn_off": [
            "Éteins le chauffage",
            "Arrête la climatisation",
            "Coupe le chauffage",
            "Ferme le chauffage",
            "Éteins la clim",
            "Ferme la clim",
            "Arrête le thermostat",
            "Coupe la clim",
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
        ],
    },
    "cover": {
        "open_cover": [
            "Ouvre les volets {location}",
            "Ouvre {entity_name}",
            "Lève les stores {location}",
            "Monte les volets {location}",
        ],
        "close_cover": [
            "Ferme les volets {location}",
            "Ferme {entity_name}",
            "Baisse les stores {location}",
            "Descends les volets {location}",
        ],
        "set_cover_position": [
            "Mets les volets {location} à {position}%",
            "Ouvre {entity_name} à {position}%",
        ],
    },
    "lock": {
        "lock": [
            "Verrouille {entity_name}",
            "Ferme à clé {location}",
            "Verrouille la porte {location}",
        ],
        "unlock": [
            "Déverrouille {entity_name}",
            "Ouvre {entity_name}",
            "Débloque la porte {location}",
        ],
    },
    "scene": {
        "turn_on": [
            "Active la scène {entity_name}",
            "Lance le mode {entity_name}",
            "Mets l'ambiance {entity_name}",
            "Scène {entity_name}",
        ],
    },
    "fan": {
        "turn_on": [
            "Allume le ventilateur {location}",
            "Démarre {entity_name}",
            "Active la ventilation {location}",
        ],
        "turn_off": [
            "Éteins le ventilateur {location}",
            "Arrête {entity_name}",
            "Coupe la ventilation {location}",
        ],
    },
}

# Mapping des noms d'entités vers les locutions françaises
# Clé: partie du nom d'entité (en minuscule), Valeur: forme française
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
        action_call = format_function_call(
            f"{self.domain}.{self.action}",
            action_params
        )

        # Format texte pour l'entraînement
        text = (
            f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n{action_call}<end_of_turn>"
        )

        return {"text": text}


class DatasetGenerator:
    """Génère un dataset de fine-tuning multi-turn pour FunctionGemma."""

    def __init__(
        self,
        entities: list[dict],
        examples_per_action: int = 20,
        examples_per_domain: int = 100,  # Limite par domaine pour équilibrer
        seed: int = 42
    ):
        self.entities = entities
        self.examples_per_action = examples_per_action
        self.examples_per_domain = examples_per_domain
        self.examples: list[MultiTurnExample] = []

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

        # Mélanger et limiter pour équilibrer les domaines
        random.shuffle(examples)
        if len(examples) > self.examples_per_domain:
            examples = examples[:self.examples_per_domain]

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
        print(f"\nTotal: {len(all_examples)} exemples générés")

        return all_examples

    def save_dataset(self, output_dir: str, val_split: float = 0.1, include_one_step: bool = True):
        """Sauvegarde le dataset au format JSON Lines."""
        os.makedirs(output_dir, exist_ok=True)

        # Split train/val
        n_val = int(len(self.examples) * val_split)
        val_examples = self.examples[:n_val]
        train_examples = self.examples[n_val:]

        # Sauvegarder
        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")

        train_count = 0
        val_count = 0

        with open(train_path, "w", encoding="utf-8") as f:
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

        with open(val_path, "w", encoding="utf-8") as f:
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

        print(f"Dataset sauvegardé:")
        print(f"  Train: {train_path} ({train_count} exemples)")
        print(f"  Val: {val_path} ({val_count} exemples)")
        if include_one_step:
            print(f"  (inclut multi-turn ET one-step pour chaque exemple)")

        # Afficher des exemples
        if train_examples:
            print(f"\nExemple multi-turn:")
            sample = train_examples[0].to_training_format()
            print(sample["text"])
            print(f"\nExemple one-step:")
            sample_one = train_examples[0].to_one_step_format()
            print(sample_one["text"])


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
