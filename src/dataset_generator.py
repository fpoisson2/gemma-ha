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
        ],
        "turn_off": [
            "Éteins la lumière {location}",
            "Éteins {entity_name}",
            "Coupe la lumière {location}",
            "Désactive l'éclairage {location}",
        ],
        "set_brightness": [
            "Mets la lumière {location} à {brightness}%",
            "Règle la luminosité {location} à {brightness}%",
            "Tamise {location} à {brightness}%",
            "{entity_name} à {brightness} pourcent",
        ],
    },
    "switch": {
        "turn_on": [
            "Allume {entity_name}",
            "Active {entity_name}",
            "Mets {entity_name} en marche",
            "Démarre {entity_name}",
        ],
        "turn_off": [
            "Éteins {entity_name}",
            "Désactive {entity_name}",
            "Arrête {entity_name}",
            "Coupe {entity_name}",
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
        ],
        "set_hvac_mode": [
            "Mets le thermostat en mode {mode}",
            "Passe en mode {mode}",
            "Active le mode {mode}",
        ],
        "turn_on": [
            "Allume le chauffage",
            "Démarre la climatisation",
            "Active le thermostat",
        ],
        "turn_off": [
            "Éteins le chauffage",
            "Arrête la climatisation",
            "Coupe le chauffage",
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

# Noms de pièces courants
LOCATIONS_FR = [
    "du salon", "de la chambre", "de la cuisine", "de la salle de bain",
    "du bureau", "du couloir", "de l'entrée", "du garage",
    "du jardin", "de la terrasse", "de la buanderie",
]

# Modes HVAC
HVAC_MODES_FR = {
    "chauffage": "heat",
    "climatisation": "cool",
    "auto": "auto",
    "éco": "eco",
    "absent": "off",
}


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


class DatasetGenerator:
    """Génère un dataset de fine-tuning multi-turn pour FunctionGemma."""

    def __init__(
        self,
        entities: list[dict],
        examples_per_action: int = 20,
        seed: int = 42
    ):
        self.entities = entities
        self.examples_per_action = examples_per_action
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
        """Génère des exemples pour un domaine."""
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

                # Utiliser TOUS les templates pour plus de variété
                for template in action_templates:
                    location = random.choice(LOCATIONS_FR)
                    action_params = {}

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

                    examples.append(MultiTurnExample(
                        user_query=query,
                        domain=domain,
                        available_entities=available_entity_ids,
                        target_entity=entity_id,
                        action=actual_action,
                        action_params=action_params,
                    ))

        return examples

    def generate_all(self) -> list[MultiTurnExample]:
        """Génère tous les exemples d'entraînement."""
        print("Génération du dataset multi-turn...")

        all_examples = []

        domains = ["light", "switch", "climate", "cover", "lock", "scene", "fan"]

        for domain in tqdm(domains, desc="Domaines"):
            examples = self._generate_domain_examples(domain)
            print(f"  {domain}: {len(examples)} exemples")
            all_examples.extend(examples)

        # Mélanger
        random.shuffle(all_examples)

        self.examples = all_examples
        print(f"\nTotal: {len(all_examples)} exemples générés")

        return all_examples

    def save_dataset(self, output_dir: str, val_split: float = 0.1):
        """Sauvegarde le dataset au format JSON Lines."""
        os.makedirs(output_dir, exist_ok=True)

        # Split train/val
        n_val = int(len(self.examples) * val_split)
        val_examples = self.examples[:n_val]
        train_examples = self.examples[n_val:]

        # Sauvegarder
        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")

        with open(train_path, "w", encoding="utf-8") as f:
            for example in train_examples:
                data = example.to_training_format()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for example in val_examples:
                data = example.to_training_format()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"Dataset sauvegardé:")
        print(f"  Train: {train_path} ({len(train_examples)} exemples)")
        print(f"  Val: {val_path} ({len(val_examples)} exemples)")

        # Afficher un exemple
        if train_examples:
            print(f"\nExemple de format:")
            sample = train_examples[0].to_training_format()
            print(sample["text"])


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
