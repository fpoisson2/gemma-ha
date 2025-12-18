"""
Générateur de dataset pour fine-tuner FunctionGemma sur Home Assistant.
Génère des paires (requête utilisateur -> appel de fonction) en français.
"""

import os
import json
import random
from typing import Optional
from dataclasses import dataclass, field

import yaml
from tqdm import tqdm


# Templates de requêtes en français par type de fonction
TEMPLATES_FR = {
    # Éclairage
    "light.turn_on": [
        "Allume {entity_name}",
        "Allume la lumière {entity_name}",
        "Allume les lumières {location}",
        "Mets la lumière {location}",
        "Éclaire {location}",
        "Peux-tu allumer {entity_name} ?",
        "J'aimerais allumer {entity_name}",
        "Active {entity_name}",
        "Lumière {location} allumée s'il te plaît",
    ],
    "light.turn_off": [
        "Éteins {entity_name}",
        "Éteins la lumière {entity_name}",
        "Éteins les lumières {location}",
        "Coupe la lumière {location}",
        "Désactive {entity_name}",
        "Peux-tu éteindre {entity_name} ?",
        "J'aimerais éteindre {entity_name}",
        "Lumière {location} éteinte",
    ],
    "light.toggle": [
        "Bascule {entity_name}",
        "Change l'état de {entity_name}",
        "Toggle {entity_name}",
        "Inverse {entity_name}",
    ],
    "light.set_brightness": [
        "Mets {entity_name} à {brightness}%",
        "Règle la luminosité de {entity_name} à {brightness}%",
        "Baisse {entity_name} à {brightness}%",
        "Augmente {entity_name} à {brightness}%",
        "{entity_name} à {brightness} pourcent",
        "Luminosité {location} à {brightness}%",
        "Tamise {entity_name} à {brightness}%",
    ],
    "light.set_color": [
        "Mets {entity_name} en {color}",
        "Change la couleur de {entity_name} en {color}",
        "Couleur {color} pour {entity_name}",
        "{entity_name} en {color}",
        "Passe {entity_name} en {color}",
    ],

    # Interrupteurs
    "switch.turn_on": [
        "Allume {entity_name}",
        "Active {entity_name}",
        "Mets {entity_name} en marche",
        "Démarre {entity_name}",
    ],
    "switch.turn_off": [
        "Éteins {entity_name}",
        "Désactive {entity_name}",
        "Arrête {entity_name}",
        "Coupe {entity_name}",
    ],

    # Climat / Thermostat
    "climate.set_temperature": [
        "Mets le chauffage à {temperature} degrés",
        "Règle la température à {temperature}°C",
        "Je veux {temperature} degrés {location}",
        "Température à {temperature} degrés",
        "Chauffe à {temperature}°C",
        "Monte le chauffage à {temperature}",
        "Baisse la température à {temperature}",
    ],
    "climate.set_hvac_mode": [
        "Mets le thermostat en mode {mode}",
        "Passe en mode {mode}",
        "Active le mode {mode}",
        "Mode {mode} pour le climat",
    ],
    "climate.turn_on": [
        "Allume le chauffage",
        "Démarre la climatisation",
        "Active le thermostat {location}",
    ],
    "climate.turn_off": [
        "Éteins le chauffage",
        "Arrête la climatisation",
        "Désactive le thermostat",
        "Coupe le chauffage",
    ],

    # Volets / Couvertures
    "cover.open_cover": [
        "Ouvre {entity_name}",
        "Ouvre les volets {location}",
        "Lève {entity_name}",
        "Monte les stores {location}",
    ],
    "cover.close_cover": [
        "Ferme {entity_name}",
        "Ferme les volets {location}",
        "Baisse {entity_name}",
        "Descends les stores {location}",
    ],
    "cover.set_cover_position": [
        "Mets {entity_name} à {position}%",
        "Ouvre {entity_name} à {position}%",
        "Position {position}% pour les volets {location}",
    ],

    # Serrures
    "lock.lock": [
        "Verrouille {entity_name}",
        "Ferme à clé {location}",
        "Verrouille la porte {location}",
        "Bloque {entity_name}",
    ],
    "lock.unlock": [
        "Déverrouille {entity_name}",
        "Ouvre {entity_name}",
        "Débloque la porte {location}",
    ],

    # Alarme
    "alarm_control_panel.arm_away": [
        "Active l'alarme en mode absence",
        "Arme l'alarme, je pars",
        "Mets l'alarme en mode absent",
        "Active la sécurité complète",
    ],
    "alarm_control_panel.arm_home": [
        "Active l'alarme en mode maison",
        "Arme l'alarme, je reste",
        "Mode nuit pour l'alarme",
        "Sécurité mode présent",
    ],
    "alarm_control_panel.disarm": [
        "Désactive l'alarme",
        "Désarme l'alarme",
        "Coupe l'alarme",
        "Éteins l'alarme",
    ],

    # Média
    "media_player.play_media": [
        "Joue de la musique {location}",
        "Lance la musique sur {entity_name}",
        "Mets de la musique",
    ],
    "media_player.pause": [
        "Pause {entity_name}",
        "Mets en pause la musique",
        "Arrête la lecture",
    ],
    "media_player.volume_set": [
        "Volume à {volume}% {location}",
        "Mets le son à {volume}%",
        "Monte le volume à {volume}",
        "Baisse le son à {volume}%",
    ],

    # Ventilateur
    "fan.turn_on": [
        "Allume le ventilateur {location}",
        "Démarre {entity_name}",
        "Active la ventilation",
    ],
    "fan.turn_off": [
        "Éteins le ventilateur {location}",
        "Arrête {entity_name}",
        "Coupe la ventilation",
    ],
    "fan.set_percentage": [
        "Mets le ventilateur à {percentage}%",
        "Vitesse {percentage}% pour {entity_name}",
    ],

    # Scènes
    "scene.turn_on": [
        "Active la scène {scene_name}",
        "Lance le mode {scene_name}",
        "Mets l'ambiance {scene_name}",
        "Scène {scene_name}",
    ],

    # Scripts
    "script.turn_on": [
        "Lance le script {script_name}",
        "Exécute {script_name}",
        "Démarre {script_name}",
    ],

    # Automations
    "automation.trigger": [
        "Déclenche l'automatisation {automation_name}",
        "Lance l'automatisation {automation_name}",
        "Exécute {automation_name}",
    ],
    "automation.turn_on": [
        "Active l'automatisation {automation_name}",
        "Réactive {automation_name}",
    ],
    "automation.turn_off": [
        "Désactive l'automatisation {automation_name}",
        "Suspends {automation_name}",
    ],
}

# Noms de pièces courants
LOCATIONS_FR = [
    "du salon", "de la chambre", "de la cuisine", "de la salle de bain",
    "du bureau", "du couloir", "de l'entrée", "du garage", "de la cave",
    "du jardin", "de la terrasse", "du grenier", "de la buanderie",
    "de la chambre d'enfant", "de la chambre parentale", "du dressing",
    "au salon", "dans la chambre", "dans la cuisine", "dans le bureau",
]

# Couleurs
COLORS_FR = {
    "rouge": [255, 0, 0],
    "vert": [0, 255, 0],
    "bleu": [0, 0, 255],
    "jaune": [255, 255, 0],
    "orange": [255, 165, 0],
    "violet": [128, 0, 128],
    "rose": [255, 192, 203],
    "blanc": [255, 255, 255],
    "blanc chaud": None,  # Pour color_temp
    "blanc froid": None,
}

# Modes HVAC
HVAC_MODES = ["chauffage", "climatisation", "auto", "éco", "confort", "absent"]
HVAC_MODE_MAP = {
    "chauffage": "heat",
    "climatisation": "cool",
    "auto": "auto",
    "éco": "eco",
    "confort": "heat",
    "absent": "off",
}


@dataclass
class TrainingExample:
    """Un exemple d'entraînement."""
    user_query: str
    function_call: dict
    entities_context: list[str] = field(default_factory=list)

    def to_functiongemma_format(self, function_schemas: list[dict]) -> dict:
        """
        Convertit en format FunctionGemma.
        Format: messages avec developer (system), user, et assistant (function call)
        """
        # Format de sortie FunctionGemma
        # <start_function_call>call:function_name{param:<escape>value<escape>}<end_function_call>
        params_str = ""
        for key, value in self.function_call.get("parameters", {}).items():
            if isinstance(value, str):
                params_str += f"{key}:<escape>{value}<escape>,"
            elif isinstance(value, (int, float)):
                params_str += f"{key}:{value},"
            elif isinstance(value, list):
                params_str += f"{key}:{json.dumps(value)},"
            elif isinstance(value, bool):
                params_str += f"{key}:{str(value).lower()},"

        if params_str.endswith(","):
            params_str = params_str[:-1]

        func_name = self.function_call["name"]
        assistant_response = f"<start_function_call>call:{func_name}{{{params_str}}}<end_function_call>"

        return {
            "messages": [
                {
                    "role": "developer",
                    "content": "Tu es un assistant qui contrôle une maison intelligente avec Home Assistant. Tu dois appeler les fonctions appropriées pour répondre aux demandes de l'utilisateur."
                },
                {
                    "role": "user",
                    "content": self.user_query
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ],
            "tools": function_schemas
        }


class DatasetGenerator:
    """Génère un dataset de fine-tuning pour FunctionGemma."""

    def __init__(
        self,
        function_schemas: list[dict],
        entities: list[dict],
        examples_per_function: int = 50,
        seed: int = 42
    ):
        self.function_schemas = function_schemas
        self.entities = entities
        self.examples_per_function = examples_per_function
        self.examples: list[TrainingExample] = []

        random.seed(seed)

        # Indexer les entités par domaine
        self.entities_by_domain = {}
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else ""
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

    def _generate_light_examples(self) -> list[TrainingExample]:
        """Génère des exemples pour les lumières."""
        examples = []
        lights = self.entities_by_domain.get("light", [])

        if not lights:
            return examples

        for light in lights:
            entity_id = light["entity_id"]
            entity_name = self._get_entity_name(light)
            location = random.choice(LOCATIONS_FR)

            # Turn on
            for template in random.sample(TEMPLATES_FR.get("light.turn_on", []), min(3, len(TEMPLATES_FR.get("light.turn_on", [])))):
                query = template.format(entity_name=entity_name, location=location)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "light.turn_on",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

            # Turn off
            for template in random.sample(TEMPLATES_FR.get("light.turn_off", []), min(3, len(TEMPLATES_FR.get("light.turn_off", [])))):
                query = template.format(entity_name=entity_name, location=location)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "light.turn_off",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

            # Brightness
            brightness = random.choice([10, 25, 50, 75, 100])
            for template in random.sample(TEMPLATES_FR.get("light.set_brightness", []), min(2, len(TEMPLATES_FR.get("light.set_brightness", [])))):
                query = template.format(entity_name=entity_name, location=location, brightness=brightness)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "light.turn_on",
                        "parameters": {
                            "entity_id": entity_id,
                            "brightness_pct": brightness
                        }
                    }
                ))

            # Colors (si supporté)
            attrs = light.get("attributes", {})
            if attrs.get("supported_color_modes") and "rgb" in str(attrs.get("supported_color_modes")):
                for color_name, rgb in COLORS_FR.items():
                    if rgb is None:
                        continue
                    for template in random.sample(TEMPLATES_FR.get("light.set_color", []), min(2, len(TEMPLATES_FR.get("light.set_color", [])))):
                        query = template.format(entity_name=entity_name, color=color_name)
                        examples.append(TrainingExample(
                            user_query=query,
                            function_call={
                                "name": "light.turn_on",
                                "parameters": {
                                    "entity_id": entity_id,
                                    "rgb_color": rgb
                                }
                            }
                        ))

        return examples

    def _generate_climate_examples(self) -> list[TrainingExample]:
        """Génère des exemples pour le climat/thermostat."""
        examples = []
        climates = self.entities_by_domain.get("climate", [])

        if not climates:
            return examples

        for climate in climates:
            entity_id = climate["entity_id"]
            entity_name = self._get_entity_name(climate)
            location = random.choice(LOCATIONS_FR)

            # Set temperature
            for temp in [18, 19, 20, 21, 22, 23, 24]:
                for template in random.sample(TEMPLATES_FR.get("climate.set_temperature", []), min(2, len(TEMPLATES_FR.get("climate.set_temperature", [])))):
                    query = template.format(temperature=temp, location=location)
                    examples.append(TrainingExample(
                        user_query=query,
                        function_call={
                            "name": "climate.set_temperature",
                            "parameters": {
                                "entity_id": entity_id,
                                "temperature": temp
                            }
                        }
                    ))

            # HVAC modes
            for mode_fr, mode_en in HVAC_MODE_MAP.items():
                for template in random.sample(TEMPLATES_FR.get("climate.set_hvac_mode", []), min(2, len(TEMPLATES_FR.get("climate.set_hvac_mode", [])))):
                    query = template.format(mode=mode_fr)
                    examples.append(TrainingExample(
                        user_query=query,
                        function_call={
                            "name": "climate.set_hvac_mode",
                            "parameters": {
                                "entity_id": entity_id,
                                "hvac_mode": mode_en
                            }
                        }
                    ))

            # Turn on/off
            for template in TEMPLATES_FR.get("climate.turn_on", []):
                examples.append(TrainingExample(
                    user_query=template.format(location=location),
                    function_call={
                        "name": "climate.turn_on",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

            for template in TEMPLATES_FR.get("climate.turn_off", []):
                examples.append(TrainingExample(
                    user_query=template,
                    function_call={
                        "name": "climate.turn_off",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

        return examples

    def _generate_cover_examples(self) -> list[TrainingExample]:
        """Génère des exemples pour les volets/stores."""
        examples = []
        covers = self.entities_by_domain.get("cover", [])

        if not covers:
            return examples

        for cover in covers:
            entity_id = cover["entity_id"]
            entity_name = self._get_entity_name(cover)
            location = random.choice(LOCATIONS_FR)

            # Open
            for template in TEMPLATES_FR.get("cover.open_cover", []):
                query = template.format(entity_name=entity_name, location=location)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "cover.open_cover",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

            # Close
            for template in TEMPLATES_FR.get("cover.close_cover", []):
                query = template.format(entity_name=entity_name, location=location)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "cover.close_cover",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

            # Position
            for position in [25, 50, 75]:
                for template in random.sample(TEMPLATES_FR.get("cover.set_cover_position", []), min(2, len(TEMPLATES_FR.get("cover.set_cover_position", [])))):
                    query = template.format(entity_name=entity_name, location=location, position=position)
                    examples.append(TrainingExample(
                        user_query=query,
                        function_call={
                            "name": "cover.set_cover_position",
                            "parameters": {
                                "entity_id": entity_id,
                                "position": position
                            }
                        }
                    ))

        return examples

    def _generate_lock_examples(self) -> list[TrainingExample]:
        """Génère des exemples pour les serrures."""
        examples = []
        locks = self.entities_by_domain.get("lock", [])

        if not locks:
            return examples

        for lock in locks:
            entity_id = lock["entity_id"]
            entity_name = self._get_entity_name(lock)
            location = random.choice(LOCATIONS_FR)

            for template in TEMPLATES_FR.get("lock.lock", []):
                query = template.format(entity_name=entity_name, location=location)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "lock.lock",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

            for template in TEMPLATES_FR.get("lock.unlock", []):
                query = template.format(entity_name=entity_name, location=location)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "lock.unlock",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

        return examples

    def _generate_alarm_examples(self) -> list[TrainingExample]:
        """Génère des exemples pour l'alarme."""
        examples = []
        alarms = self.entities_by_domain.get("alarm_control_panel", [])

        if not alarms:
            return examples

        for alarm in alarms:
            entity_id = alarm["entity_id"]

            for template in TEMPLATES_FR.get("alarm_control_panel.arm_away", []):
                examples.append(TrainingExample(
                    user_query=template,
                    function_call={
                        "name": "alarm_control_panel.arm_away",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

            for template in TEMPLATES_FR.get("alarm_control_panel.arm_home", []):
                examples.append(TrainingExample(
                    user_query=template,
                    function_call={
                        "name": "alarm_control_panel.arm_home",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

            for template in TEMPLATES_FR.get("alarm_control_panel.disarm", []):
                examples.append(TrainingExample(
                    user_query=template,
                    function_call={
                        "name": "alarm_control_panel.disarm",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

        return examples

    def _generate_switch_examples(self) -> list[TrainingExample]:
        """Génère des exemples pour les interrupteurs."""
        examples = []
        switches = self.entities_by_domain.get("switch", [])

        if not switches:
            return examples

        for switch in switches:
            entity_id = switch["entity_id"]
            entity_name = self._get_entity_name(switch)

            for template in TEMPLATES_FR.get("switch.turn_on", []):
                query = template.format(entity_name=entity_name)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "switch.turn_on",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

            for template in TEMPLATES_FR.get("switch.turn_off", []):
                query = template.format(entity_name=entity_name)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "switch.turn_off",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

        return examples

    def _generate_scene_examples(self) -> list[TrainingExample]:
        """Génère des exemples pour les scènes."""
        examples = []
        scenes = self.entities_by_domain.get("scene", [])

        if not scenes:
            return examples

        for scene in scenes:
            entity_id = scene["entity_id"]
            scene_name = self._get_entity_name(scene)

            for template in TEMPLATES_FR.get("scene.turn_on", []):
                query = template.format(scene_name=scene_name)
                examples.append(TrainingExample(
                    user_query=query,
                    function_call={
                        "name": "scene.turn_on",
                        "parameters": {"entity_id": entity_id}
                    }
                ))

        return examples

    def generate_all(self) -> list[TrainingExample]:
        """Génère tous les exemples d'entraînement."""
        print("Génération du dataset...")

        all_examples = []

        # Générer par catégorie
        generators = [
            ("Lumières", self._generate_light_examples),
            ("Climat", self._generate_climate_examples),
            ("Volets", self._generate_cover_examples),
            ("Serrures", self._generate_lock_examples),
            ("Alarme", self._generate_alarm_examples),
            ("Interrupteurs", self._generate_switch_examples),
            ("Scènes", self._generate_scene_examples),
        ]

        for name, generator in tqdm(generators, desc="Catégories"):
            examples = generator()
            print(f"  {name}: {len(examples)} exemples")
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

        # Filtrer les schémas pertinents pour chaque exemple
        relevant_schemas = [s for s in self.function_schemas if any(
            s.get("function", {}).get("name", "").split(".")[0] in domain
            for domain in ["light", "switch", "climate", "cover", "lock",
                          "alarm_control_panel", "media_player", "fan",
                          "scene", "script", "automation"]
        )]

        with open(train_path, "w", encoding="utf-8") as f:
            for example in train_examples:
                data = example.to_functiongemma_format(relevant_schemas)
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for example in val_examples:
                data = example.to_functiongemma_format(relevant_schemas)
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"Dataset sauvegardé:")
        print(f"  Train: {train_path} ({len(train_examples)} exemples)")
        print(f"  Val: {val_path} ({len(val_examples)} exemples)")


async def main():
    """Génère le dataset depuis Home Assistant."""
    import asyncio
    from ha_client import HomeAssistantClient

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Connexion à Home Assistant
    client = HomeAssistantClient.from_env(config["home_assistant"]["url"])

    print("Récupération des données de Home Assistant...")
    functions = await client.build_function_schemas()
    entities = client.entities

    print(f"  Fonctions: {len(functions)}")
    print(f"  Entités: {len(entities)}")

    # Convertir les fonctions en schémas
    function_schemas = [f.to_schema() for f in functions]

    # Générer le dataset
    generator = DatasetGenerator(
        function_schemas=function_schemas,
        entities=entities,
        examples_per_function=config["dataset"]["examples_per_function"],
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
