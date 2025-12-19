"""
Chat terminal pour contrôler Home Assistant avec FunctionGemma fine-tuné.

Format one-step:
1. User query + liste des entités disponibles
2. Model → action{entity_id, params}
3. Exécution sur Home Assistant
"""

import sys
import time

# Forcer l'affichage immédiat
print("Démarrage...", flush=True)

import os
import re
import asyncio
from typing import Optional

print("Chargement de llama.cpp...", flush=True)
from llama_cpp import Llama

print("Chargement de YAML...", flush=True)
import yaml
from dotenv import load_dotenv

from ha_client import HomeAssistantClient
from dataset_generator import filter_entities, USEFUL_DOMAINS


class GemmaHAChat:
    """Chat avec FunctionGemma pour contrôler Home Assistant via llama.cpp."""

    def __init__(
        self,
        gguf_path: str,
        ha_client: Optional[HomeAssistantClient] = None,
        n_ctx: int = 2048,
        n_threads: int = -1,
        n_gpu_layers: int = 0,
    ):
        self.gguf_path = gguf_path
        self.ha_client = ha_client
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self.entities_cache: dict[str, list[str]] = {}

    def load_model(self):
        """Charge le modèle GGUF avec llama.cpp."""
        print("=" * 50, flush=True)
        print("Chargement de FunctionGemma avec llama.cpp...", flush=True)
        print("=" * 50, flush=True)

        if not os.path.exists(self.gguf_path):
            print(f"Erreur: Modèle GGUF non trouvé: {self.gguf_path}")
            print("\nPour convertir le modèle HuggingFace en GGUF:")
            print("  cd llama.cpp")
            print("  python convert_hf_to_gguf.py ../functiongemma-ha-merged --outfile model.gguf")
            print("\nPuis relancer avec:")
            print("  python chat.py --model /chemin/vers/model.gguf")
            raise FileNotFoundError(f"Modèle GGUF non trouvé: {self.gguf_path}")

        print(f"Modèle: {self.gguf_path}", flush=True)
        print(f"Threads: {self.n_threads}", flush=True)
        print(f"GPU layers: {self.n_gpu_layers}", flush=True)
        print(f"Context: {self.n_ctx}", flush=True)

        # Charger le modèle avec llama.cpp
        print("Chargement du modèle...", flush=True)
        self.llm = Llama(
            model_path=self.gguf_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

        print("Modèle llama.cpp prêt", flush=True)

    async def init_ha(self):
        """Initialise la connexion Home Assistant et cache les entités filtrées."""
        if not self.ha_client:
            print("\nMode simulation (pas de connexion HA)", flush=True)
            return

        print(f"\nConnexion à Home Assistant...", flush=True)
        print(f"   URL: {self.ha_client.url}", flush=True)
        await self.ha_client.fetch_entities()

        # Filtrer les entités pour ne garder que les utiles
        raw_entities = self.ha_client.entities
        filtered_entities = filter_entities(raw_entities)

        print(f"   {len(raw_entities)} entités totales → {len(filtered_entities)} utiles", flush=True)

        # Cacher les entités filtrées par domaine
        for entity in filtered_entities:
            entity_id = entity.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else ""
            if domain:
                if domain not in self.entities_cache:
                    self.entities_cache[domain] = []
                self.entities_cache[domain].append(entity_id)

        # Résumé des domaines
        available = {
            d: len(e) for d, e in self.entities_cache.items() if d in USEFUL_DOMAINS and e
        }
        if available:
            print(
                f"   Domaines: {', '.join(f'{d}({n})' for d, n in available.items())}",
                flush=True,
            )

    def parse_function_call(self, response: str) -> tuple[Optional[str], dict]:
        """Parse un appel de fonction FunctionGemma."""
        match = re.search(r"call:([a-z_\.]+)\{([^}]*)\}", response)
        if not match:
            return None, {}

        func_name = match.group(1)
        params_str = match.group(2)

        params = {}
        for param in params_str.split(","):
            if ":" in param:
                key, value = param.split(":", 1)
                value = value.replace("<escape>", "").strip()
                # Convertir les nombres
                if value.isdigit():
                    value = int(value)
                params[key.strip()] = value

        # Corriger l'entity_id si elle n'existe pas exactement
        if "entity_id" in params:
            best_match = self._find_best_entity(params["entity_id"])
            if best_match and best_match != params["entity_id"]:
                params["entity_id"] = best_match

        # Corriger le domaine du service basé sur l'entity_id
        func_name = self._fix_service_domain(func_name, params)

        return func_name, params

    def _fix_service_domain(self, func_name: str, params: dict) -> str:
        """Corrige le domaine du service pour correspondre à l'entity_id."""
        entity_id = params.get("entity_id", "")
        if not entity_id or "." not in func_name:
            return func_name

        entity_domain = entity_id.split(".")[0]
        service_domain, service = func_name.split(".", 1)

        # Si le domaine du service ne correspond pas à l'entité, corriger
        if service_domain != entity_domain:
            # Mapping des services compatibles
            if service in ("turn_on", "turn_off", "toggle"):
                return f"{entity_domain}.{service}"

        return func_name

    def _find_best_entity(self, entity_id: str) -> Optional[str]:
        """Trouve l'entité la plus proche si elle n'existe pas exactement."""
        # Collecter toutes les entités
        all_entities = []
        for entities in self.entities_cache.values():
            all_entities.extend(entities)

        # Entité exacte existe
        if entity_id in all_entities:
            return entity_id

        # Chercher par correspondance partielle
        entity_name = entity_id.split(".")[-1] if "." in entity_id else entity_id
        domain = entity_id.split(".")[0] if "." in entity_id else None

        # Chercher dans le même domaine d'abord
        if domain and domain in self.entities_cache:
            for e in self.entities_cache[domain]:
                if entity_name in e or e.split(".")[-1] in entity_name:
                    return e

        # Chercher dans tous les domaines
        for e in all_entities:
            if entity_name in e or e.split(".")[-1] in entity_name:
                return e

        return None

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Génère une réponse avec llama.cpp."""
        if self.llm is None:
            raise RuntimeError("Modèle non chargé. Appelez load_model() d'abord.")

        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            stop=["<end_of_turn>", "<eos>"],
            echo=False,
        )

        response = output["choices"][0]["text"]
        return response.strip()

    async def get_all_states(self) -> str:
        """
        Récupère tous les états Home Assistant via GET /api/states.
        C'est le handler pour le tool MCP ha.get_states (sans paramètres).
        """
        if not self.ha_client:
            return "[Simulation] États non disponibles"

        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ha_client.url}/api/states"
                async with session.get(
                    url,
                    headers=self.ha_client._headers(),
                ) as resp:
                    if resp.status == 200:
                        all_states = await resp.json()

                        # Formater les états par domaine (limité aux domaines supportés)
                        supported_domains = ["person", "light", "switch", "climate", "cover", "lock", "fan"]
                        results = []

                        for entity_data in all_states:
                            entity_id = entity_data.get("entity_id", "")
                            domain = entity_id.split(".")[0] if "." in entity_id else ""

                            if domain not in supported_domains:
                                continue

                            state = entity_data.get("state", "unknown")
                            attrs = entity_data.get("attributes", {})
                            friendly_name = attrs.get("friendly_name", entity_id.split(".")[-1])

                            if domain == "person":
                                results.append(f"{friendly_name}: {state}")
                            elif domain == "light":
                                if state == "on":
                                    brightness = attrs.get("brightness", 255)
                                    pct = int(brightness / 255 * 100)
                                    results.append(f"{friendly_name}: on ({pct}%)")
                                else:
                                    results.append(f"{friendly_name}: off")
                            elif domain == "climate":
                                current_temp = attrs.get("current_temperature", "?")
                                results.append(f"{friendly_name}: {current_temp}°C")
                            elif domain == "cover":
                                position = attrs.get("current_position", "?")
                                results.append(f"{friendly_name}: {state} ({position}%)")
                            elif domain == "lock":
                                results.append(f"{friendly_name}: {state}")
                            elif domain == "switch":
                                results.append(f"{friendly_name}: {state}")
                            elif domain == "fan":
                                results.append(f"{friendly_name}: {state}")

                        return "\n".join(results) if results else "Aucun état disponible"
                    else:
                        text = await resp.text()
                        return f"Erreur {resp.status}: {text}"
        except Exception as e:
            return f"Erreur: {e}"

    async def call_ha_service(self, func_name: str, params: dict) -> str:
        """Appelle un service Home Assistant."""
        if not self.ha_client:
            return f"[Simulation] {func_name}({params})"

        # Handler spécial pour ha.get_states (tool MCP sans paramètres)
        if func_name == "ha.get_states":
            return await self.get_all_states()

        # Parser domain.service
        if "." not in func_name:
            return f"Erreur: format invalide {func_name}"

        domain, service = func_name.split(".", 1)
        entity_id = params.pop("entity_id", None)

        # Construire les données du service
        service_data = {}
        if entity_id:
            service_data["entity_id"] = entity_id
        service_data.update(params)

        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ha_client.url}/api/services/{domain}/{service}"
                async with session.post(
                    url,
                    headers=self.ha_client._headers(),
                    json=service_data,
                ) as resp:
                    if resp.status == 200:
                        return f"OK: {func_name} exécuté"
                    else:
                        text = await resp.text()
                        return f"Erreur {resp.status}: {text}"
        except Exception as e:
            return f"Erreur: {e}"

    def _build_entities_context(self) -> str:
        """Construit le contexte des entités disponibles (filtrées)."""
        parts = []
        for domain in USEFUL_DOMAINS:
            entities = self.entities_cache.get(domain, [])
            if entities:
                # Toutes les entités (déjà filtrées, donc peu nombreuses)
                entities_str = ", ".join(entities)
                parts.append(f"Entités {domain} disponibles: {entities_str}")
        return "\n".join(parts) if parts else "Aucune entité disponible"

    async def process_query(self, query: str) -> str:
        """
        Traite une requête utilisateur (format one-step).
        Les entités sont injectées directement dans le prompt.
        """
        output = []
        total_start = time.perf_counter()

        # Construire le contexte avec toutes les entités filtrées
        entities_context = self._build_entities_context()

        # Une seule passe d'inférence (format identique à l'entraînement)
        t1 = time.perf_counter()
        prompt = (
            f"<start_of_turn>user\n{query}\n\n"
            f"{entities_context}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        response = self.generate(prompt, max_new_tokens=80)
        t1_elapsed = time.perf_counter() - t1
        output.append(f"Inference ({t1_elapsed:.2f}s)")
        output.append(f"   Modèle: {response}")

        action, action_params = self.parse_function_call(response)

        if not action:
            output.append(f"\nPas d'action détectée")
            return "\n".join(output)

        output.append(f"   -> Action: {action}")
        output.append(f"   -> Paramètres: {action_params}")

        # Exécuter sur Home Assistant
        t2 = time.perf_counter()
        result = await self.call_ha_service(action, action_params)
        t2_elapsed = time.perf_counter() - t2

        total_elapsed = time.perf_counter() - total_start
        output.append(f"\nHA ({t2_elapsed:.2f}s)")
        output.append(f"   {result}")
        output.append(f"\nTotal: {total_elapsed:.2f}s")

        return "\n".join(output)

    async def chat_loop(self):
        """Boucle de chat interactive."""
        print("\n" + "=" * 50)
        print("Chat Home Assistant (format one-step)")
        print("=" * 50)
        print("\nExemples de commandes:")
        print("  - Allume la lumière du salon")
        print("  - Mets le chauffage à 21 degrés")
        print("  - Active la scène cinéma")
        print("  - Ferme les volets")
        print("\nCommandes spéciales:")
        print("  - entities       -> liste les domaines")
        print("  - entities light -> liste les lumières")
        print("  - quit           -> quitter")
        print()

        while True:
            try:
                query = input("\033[94mVous:\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAu revoir!")
                break

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Au revoir!")
                break

            if query.lower() == "entities":
                for domain, entities in self.entities_cache.items():
                    print(f"  {domain}: {len(entities)} entités")
                continue

            if query.lower().startswith("entities "):
                domain = query.split()[1]
                entities = self.entities_cache.get(domain, [])
                for e in entities:
                    print(f"  {e}")
                continue

            # Traiter la requête
            print("\033[90mTraitement...\033[0m")
            try:
                result = await self.process_query(query)
                print(f"\033[92mAssistant:\033[0m\n{result}\n")
            except Exception as e:
                print(f"\033[91mErreur: {e}\033[0m\n")


async def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Chat Home Assistant avec FunctionGemma (llama.cpp)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Chemin vers le fichier GGUF du modèle",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Taille du contexte (défaut: 2048)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=-1,
        help="Nombre de threads CPU (défaut: -1 = auto)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Nombre de layers GPU (défaut: 0 = CPU only)",
    )
    parser.add_argument(
        "--no-ha",
        action="store_true",
        help="Mode simulation sans Home Assistant",
    )
    args = parser.parse_args()

    load_dotenv()

    # Charger la config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Client Home Assistant
    ha_client = None
    if not args.no_ha:
        ha_client = HomeAssistantClient.from_env(config["home_assistant"]["url"])

    # Créer le chat
    chat = GemmaHAChat(
        gguf_path=args.model,
        ha_client=ha_client,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )

    # Charger le modèle
    chat.load_model()

    # Initialiser Home Assistant
    await chat.init_ha()

    # Lancer la boucle de chat
    await chat.chat_loop()


if __name__ == "__main__":
    asyncio.run(main())
