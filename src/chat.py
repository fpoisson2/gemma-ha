"""
Chat terminal pour contrôler Home Assistant avec FunctionGemma fine-tuné.

Flow multi-turn:
1. User query
2. Model → get_entities{domain}
3. Tool → liste des entités
4. Model → action{entity_id, params}
5. Exécution sur Home Assistant
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


class GemmaHAChat:
    """Chat avec FunctionGemma pour contrôler Home Assistant via llama.cpp."""

    def __init__(
        self,
        model_path: str,
        ha_client: Optional[HomeAssistantClient] = None,
        n_ctx: int = 2048,
        n_threads: int = -1,
        n_gpu_layers: int = 0,
    ):
        self.model_path = model_path
        self.gguf_path = os.path.join(
            model_path, "Functiongemma-Ha-Merged-268M-F16.gguf"
        )
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
        """Initialise la connexion Home Assistant et cache les entités."""
        if not self.ha_client:
            print("\nMode simulation (pas de connexion HA)", flush=True)
            return

        print(f"\nConnexion à Home Assistant...", flush=True)
        print(f"   URL: {self.ha_client.url}", flush=True)
        await self.ha_client.fetch_entities()

        # Cacher les entités par domaine
        for entity in self.ha_client.entities:
            entity_id = entity.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else ""
            if domain:
                if domain not in self.entities_cache:
                    self.entities_cache[domain] = []
                self.entities_cache[domain].append(entity_id)

        print(f"   {len(self.ha_client.entities)} entités chargées", flush=True)

        # Résumé des domaines supportés
        supported = ["light", "switch", "climate", "scene", "lock", "cover", "fan"]
        available = {
            d: len(e) for d, e in self.entities_cache.items() if d in supported and e
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

        return func_name, params

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

    def get_entities_for_domain(self, domain: str) -> str:
        """Retourne la liste des entités pour un domaine."""
        entities = self.entities_cache.get(domain, [])
        if not entities:
            return f"Aucune entité {domain} disponible"

        # Limiter à 10 pour ne pas surcharger
        entities_str = ", ".join(entities[:10])
        return f"Entités {domain} disponibles: {entities_str}"

    async def call_ha_service(self, func_name: str, params: dict) -> str:
        """Appelle un service Home Assistant."""
        if not self.ha_client:
            return f"[Simulation] {func_name}({params})"

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

    async def process_query(self, query: str) -> str:
        """
        Traite une requête utilisateur avec le flow multi-turn complet.

        1. User query → Model génère get_entities
        2. Tool retourne les entités
        3. Model génère l'action finale
        4. Exécution sur Home Assistant
        """
        output = []
        total_start = time.perf_counter()

        # Étape 1: Requête → get_entities
        t1 = time.perf_counter()
        prompt1 = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        response1 = self.generate(prompt1, max_new_tokens=40)
        t1_elapsed = time.perf_counter() - t1
        output.append(f"Étape 1: get_entities ({t1_elapsed:.2f}s)")
        output.append(f"   Modèle: {response1}")

        func_name, params = self.parse_function_call(response1)

        if func_name != "get_entities":
            # Le modèle a peut-être directement appelé une action
            if func_name:
                output.append(f"\nAction directe détectée: {func_name}")
                result = await self.call_ha_service(func_name, params)
                output.append(f"   {result}")
                return "\n".join(output)
            output.append(f"\nRéponse inattendue: {response1}")
            return "\n".join(output)

        domain = params.get("domain", "unknown")
        output.append(f"   -> Domaine: {domain}")

        # Étape 2: Récupérer les entités
        tool_response = self.get_entities_for_domain(domain)
        output.append(
            f"\nÉtape 2: {len(self.entities_cache.get(domain, []))} entités {domain}"
        )

        # Étape 3: Générer l'action finale
        t2 = time.perf_counter()
        prompt2 = (
            f"<start_of_turn>user\n{query}<end_of_turn>\n"
            f"<start_of_turn>model\n{response1}<end_of_turn>\n"
            f"<start_of_turn>tool\n{tool_response}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        response2 = self.generate(prompt2, max_new_tokens=50)
        t2_elapsed = time.perf_counter() - t2
        output.append(f"\nÉtape 3: action ({t2_elapsed:.2f}s)")
        output.append(f"   Modèle: {response2}")

        action, action_params = self.parse_function_call(response2)

        if not action:
            output.append(f"\nPas d'action détectée")
            return "\n".join(output)

        output.append(f"   -> Action: {action}")
        output.append(f"   -> Paramètres: {action_params}")

        # Étape 4: Exécuter sur Home Assistant
        t3 = time.perf_counter()
        result = await self.call_ha_service(action, action_params)
        t3_elapsed = time.perf_counter() - t3

        total_elapsed = time.perf_counter() - total_start
        output.append(f"\nÉtape 4: HA ({t3_elapsed:.2f}s)")
        output.append(f"   {result}")
        output.append(
            f"\nTotal: {total_elapsed:.2f}s (inference: {t1_elapsed + t2_elapsed:.2f}s)"
        )

        return "\n".join(output)

    async def chat_loop(self):
        """Boucle de chat interactive."""
        print("\n" + "=" * 50)
        print("Chat Home Assistant")
        print("=" * 50)
        print("\nExemples de commandes:")
        print("  - Allume la lumière du salon")
        print("  - Mets le chauffage à 21 degrés")
        print("  - Active la scène cinéma")
        print("  - Éteins la TV")
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
                for e in entities[:20]:
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

    # Chemin du modèle fine-tuné
    model_path = os.path.join(os.path.dirname(__file__), "..", "functiongemma-ha")

    # Créer le chat
    chat = GemmaHAChat(
        model_path=model_path,
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
