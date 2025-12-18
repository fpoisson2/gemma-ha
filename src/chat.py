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

# Forcer l'affichage immédiat
print("Démarrage...", flush=True)

import os
import re
import asyncio
from typing import Optional

print("Chargement de PyTorch...", flush=True)
import torch

print("Chargement de Transformers...", flush=True)
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

from ha_client import HomeAssistantClient


class GemmaHAChat:
    """Chat avec FunctionGemma pour contrôler Home Assistant."""

    def __init__(
        self,
        model_path: str,
        base_model: str = "google/functiongemma-270m-it",
        ha_client: Optional[HomeAssistantClient] = None,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.ha_client = ha_client
        self.model = None
        self.tokenizer = None
        self.entities_cache: dict[str, list[str]] = {}

    def load_model(self):
        """Charge le modèle fine-tuné."""
        print("=" * 50, flush=True)
        print("🤖 Chargement de FunctionGemma...", flush=True)
        print("=" * 50, flush=True)
        print(f"\n📦 Modèle de base: {self.base_model}", flush=True)

        # Charger le tokenizer depuis le modèle fine-tuné
        print("   Chargement du tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Détecter si CUDA est disponible
        if torch.cuda.is_available():
            print("   GPU CUDA détecté", flush=True)
            dtype = torch.bfloat16  # bfloat16 plus stable que float16 pour LoRA
            device_map = "auto"
        else:
            print("   Pas de GPU, utilisation du CPU (plus lent)", flush=True)
            dtype = torch.float32  # float16 cause des erreurs sur CPU
            device_map = None

        # Charger le modèle de base
        print("   Chargement du modèle (peut prendre 30-60s)...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Charger les adapters LoRA
        print("   Chargement des adapters LoRA...", flush=True)
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()

        device = next(self.model.parameters()).device
        print(f"\n✅ Modèle prêt sur: {device}", flush=True)

    async def init_ha(self):
        """Initialise la connexion Home Assistant et cache les entités."""
        if not self.ha_client:
            print("\n⚠️  Mode simulation (pas de connexion HA)", flush=True)
            return

        print(f"\n🏠 Connexion à Home Assistant...", flush=True)
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

        print(f"   ✅ {len(self.ha_client.entities)} entités chargées", flush=True)

        # Résumé des domaines supportés
        supported = ["light", "switch", "climate", "scene", "lock", "cover", "fan"]
        available = {d: len(e) for d, e in self.entities_cache.items() if d in supported and e}
        if available:
            print(f"   Domaines: {', '.join(f'{d}({n})' for d, n in available.items())}", flush=True)

    def parse_function_call(self, response: str) -> tuple[Optional[str], dict]:
        """Parse un appel de fonction FunctionGemma."""
        match = re.search(r'call:([a-z_\.]+)\{([^}]*)\}', response)
        if not match:
            return None, {}

        func_name = match.group(1)
        params_str = match.group(2)

        params = {}
        for param in params_str.split(','):
            if ':' in param:
                key, value = param.split(':', 1)
                value = value.replace('<escape>', '').strip()
                # Convertir les nombres
                if value.isdigit():
                    value = int(value)
                params[key.strip()] = value

        return func_name, params

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Génère une réponse du modèle."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding - plus stable
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extraire la dernière réponse du modèle
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1]
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]

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

        # Étape 1: Requête → get_entities
        output.append("📝 Étape 1: Analyse de la requête...")
        prompt1 = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        response1 = self.generate(prompt1, max_new_tokens=80)
        output.append(f"   Modèle: {response1}")

        func_name, params = self.parse_function_call(response1)

        if func_name != "get_entities":
            # Le modèle a peut-être directement appelé une action
            if func_name:
                output.append(f"\n⚡ Action directe détectée: {func_name}")
                result = await self.call_ha_service(func_name, params)
                output.append(f"   {result}")
                return "\n".join(output)
            output.append(f"\n❌ Réponse inattendue: {response1}")
            return "\n".join(output)

        domain = params.get("domain", "unknown")
        output.append(f"   → Domaine détecté: {domain}")

        # Étape 2: Récupérer les entités
        output.append(f"\n🔍 Étape 2: Récupération des entités {domain}...")
        tool_response = self.get_entities_for_domain(domain)
        entities_list = self.entities_cache.get(domain, [])[:5]
        output.append(f"   {len(self.entities_cache.get(domain, []))} entités trouvées")
        for e in entities_list:
            output.append(f"   - {e}")
        if len(self.entities_cache.get(domain, [])) > 5:
            output.append(f"   ...")

        # Étape 3: Générer l'action finale
        output.append(f"\n🎯 Étape 3: Génération de l'action...")
        prompt2 = (
            f"<start_of_turn>user\n{query}<end_of_turn>\n"
            f"<start_of_turn>model\n{response1}<end_of_turn>\n"
            f"<start_of_turn>tool\n{tool_response}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        response2 = self.generate(prompt2, max_new_tokens=100)
        output.append(f"   Modèle: {response2}")

        action, action_params = self.parse_function_call(response2)

        if not action:
            output.append(f"\n❌ Pas d'action détectée")
            return "\n".join(output)

        output.append(f"   → Action: {action}")
        output.append(f"   → Paramètres: {action_params}")

        # Étape 4: Exécuter sur Home Assistant
        output.append(f"\n🏠 Étape 4: Exécution sur Home Assistant...")
        result = await self.call_ha_service(action, action_params)
        output.append(f"   {result}")

        return "\n".join(output)

    async def chat_loop(self):
        """Boucle de chat interactive."""
        print("\n" + "="*50)
        print("💬 Chat Home Assistant")
        print("="*50)
        print("\nExemples de commandes:")
        print("  • Allume la lumière du salon")
        print("  • Mets le chauffage à 21 degrés")
        print("  • Active la scène cinéma")
        print("  • Éteins la TV")
        print("\nCommandes spéciales:")
        print("  • entities       → liste les domaines")
        print("  • entities light → liste les lumières")
        print("  • quit           → quitter")
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
    load_dotenv()

    # Charger la config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Client Home Assistant
    ha_client = HomeAssistantClient.from_env(config["home_assistant"]["url"])

    # Chemin du modèle fine-tuné
    model_path = os.path.join(os.path.dirname(__file__), "..", "functiongemma-ha")

    # Créer le chat
    chat = GemmaHAChat(
        model_path=model_path,
        base_model=config["model"]["name"],
        ha_client=ha_client,
    )

    # Charger le modèle
    chat.load_model()

    # Initialiser Home Assistant
    await chat.init_ha()

    # Lancer la boucle de chat
    await chat.chat_loop()


if __name__ == "__main__":
    asyncio.run(main())
