"""
Client pour récupérer les tools Home Assistant via l'API MCP.
"""

import os
import json
import asyncio
from typing import Optional
from dataclasses import dataclass, field

import aiohttp
from dotenv import load_dotenv


@dataclass
class HAFunction:
    """Représente une fonction Home Assistant."""
    name: str
    description: str
    parameters: dict

    def to_schema(self) -> dict:
        """Convertit en schéma JSON pour FunctionGemma."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


@dataclass
class HomeAssistantClient:
    """Client pour interagir avec Home Assistant via MCP."""

    url: str
    token: str
    functions: list[HAFunction] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)

    @classmethod
    def from_env(cls, url: str) -> "HomeAssistantClient":
        """Crée un client depuis les variables d'environnement."""
        load_dotenv()
        token = os.getenv("HA_TOKEN")
        if not token:
            raise ValueError("HA_TOKEN non défini dans .env")
        return cls(url=url, token=token)

    def _headers(self) -> dict:
        """Headers pour l'authentification."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    async def fetch_entities(self) -> list[dict]:
        """Récupère toutes les entités exposées."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.url}/api/states",
                headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Erreur API: {resp.status}")
                self.entities = await resp.json()
                return self.entities

    async def fetch_services(self) -> dict:
        """Récupère tous les services disponibles."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.url}/api/services",
                headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Erreur API: {resp.status}")
                return await resp.json()

    async def fetch_mcp_tools(self) -> list[dict]:
        """
        Récupère les tools via l'API MCP.
        L'API MCP utilise JSON-RPC sur HTTP.
        """
        async with aiohttp.ClientSession() as session:
            # Initialisation MCP
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "gemma-ha-trainer",
                        "version": "1.0.0"
                    }
                }
            }

            async with session.post(
                f"{self.url}/api/mcp",
                headers=self._headers(),
                json=init_request
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Erreur MCP init: {resp.status} - {text}")
                init_response = await resp.json()

            # Récupérer la liste des tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }

            async with session.post(
                f"{self.url}/api/mcp",
                headers=self._headers(),
                json=tools_request
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Erreur MCP tools: {resp.status} - {text}")
                tools_response = await resp.json()

            return tools_response.get("result", {}).get("tools", [])

    async def build_function_schemas(self) -> list[HAFunction]:
        """
        Construit les schémas de fonctions à partir des services HA.
        Ces schémas seront utilisés pour le fine-tuning.
        """
        services = await self.fetch_services()
        await self.fetch_entities()

        functions = []

        # Domaines prioritaires pour le contrôle domotique
        priority_domains = [
            "light", "switch", "climate", "cover", "lock", "alarm_control_panel",
            "media_player", "fan", "vacuum", "scene", "script", "automation"
        ]

        for domain_info in services:
            domain = domain_info.get("domain", "")
            if domain not in priority_domains:
                continue

            for service_name, service_info in domain_info.get("services", {}).items():
                func_name = f"{domain}.{service_name}"
                description = service_info.get("description", f"Service {func_name}")

                # Construire les paramètres
                params = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

                # entity_id est souvent requis
                params["properties"]["entity_id"] = {
                    "type": "string",
                    "description": f"ID de l'entité {domain} à contrôler"
                }

                # Ajouter les autres champs du service
                fields = service_info.get("fields", {})
                for field_name, field_info in fields.items():
                    if field_name == "entity_id":
                        continue

                    field_schema = {"description": field_info.get("description", "")}

                    # Déterminer le type
                    selector = field_info.get("selector", {})
                    if "number" in selector:
                        field_schema["type"] = "number"
                        num_sel = selector["number"]
                        if "min" in num_sel:
                            field_schema["minimum"] = num_sel["min"]
                        if "max" in num_sel:
                            field_schema["maximum"] = num_sel["max"]
                    elif "boolean" in selector:
                        field_schema["type"] = "boolean"
                    elif "select" in selector:
                        field_schema["type"] = "string"
                        field_schema["enum"] = selector["select"].get("options", [])
                    elif "color_rgb" in selector:
                        field_schema["type"] = "array"
                        field_schema["items"] = {"type": "integer"}
                        field_schema["description"] += " (format: [R, G, B])"
                    elif "color_temp" in selector:
                        field_schema["type"] = "integer"
                        field_schema["description"] += " (température en Kelvin ou mireds)"
                    else:
                        field_schema["type"] = "string"

                    params["properties"][field_name] = field_schema

                    if field_info.get("required", False):
                        params["required"].append(field_name)

                functions.append(HAFunction(
                    name=func_name,
                    description=description,
                    parameters=params
                ))

        self.functions = functions
        return functions

    def get_entities_by_domain(self, domain: str) -> list[dict]:
        """Retourne les entités d'un domaine spécifique."""
        return [e for e in self.entities if e["entity_id"].startswith(f"{domain}.")]

    def save_schemas(self, output_path: str):
        """Sauvegarde les schémas de fonctions en JSON."""
        schemas = [f.to_schema() for f in self.functions]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schemas, f, indent=2, ensure_ascii=False)
        print(f"Schémas sauvegardés: {output_path}")

    def save_entities(self, output_path: str):
        """Sauvegarde les entités en JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.entities, f, indent=2, ensure_ascii=False)
        print(f"Entités sauvegardées: {output_path}")


async def main():
    """Test de connexion à Home Assistant."""
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    client = HomeAssistantClient.from_env(config["home_assistant"]["url"])

    print("Connexion à Home Assistant...")
    print(f"URL: {client.url}")

    # Récupérer les schémas de fonctions
    print("\nRécupération des services...")
    functions = await client.build_function_schemas()
    print(f"Fonctions trouvées: {len(functions)}")

    for func in functions[:5]:
        print(f"  - {func.name}: {func.description[:60]}...")

    # Sauvegarder
    os.makedirs("data", exist_ok=True)
    client.save_schemas("data/function_schemas.json")
    client.save_entities("data/entities.json")

    # Résumé des entités par domaine
    print("\nEntités par domaine:")
    domains = {}
    for e in client.entities:
        domain = e["entity_id"].split(".")[0]
        domains[domain] = domains.get(domain, 0) + 1

    for domain, count in sorted(domains.items(), key=lambda x: -x[1])[:10]:
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
