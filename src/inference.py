"""
Script d'inférence pour le modèle FunctionGemma fine-tuné.
Permet de tester le modèle et de l'utiliser pour contrôler Home Assistant.
"""

import os
import re
import json
import argparse
import asyncio
from typing import Optional

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> dict:
    """Charge la configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class FunctionGemmaHA:
    """Modèle FunctionGemma fine-tuné pour Home Assistant."""

    def __init__(
        self,
        model_path: str,
        base_model: str = "google/functiongemma-270m-it",
        device: str = "auto",
        use_lora: bool = True,
        quantization: Optional[str] = None,  # None, "int8", "int4", "gptq"
    ):
        self.device = device
        self.quantization = quantization

        print(f"Chargement du modèle depuis {model_path}...")
        if quantization:
            print(f"Quantification: {quantization.upper()}")

        # Charger le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if not use_lora else base_model,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configuration de quantification
        quantization_config = self._get_quantization_config(quantization)

        # Dtype: bfloat16 pour tous (plus stable avec LoRA)
        torch_dtype = torch.bfloat16

        # Charger le modèle
        if use_lora:
            # Charger le modèle de base puis les poids LoRA
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                attn_implementation="eager",
                quantization_config=quantization_config,
            )
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                attn_implementation="eager",
                quantization_config=quantization_config,
            )

        self.model.eval()
        self._print_model_info()

        # Charger les schémas de fonctions
        self.function_schemas = self._load_function_schemas()

        print("Modèle chargé!")

    def _get_quantization_config(
        self, quantization: Optional[str]
    ) -> Optional[BitsAndBytesConfig]:
        """Retourne la configuration de quantification."""
        if quantization is None:
            return None

        if quantization == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True,  # Évite le cast bfloat16 -> float16
            )
        elif quantization == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 plus stable avec LoRA
                bnb_4bit_use_double_quant=False,  # Désactivé pour stabilité
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "gptq":
            # GPTQ nécessite un modèle pré-quantifié ou auto-gptq
            # On retourne None ici, le modèle GPTQ sera chargé différemment
            return None
        else:
            raise ValueError(f"Quantification non supportée: {quantization}")

    def _print_model_info(self):
        """Affiche les informations sur le modèle chargé."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Estimer la mémoire utilisée
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size_mb = (param_size + buffer_size) / (1024**2)

        print(f"  Paramètres totaux: {total_params:,}")
        print(f"  Paramètres entraînables: {trainable_params:,}")
        print(f"  Mémoire estimée: {total_size_mb:.1f} MB")

    def _load_function_schemas(self) -> list[dict]:
        """Charge les schémas de fonctions depuis le fichier."""
        schema_path = "data/function_schemas.json"
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                return json.load(f)
        return []

    def _parse_function_call(self, output: str) -> Optional[dict]:
        """
        Parse la sortie du modèle pour extraire l'appel de fonction.
        Format: <start_function_call>call:function_name{param:<escape>value<escape>}<end_function_call>
        """
        # Chercher le pattern d'appel de fonction
        pattern = r"<start_function_call>call:([^{]+)\{([^}]*)\}<end_function_call>"
        match = re.search(pattern, output)

        if not match:
            return None

        func_name = match.group(1).strip()
        params_str = match.group(2).strip()

        # Parser les paramètres
        params = {}
        if params_str:
            # Format: param:<escape>value<escape>,param2:value2
            param_pattern = r"([^:,]+):(?:<escape>([^<]*)<escape>|([^,]*))"
            for pmatch in re.finditer(param_pattern, params_str):
                key = pmatch.group(1).strip()
                value = pmatch.group(2) if pmatch.group(2) else pmatch.group(3)

                # Essayer de parser comme JSON (pour les nombres, listes, etc.)
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass

                params[key] = value

        return {"name": func_name, "parameters": params}

    def generate(
        self,
        user_query: str,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
    ) -> dict:
        """
        Génère un appel de fonction à partir d'une requête utilisateur.

        Args:
            user_query: La commande de l'utilisateur en français
            max_new_tokens: Nombre maximum de tokens à générer
            temperature: Température pour la génération

        Returns:
            Dict avec 'function_call' (si trouvé) et 'raw_output'
        """
        # Construire les messages
        messages = [
            {
                "role": "developer",
                "content": "Tu es un assistant qui contrôle une maison intelligente avec Home Assistant. Tu dois appeler les fonctions appropriées pour répondre aux demandes de l'utilisateur.",
            },
            {"role": "user", "content": user_query},
        ]

        # Appliquer le chat template
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=self.function_schemas[:20],  # Limiter le nombre de tools
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            # Fallback manuel
            text = ""
            text += "<start_of_turn>developer\n"
            text += messages[0]["content"]
            text += "<end_of_turn>\n"
            text += "<start_of_turn>user\n"
            text += messages[1]["content"]
            text += "<end_of_turn>\n"
            text += "<start_of_turn>model\n"

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
            )

        # Déplacer vers le device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Générer
        with torch.no_grad():
            # Utiliser greedy decoding pour les modèles quantifiés (plus stable)
            if self.quantization:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding pour éviter NaN avec quantification
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        # Décoder la sortie
        input_length = inputs["input_ids"].shape[1]
        generated = outputs[0][input_length:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=False)

        # Parser l'appel de fonction
        function_call = self._parse_function_call(raw_output)

        return {
            "function_call": function_call,
            "raw_output": raw_output,
        }


async def execute_function_call(func_call: dict, ha_url: str, ha_token: str) -> dict:
    """
    Exécute un appel de fonction sur Home Assistant.

    Args:
        func_call: L'appel de fonction parsé
        ha_url: URL de Home Assistant
        ha_token: Token d'authentification

    Returns:
        La réponse de Home Assistant
    """
    import aiohttp

    name = func_call["name"]
    params = func_call["parameters"]

    # Séparer domaine et service
    if "." in name:
        domain, service = name.split(".", 1)
    else:
        return {"error": f"Format de fonction invalide: {name}"}

    url = f"{ha_url}/api/services/{domain}/{service}"

    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=params) as resp:
            if resp.status == 200:
                return {"success": True, "response": await resp.json()}
            else:
                return {
                    "success": False,
                    "status": resp.status,
                    "error": await resp.text(),
                }


def interactive_mode(model: FunctionGemmaHA, config: dict):
    """Mode interactif pour tester le modèle."""
    load_dotenv()

    ha_url = config["home_assistant"]["url"]
    ha_token = os.getenv("HA_TOKEN")

    print("\n" + "=" * 50)
    print("Mode interactif - FunctionGemma Home Assistant")
    print("=" * 50)
    print("Tapez vos commandes en français.")
    print("Commandes spéciales:")
    print("  !exec - Exécuter le dernier appel de fonction")
    print("  !quit - Quitter")
    print("=" * 50 + "\n")

    last_function_call = None

    while True:
        try:
            user_input = input("Vous: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "!quit":
                print("Au revoir!")
                break

            if user_input.lower() == "!exec":
                if last_function_call:
                    print(f"Exécution de {last_function_call['name']}...")
                    result = asyncio.run(
                        execute_function_call(last_function_call, ha_url, ha_token)
                    )
                    if result.get("success"):
                        print("Commande exécutée avec succès!")
                    else:
                        print(f"Erreur: {result.get('error', 'Inconnue')}")
                else:
                    print("Aucun appel de fonction à exécuter.")
                continue

            # Générer la réponse
            result = model.generate(user_input)

            if result["function_call"]:
                last_function_call = result["function_call"]
                print(f"\nFonction: {result['function_call']['name']}")
                print(
                    f"Paramètres: {json.dumps(result['function_call']['parameters'], ensure_ascii=False, indent=2)}"
                )
                print("\n(Tapez !exec pour exécuter)")
            else:
                print(f"\nSortie brute: {result['raw_output']}")
                print("Aucun appel de fonction détecté.")

            print()

        except KeyboardInterrupt:
            print("\nAu revoir!")
            break
        except Exception as e:
            print(f"Erreur: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inférence avec FunctionGemma HA")
    parser.add_argument(
        "--model-path",
        type=str,
        default="output/final",
        help="Chemin vers le modèle fine-tuné",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Chemin vers la configuration"
    )
    parser.add_argument(
        "--query", type=str, help="Requête unique (mode non-interactif)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Exécuter automatiquement l'appel de fonction",
    )
    parser.add_argument(
        "--no-lora", action="store_true", help="Charger un modèle complet (pas LoRA)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["int8", "int4", "gptq"],
        default=None,
        help="Mode de quantification: int8, int4 (bitsandbytes) ou gptq",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Charger le modèle
    model = FunctionGemmaHA(
        model_path=args.model_path,
        base_model=config["model"]["name"],
        use_lora=not args.no_lora,
        quantization=args.quantization,
    )

    if args.query:
        # Mode requête unique
        result = model.generate(args.query)

        if result["function_call"]:
            print(f"Fonction: {result['function_call']['name']}")
            print(
                f"Paramètres: {json.dumps(result['function_call']['parameters'], ensure_ascii=False, indent=2)}"
            )

            if args.execute:
                load_dotenv()
                ha_url = config["home_assistant"]["url"]
                ha_token = os.getenv("HA_TOKEN")

                exec_result = asyncio.run(
                    execute_function_call(result["function_call"], ha_url, ha_token)
                )
                print(f"Résultat: {exec_result}")
        else:
            print(f"Sortie: {result['raw_output']}")
    else:
        # Mode interactif
        interactive_mode(model, config)


if __name__ == "__main__":
    main()
