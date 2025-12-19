#!/usr/bin/env python3
"""
Script d'export: Merge LoRA + Conversion GGUF.

Pipeline:
    output/final/ (LoRA)
         ↓
    [MERGE avec modèle de base]
         ↓
    functiongemma-ha-merged/ (HuggingFace)
         ↓
    [CONVERT via llama.cpp]
         ↓
    model.gguf (pour llama.cpp)

Usage:
    python scripts/export_gguf.py
    python scripts/export_gguf.py --lora-path output/final --output model.gguf
    python scripts/export_gguf.py --quantize q4_0
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import login as hf_login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> dict:
    """Charge la configuration du projet."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def merge_lora(
    base_model_name: str,
    lora_path: str,
    output_path: str,
    device: str = "auto",
) -> None:
    """
    Fusionne les poids LoRA avec le modèle de base.

    Args:
        base_model_name: Nom du modèle de base HuggingFace
        lora_path: Chemin vers les poids LoRA
        output_path: Chemin de sortie pour le modèle fusionné
        device: Device pour le chargement (auto, cpu, cuda)
    """
    print("=" * 60)
    print("ÉTAPE 1: Fusion LoRA")
    print("=" * 60)
    print(f"  Modèle de base: {base_model_name}")
    print(f"  Poids LoRA: {lora_path}")
    print(f"  Sortie: {output_path}")
    print()

    # Vérifier que les poids LoRA existent
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"Poids LoRA non trouvés: {lora_path}")

    # Login HuggingFace si nécessaire
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token)
        print("  Connecté à Hugging Face")

    # Charger le modèle de base
    print("  Chargement du modèle de base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    # Charger le tokenizer
    print("  Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    # Charger et fusionner les poids LoRA
    print("  Fusion des poids LoRA...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = model.merge_and_unload()

    # Sauvegarder le modèle fusionné
    print(f"  Sauvegarde vers {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("  Fusion terminée!")
    print()


def find_llama_cpp() -> Path:
    """
    Trouve le répertoire llama.cpp.

    Cherche dans l'ordre:
    1. Variable d'environnement LLAMA_CPP_PATH
    2. ../llama.cpp (relatif au projet)
    3. ~/llama.cpp
    4. /opt/llama.cpp
    """
    # Variable d'environnement
    env_path = os.getenv("LLAMA_CPP_PATH")
    if env_path and os.path.exists(env_path):
        return Path(env_path)

    # Chemins standards
    project_root = Path(__file__).parent.parent
    candidates = [
        project_root / "llama.cpp",
        project_root.parent / "llama.cpp",
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
    ]

    for path in candidates:
        if path.exists() and (path / "convert_hf_to_gguf.py").exists():
            return path

    return None


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantize: str = None,
    llama_cpp_path: str = None,
) -> None:
    """
    Convertit un modèle HuggingFace en format GGUF.

    Args:
        model_path: Chemin vers le modèle HuggingFace
        output_path: Chemin de sortie pour le fichier GGUF
        quantize: Type de quantization (q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32)
        llama_cpp_path: Chemin vers llama.cpp (optionnel)
    """
    print("=" * 60)
    print("ÉTAPE 2: Conversion GGUF")
    print("=" * 60)
    print(f"  Modèle source: {model_path}")
    print(f"  Fichier GGUF: {output_path}")
    if quantize:
        print(f"  Quantization: {quantize}")
    print()

    # Trouver llama.cpp
    if llama_cpp_path:
        llama_cpp = Path(llama_cpp_path)
    else:
        llama_cpp = find_llama_cpp()

    if not llama_cpp:
        print("  ERREUR: llama.cpp non trouvé!")
        print()
        print("  Pour installer llama.cpp:")
        print("    git clone https://github.com/ggerganov/llama.cpp")
        print("    cd llama.cpp && pip install -r requirements.txt")
        print()
        print("  Ou définissez LLAMA_CPP_PATH:")
        print("    export LLAMA_CPP_PATH=/chemin/vers/llama.cpp")
        raise FileNotFoundError("llama.cpp non trouvé")

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Script de conversion non trouvé: {convert_script}")

    print(f"  llama.cpp: {llama_cpp}")

    # Construire la commande de conversion
    # Sortie temporaire si quantization demandée
    if quantize:
        temp_output = output_path.replace(".gguf", "_f16.gguf")
        outtype = "f16"
    else:
        temp_output = output_path
        outtype = "f16"

    cmd = [
        sys.executable,
        str(convert_script),
        model_path,
        "--outfile", temp_output,
        "--outtype", outtype,
    ]

    print(f"  Commande: {' '.join(cmd)}")
    print()

    # Exécuter la conversion
    print("  Conversion en cours...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("  ERREUR lors de la conversion:")
        print(result.stderr)
        raise RuntimeError("Échec de la conversion GGUF")

    print("  Conversion f16 terminée!")

    # Quantization si demandée
    if quantize:
        print()
        print(f"  Quantization {quantize} en cours...")

        quantize_bin = llama_cpp / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"

        if not quantize_bin.exists():
            print("  ATTENTION: llama-quantize non trouvé, compilation nécessaire")
            print("    cd llama.cpp && cmake -B build && cmake --build build --target llama-quantize")
            print(f"  Le modèle f16 est disponible: {temp_output}")
            return

        cmd_quant = [
            str(quantize_bin),
            temp_output,
            output_path,
            quantize,
        ]

        print(f"  Commande: {' '.join(cmd_quant)}")
        result = subprocess.run(cmd_quant, capture_output=True, text=True)

        if result.returncode != 0:
            print("  ERREUR lors de la quantization:")
            print(result.stderr)
            raise RuntimeError("Échec de la quantization")

        # Supprimer le fichier f16 temporaire
        os.remove(temp_output)
        print(f"  Quantization {quantize} terminée!")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Export: Merge LoRA + Conversion GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Export avec valeurs par défaut
  python scripts/export_gguf.py

  # Spécifier les chemins
  python scripts/export_gguf.py --lora-path output/final --output model.gguf

  # Avec quantization Q4
  python scripts/export_gguf.py --quantize q4_0

  # Quantization Q8 (meilleure qualité)
  python scripts/export_gguf.py --quantize q8_0

Types de quantization disponibles:
  f32   - Float 32 bits (le plus précis, le plus lourd)
  f16   - Float 16 bits (bon compromis par défaut)
  q8_0  - Int 8 bits (bonne qualité, ~50% taille f16)
  q5_1  - Int 5 bits avec delta
  q5_0  - Int 5 bits
  q4_1  - Int 4 bits avec delta
  q4_0  - Int 4 bits (le plus léger, qualité réduite)
        """,
    )

    parser.add_argument(
        "--lora-path",
        type=str,
        default="output/final",
        help="Chemin vers les poids LoRA (défaut: output/final)",
    )
    parser.add_argument(
        "--merged-path",
        type=str,
        default="functiongemma-ha-merged",
        help="Chemin pour le modèle fusionné (défaut: functiongemma-ha-merged)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="model.gguf",
        help="Fichier GGUF de sortie (défaut: model.gguf)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Modèle de base HuggingFace (défaut: depuis config.yaml)",
    )
    parser.add_argument(
        "--quantize",
        "-q",
        type=str,
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
        default=None,
        help="Type de quantization (défaut: f16)",
    )
    parser.add_argument(
        "--llama-cpp",
        type=str,
        default=None,
        help="Chemin vers le répertoire llama.cpp",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Sauter l'étape de fusion (utiliser modèle déjà fusionné)",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Sauter la conversion GGUF (merge seulement)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device pour le chargement du modèle (défaut: auto)",
    )

    args = parser.parse_args()

    # Charger la configuration
    config = load_config()
    base_model = args.base_model or config.get("model", {}).get(
        "name", "google/functiongemma-270m-it"
    )

    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           FunctionGemma-HA Export Pipeline                 ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    try:
        # Étape 1: Fusion LoRA
        if not args.skip_merge:
            merge_lora(
                base_model_name=base_model,
                lora_path=args.lora_path,
                output_path=args.merged_path,
                device=args.device,
            )
        else:
            print("Fusion LoRA: IGNORÉE (--skip-merge)")
            print()

        # Étape 2: Conversion GGUF
        if not args.skip_convert:
            convert_to_gguf(
                model_path=args.merged_path,
                output_path=args.output,
                quantize=args.quantize,
                llama_cpp_path=args.llama_cpp,
            )
        else:
            print("Conversion GGUF: IGNORÉE (--skip-convert)")
            print()

        # Résumé
        print("=" * 60)
        print("EXPORT TERMINÉ!")
        print("=" * 60)
        if not args.skip_merge:
            print(f"  Modèle HuggingFace: {args.merged_path}/")
        if not args.skip_convert:
            print(f"  Modèle GGUF: {args.output}")
        print()
        print("Pour utiliser le modèle:")
        print(f"  python src/chat.py --model {args.output}")
        print()

    except FileNotFoundError as e:
        print(f"\nERREUR: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nERREUR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERREUR inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
