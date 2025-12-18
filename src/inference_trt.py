"""
Inférence ultra-rapide avec TensorRT-LLM pour FunctionGemma.
Nécessite d'avoir buildé l'engine avec scripts/tensorrt_build.sh
"""

import os
import sys
import json
import argparse
from pathlib import Path


def check_tensorrt_llm():
    """Vérifie si TensorRT-LLM est disponible."""
    try:
        import tensorrt_llm
        return True
    except ImportError:
        return False


class TensorRTInference:
    """Inférence TensorRT-LLM pour FunctionGemma."""

    def __init__(self, engine_dir: str, tokenizer_dir: str):
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Chargement de l'engine TensorRT: {engine_dir}")
        self.runner = ModelRunner.from_dir(engine_dir)
        print("Engine chargé!")

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Génère une réponse."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        outputs = self.runner.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
        )

        output_ids = outputs[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        return response


def main():
    parser = argparse.ArgumentParser(description="Inférence TensorRT-LLM")
    parser.add_argument(
        "--engine-dir",
        type=str,
        default="trt-engine",
        help="Répertoire de l'engine TensorRT",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="functiongemma-ha",
        help="Répertoire du tokenizer",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Requête à exécuter",
    )
    args = parser.parse_args()

    if not check_tensorrt_llm():
        print("TensorRT-LLM non disponible.")
        print("Exécutez ce script dans le container Docker:")
        print("  docker run --gpus all -v $(pwd):/workspace -it nvcr.io/nvidia/tensorrt-llm:latest")
        sys.exit(1)

    if not Path(args.engine_dir).exists():
        print(f"Engine non trouvé: {args.engine_dir}")
        print("Buildez d'abord avec: ./scripts/tensorrt_build.sh")
        sys.exit(1)

    # Charger le modèle
    model = TensorRTInference(args.engine_dir, args.tokenizer_dir)

    if args.query:
        # Mode single query
        prompt = f"<start_of_turn>user\n{args.query}<end_of_turn>\n<start_of_turn>model\n"
        import time
        start = time.perf_counter()
        response = model.generate(prompt)
        elapsed = time.perf_counter() - start
        print(f"Réponse: {response}")
        print(f"Temps: {elapsed*1000:.1f}ms")
    else:
        # Mode interactif
        print("\nMode interactif (quit pour quitter)")
        while True:
            try:
                query = input("\nVous: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break
                if not query:
                    continue

                prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
                import time
                start = time.perf_counter()
                response = model.generate(prompt, max_new_tokens=50)
                elapsed = time.perf_counter() - start

                print(f"Assistant: {response}")
                print(f"({elapsed*1000:.1f}ms)")

            except KeyboardInterrupt:
                print("\nAu revoir!")
                break


if __name__ == "__main__":
    main()
