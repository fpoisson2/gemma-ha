#!/usr/bin/env python3
"""
Script pour générer le dataset d'entraînement.
Étape 2 du pipeline de fine-tuning.
"""

import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
from dataset_generator import main

if __name__ == "__main__":
    asyncio.run(main())
