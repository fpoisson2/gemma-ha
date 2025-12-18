#!/usr/bin/env python3
"""
Script pour lancer l'entraînement.
Étape 3 du pipeline de fine-tuning.
"""

import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import main

if __name__ == "__main__":
    main()
