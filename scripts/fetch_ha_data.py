#!/usr/bin/env python3
"""
Script pour récupérer les données de Home Assistant.
Étape 1 du pipeline de fine-tuning.
"""

import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
from ha_client import main

if __name__ == "__main__":
    asyncio.run(main())
