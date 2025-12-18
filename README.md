# FunctionGemma Home Assistant

Fine-tuning de [FunctionGemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) pour contrôler Home Assistant en français.

## Prérequis

- Python 3.11+
- Home Assistant avec un token d'accès longue durée
- Compte Hugging Face avec accès à FunctionGemma (modèle gated)
- **Pour l'entraînement local**: GPU NVIDIA avec CUDA (RTX 3090/4090 recommandé)
- **Pour l'entraînement cloud**: Compte Google Colab (gratuit ou Pro)

## Installation

```bash
# Cloner le repo
cd gemma-ha

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Configuration

1. Copier `.env.example` vers `.env` et remplir les tokens
2. Modifier `config.yaml` selon vos besoins

## Pipeline de Fine-tuning

### Étape 1: Récupérer les données de Home Assistant

```bash
python scripts/fetch_ha_data.py
```

Cela récupère:
- Les schémas de fonctions (services HA)
- La liste des entités

### Étape 2: Générer le dataset

```bash
python scripts/generate_dataset.py
```

Génère des paires (requête utilisateur → appel de fonction) en français.

### Étape 3: Lancer l'entraînement

#### Option A: Entraînement local (GPU requis)

```bash
python scripts/train_model.py
```

#### Option B: Entraînement sur Google Colab (recommandé)

1. Ouvrir [Google Colab](https://colab.google.com)
2. File → Upload notebook → `notebooks/train_colab.ipynb`
3. Runtime → Change runtime type → **T4 GPU**
4. Configurer le token HuggingFace dans Colab Secrets (`HF_TOKEN`)
5. Exécuter les cellules et uploader `data/train.jsonl` + `data/val.jsonl`

| Option | GPU | Temps estimé | Coût |
|--------|-----|--------------|------|
| Colab gratuit | T4 | 1-2h | Gratuit |
| Colab Pro | A100 | 30-45min | ~$12/mois |

Le modèle fine-tuné sera sauvegardé dans `output/final/`.

## Utilisation

### Mode interactif

```bash
python src/inference.py
```

### Commande unique

```bash
python src/inference.py --query "Allume la lumière du salon"
```

### Exécution automatique

```bash
python src/inference.py --query "Allume la lumière du salon" --execute
```

## Structure du projet

```
gemma-ha/
├── config.yaml          # Configuration
├── requirements.txt     # Dépendances
├── .env                 # Tokens (non versionné)
├── data/                # Données générées
│   ├── train.jsonl      # Dataset d'entraînement
│   └── val.jsonl        # Dataset de validation
├── output/              # Modèle fine-tuné
├── notebooks/
│   └── train_colab.ipynb  # Notebook pour Google Colab
├── src/
│   ├── ha_client.py     # Client Home Assistant
│   ├── dataset_generator.py
│   ├── train.py         # Script d'entraînement
│   └── inference.py     # Script d'inférence
└── scripts/             # Scripts d'exécution
```

## Format des appels de fonction

Le modèle génère des appels au format FunctionGemma:

```
<start_function_call>call:light.turn_on{entity_id:<escape>light.salon<escape>}<end_function_call>
```

## Personnalisation

### Ajouter des templates de phrases

Modifier `TEMPLATES_FR` dans `src/dataset_generator.py` pour ajouter des variations de requêtes.

### Ajuster l'entraînement

Modifier `config.yaml`:
- `lora_r`: Rang LoRA (16 par défaut)
- `learning_rate`: Taux d'apprentissage
- `num_epochs`: Nombre d'époques
