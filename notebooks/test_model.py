"""
Script de test pour le modèle fine-tuné.
Simule le flow complet multi-turn avec injection des entités.
"""

import re

# Fake entities pour les tests (à remplacer par les vraies de ton HA)
FAKE_ENTITIES = {
    "light": ["light.salon", "light.cuisine", "light.chambre", "light.bureau", "light.couloir"],
    "switch": ["switch.tv", "switch.ventilateur", "switch.espresense_salon", "switch.prise_bureau"],
    "climate": ["climate.thermostat_salon", "climate.thermostat_bureau", "climate.climatisation"],
    "cover": ["cover.volets_salon", "cover.volets_chambre", "cover.store_terrasse"],
    "scene": ["scene.cinema", "scene.nuit", "scene.romantique", "scene.travail"],
    "fan": ["fan.ventilateur_salon", "fan.ventilateur_chambre"],
    "lock": ["lock.porte_entree", "lock.porte_garage"],
}


def parse_function_call(response: str) -> tuple[str, dict]:
    """Parse un appel de fonction FunctionGemma."""
    # Pattern: <start_function_call>call:func_name{params}<end_function_call>
    match = re.search(r'call:([a-z_\.]+)\{([^}]*)\}', response)
    if not match:
        return None, {}

    func_name = match.group(1)
    params_str = match.group(2)

    # Parser les paramètres
    params = {}
    for param in params_str.split(','):
        if ':' in param:
            key, value = param.split(':', 1)
            # Nettoyer les <escape>
            value = value.replace('<escape>', '').strip()
            params[key.strip()] = value

    return func_name, params


def test_full_flow(model, tokenizer, query: str):
    """
    Test le flow complet multi-turn:
    1. User query
    2. Model: get_entities
    3. Tool: liste des entités
    4. Model: action finale
    """
    import torch

    print(f"User: {query}")

    # === ÉTAPE 1: Requête → get_entities ===
    text1 = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
    inputs1 = tokenizer(text1, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out1 = model.generate(
            **inputs1,
            max_new_tokens=80,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    resp1 = tokenizer.decode(out1[0], skip_special_tokens=False)
    if "<start_of_turn>model" in resp1:
        resp1 = resp1.split("<start_of_turn>model")[-1]
    if "<end_of_turn>" in resp1:
        resp1 = resp1.split("<end_of_turn>")[0]
    resp1 = resp1.strip()

    print(f"Model (1): {resp1}")

    # Parser l'appel get_entities
    func_name, params = parse_function_call(resp1)

    if func_name != "get_entities":
        print(f"  ⚠️  Attendu get_entities, reçu: {func_name}")
        return

    domain = params.get("domain", "unknown")
    print(f"  → Domaine demandé: {domain}")

    # === ÉTAPE 2: Injecter la liste d'entités ===
    entities = FAKE_ENTITIES.get(domain, [f"{domain}.entity1", f"{domain}.entity2"])
    entities_str = ", ".join(entities)
    tool_response = f"Entités {domain} disponibles: {entities_str}"

    print(f"Tool: {tool_response}")

    # === ÉTAPE 3: Continuer le flow → action finale ===
    text2 = (
        f"<start_of_turn>user\n{query}<end_of_turn>\n"
        f"<start_of_turn>model\n{resp1}<end_of_turn>\n"
        f"<start_of_turn>tool\n{tool_response}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    inputs2 = tokenizer(text2, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out2 = model.generate(
            **inputs2,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    resp2 = tokenizer.decode(out2[0], skip_special_tokens=False)
    if "<start_of_turn>model" in resp2:
        resp2 = resp2.split("<start_of_turn>model")[-1]
    if "<end_of_turn>" in resp2:
        resp2 = resp2.split("<end_of_turn>")[0]
    resp2 = resp2.strip()

    print(f"Model (2): {resp2}")

    # Parser l'action finale
    action, action_params = parse_function_call(resp2)
    if action:
        print(f"  → Action: {action}")
        print(f"  → Params: {action_params}")

    print()


# === Pour utiliser dans Colab ===
TEST_CODE = '''
# Copier ce code dans une cellule Colab après l'entraînement

import re

FAKE_ENTITIES = {
    "light": ["light.salon", "light.cuisine", "light.chambre"],
    "switch": ["switch.tv", "switch.ventilateur", "switch.espresense"],
    "climate": ["climate.thermostat_salon", "climate.thermostat_bureau"],
    "scene": ["scene.cinema", "scene.nuit", "scene.romantique"],
}

def parse_function_call(response):
    match = re.search(r'call:([a-z_\\.]+)\\{([^}]*)\\}', response)
    if not match:
        return None, {}
    func_name = match.group(1)
    params = {}
    for param in match.group(2).split(','):
        if ':' in param:
            k, v = param.split(':', 1)
            params[k.strip()] = v.replace('<escape>', '').strip()
    return func_name, params

def test_full(query):
    print(f"User: {query}")

    # Step 1: get_entities
    text1 = f"<start_of_turn>user\\n{query}<end_of_turn>\\n<start_of_turn>model\\n"
    inputs1 = tokenizer(text1, return_tensors="pt").to(model.device)
    out1 = model.generate(**inputs1, max_new_tokens=80, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    resp1 = tokenizer.decode(out1[0], skip_special_tokens=False)
    resp1 = resp1.split("<start_of_turn>model")[-1].split("<end_of_turn>")[0].strip()
    print(f"Model (1): {resp1}")

    func, params = parse_function_call(resp1)
    domain = params.get("domain", "unknown")

    # Step 2: inject entities
    entities = FAKE_ENTITIES.get(domain, [f"{domain}.entity1"])
    tool_resp = f"Entités {domain} disponibles: {', '.join(entities)}"
    print(f"Tool: {tool_resp}")

    # Step 3: final action
    text2 = f"<start_of_turn>user\\n{query}<end_of_turn>\\n<start_of_turn>model\\n{resp1}<end_of_turn>\\n<start_of_turn>tool\\n{tool_resp}<end_of_turn>\\n<start_of_turn>model\\n"
    inputs2 = tokenizer(text2, return_tensors="pt").to(model.device)
    out2 = model.generate(**inputs2, max_new_tokens=100, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    resp2 = tokenizer.decode(out2[0], skip_special_tokens=False)
    resp2 = resp2.split("<start_of_turn>model")[-1].split("<end_of_turn>")[0].strip()
    print(f"Model (2): {resp2}")
    print()

# Tests
test_full("Allume la lumière du salon")
test_full("Mets le chauffage à 21 degrés")
test_full("Active la scène cinéma")
'''

if __name__ == "__main__":
    print("Code de test pour Colab:")
    print(TEST_CODE)
