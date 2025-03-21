import os
import json
import pytesseract
from PIL import Image
from transformers import pipeline
from dotenv import load_dotenv
import re

# Charger clé API Hugging Face
load_dotenv()
hf_token = os.getenv('HUGGING_FACE_TOKEN')

# Dossier de factures et résultats
invoice_dir = './Factures'
results_dir = './Factures/Resultats'

os.makedirs(results_dir, exist_ok=True)

# Initialiser pipeline HF avec Mistral
pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=hf_token,
    device_map="auto"
)

# Fonction pour OCR d'une image
def ocr_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='fra')
    return text

# Fonction pour nettoyer texte OCR
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\u00C0-\u00FF]', '', text)
    return text.strip()

# Fonction pour générer le prompt à Mistral
def generate_prompt(text):
    prompt = f"""
    [INST] Voici un texte OCR d'une facture française :

    {text}

    Extrais strictement les informations suivantes au format JSON sans aucun texte supplémentaire :
    {{
        "numéro_facture": "string",
        "date_facture": "JJ/MM/AAAA",
        "nom_fournisseur": "string",
        "adresse_fournisseur": "string complète",
        "articles": [
            {{
                "description": "string",
                "quantité": nombre entier,
                "prix_unitaire_ht": nombre décimal,
                "montant_ht": nombre décimal
            }}
        ],
        "remise": nombre décimal négatif ou 0,
        "total_ht": nombre décimal,
        "total_tva": nombre décimal,
        "total_ttc": nombre décimal,
        "mode_règlement": "string",
        "date_échéance": "JJ/MM/AAAA",
        "iban": "IBAN",
        "bic": "BIC"
    }} [/INST]
    """
    return prompt.strip()

# Traitement des factures
for filename in os.listdir(invoice_dir):
    if filename.lower().endswith('.png'):
        invoice_path = os.path.join(invoice_dir, filename)
        print(f"Traitement : {invoice_path}")

        # OCR + Nettoyage
        raw_text = ocr_image(invoice_path)
        clean_ocr = clean_text(raw_text)

        # Prompt HF
        prompt = generate_prompt(clean_ocr)

        # Appel à Mistral
        try:
            result = pipe(
                prompt,
                max_new_tokens=1024,
                temperature=0.1,
                return_full_text=False
            )

            raw_output = result[0]['generated_text']
            print("Réponse brute IA:", raw_output)

            # Extraction du JSON
            json_match = re.search(r'({.*})', raw_output, re.DOTALL)
            if json_match:
                invoice_data = json.loads(json_match.group(1))
            else:
                invoice_data = {"erreur": "JSON non détecté", "texte": raw_output[:500]}

        except Exception as e:
            invoice_data = {"erreur": str(e)}

        # Sauvegarde JSON
        result_filename = filename.replace('.png', '.json')
        with open(os.path.join(results_dir, result_filename), 'w', encoding='utf-8') as f:
            json.dump(invoice_data, f, ensure_ascii=False, indent=4)

        print(f"Résultat enregistré : {result_filename}")

print("Traitement terminé.")
