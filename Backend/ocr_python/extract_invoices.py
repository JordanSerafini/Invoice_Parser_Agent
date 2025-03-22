import os
# Configuration pour forcer GPU NVIDIA sans MESA
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force utilisation du premier GPU

# Configuration pour débuguer CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Pour des erreurs plus détaillées

# Désactive complètement les logs de MESA
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import pytesseract
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, DonutProcessor, VisionEncoderDecoderModel
from dotenv import load_dotenv
import re
from huggingface_hub import login
import torch
import gc
import time
import sys
import requests

# Fonction de log détaillé
def log_info(message):
    print(f"[INFO] {message}")
    sys.stdout.flush()  # Force l'affichage immédiat

# Fonction pour vérifier la mémoire GPU
def log_gpu_memory():
    if torch.cuda.is_available():
        log_info(f"Mémoire GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        log_info(f"Mémoire allouée: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        log_info(f"Mémoire réservée: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        log_info(f"Mémoire max allouée: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    else:
        log_info("CUDA n'est pas disponible, impossible de vérifier la mémoire GPU")

# Fonction pour libérer la mémoire GPU
def free_gpu_memory():
    # Libérer la mémoire GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_info("Mémoire GPU libérée")

# Configuration initiale et info CUDA
log_info(f"Version PyTorch: {torch.__version__}")
log_info(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log_info(f"Appareil CUDA: {torch.cuda.get_device_name(0)}")
    log_info(f"Version CUDA: {torch.version.cuda}")
    log_info(f"Nombre de GPUs: {torch.cuda.device_count()}")
    log_info(f"Index GPU actuel: {torch.cuda.current_device()}")
    log_gpu_memory()

# Tester l'allocation mémoire GPU
if torch.cuda.is_available():
    try:
        log_info("Test allocation mémoire GPU...")
        # Créer un petit tenseur sur GPU pour tester l'accès
        test_tensor = torch.ones(100, 100, device="cuda")
        log_info(f"Test tensor créé sur GPU: {test_tensor.device}")
        del test_tensor
        free_gpu_memory()
        log_info("Test allocation réussi")
    except Exception as e:
        log_info(f"Erreur lors du test d'allocation: {str(e)}")

# Charger clé API Hugging Face et se connecter
load_dotenv()
hf_token = os.getenv('HUGGING_FACE_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')
log_info(f"Token HF disponible: {hf_token is not None}")
log_info(f"Token OpenAI disponible: {openai_api_key is not None}")
login(token=hf_token)

# Dossier de factures et résultats
invoice_dir = './Factures'
results_dir = './Factures/Resultats'
ocr_results_dir = os.path.join(results_dir, 'OCR_Mistral')
donut_results_dir = os.path.join(results_dir, 'Donut')
final_results_dir = os.path.join(results_dir, 'Final_JSON')

# Création des dossiers de résultats
os.makedirs(results_dir, exist_ok=True)
os.makedirs(ocr_results_dir, exist_ok=True)
os.makedirs(donut_results_dir, exist_ok=True)
os.makedirs(final_results_dir, exist_ok=True)

########################################
# PARTIE 1: Utilitaires                #
########################################

# Fonction pour appeler OpenAI pour corriger JSON mal formé
def fix_json_with_openai(text):
    if not openai_api_key:
        log_info("Clé API OpenAI non disponible, impossible de corriger le JSON")
        return None
    
    try:
        log_info("Tentative de correction JSON via OpenAI...")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        prompt = f"""Le texte suivant est censé être du JSON mais il est mal formé. 
Réécris-le proprement en JSON strict sans aucun commentaire ni texte additionnel.

{text}"""
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            log_info("Correction JSON réussie via OpenAI")
            return content
        else:
            log_info(f"Erreur OpenAI: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        log_info(f"Erreur lors de la correction JSON via OpenAI: {str(e)}")
        return None

# Fonction pour parser JSON de manière robuste
def parse_json_safely(raw_output):
    try:
        # Première tentative: extraction standard
        json_match = re.search(r'({.*})', raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                # Si le parsing échoue, on tente de nettoyer
                json_text = json_match.group(1)
                # Remplacer les erreurs courantes
                json_text = re.sub(r',\s*}', '}', json_text)
                json_text = re.sub(r',\s*]', ']', json_text)
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    # Si toujours échec, on utilise OpenAI
                    fixed_json = fix_json_with_openai(json_text)
                    if fixed_json:
                        try:
                            return json.loads(fixed_json)
                        except json.JSONDecodeError:
                            log_info("Impossible de corriger le JSON même avec OpenAI")
        
        # Si aucun JSON n'est trouvé ou si parsing a échoué
        log_info("JSON non détecté ou impossible à parser")
        return {"erreur": "JSON invalide", "texte_brut": raw_output[:500]}
    
    except Exception as e:
        log_info(f"Erreur inattendue lors du parsing JSON: {str(e)}")
        return {"erreur": str(e), "texte_brut": raw_output[:500]}

# Fonction pour OCR d'une image avec Tesseract
def ocr_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='fra')
        return text
    except Exception as e:
        log_info(f"Erreur OCR: {str(e)}")
        return ""

# Fonction pour nettoyer texte OCR
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\u00C0-\u00FF]', '', text)
    return text.strip()

########################################
# PARTIE 2: Extraction OCR + Mistral   #
########################################

# Fonction pour générer le prompt au format Mistral
def generate_prompt_extraction(text):
    prompt = f"""<s>[INST] Tu es un expert en extraction d'informations de factures. Analyse cette facture française et extrait les informations structurées au format JSON.

Facture: 
{text}

Format de réponse attendu (JSON uniquement):
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
}}

Réponds uniquement avec le JSON, sans commentaires additionnels.
[/INST]</s>
"""
    return prompt

# Fonction pour extraire avec Mistral
def extract_with_mistral(text):
    log_info("=== EXTRACTION AVEC MISTRAL (OCR+LLM) ===")
    
    # Chargement du modèle Mistral
    log_info("Chargement du modèle Mistral...")
    mistral_model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Version v0.1 open source

    # Force GC avant chargement du modèle
    free_gpu_memory()
    log_gpu_memory()

    try:
        log_info(f"Chargement tokenizer Mistral: {time.strftime('%H:%M:%S')}")
        mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
        log_info("Tokenizer Mistral chargé avec succès")
        
        log_info(f"Chargement modèle Mistral: {time.strftime('%H:%M:%S')}")
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # Configuration pour quantification 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,              # Quantification 4-bit pour économiser mémoire
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        mistral_model = AutoModelForCausalLM.from_pretrained(
            mistral_model_name,
            quantization_config=bnb_config,  # Utiliser quantification 4-bit
            device_map="auto"                # Laisser transformers optimiser
        )
        log_info("Modèle Mistral chargé avec succès")
        
        # Génération du prompt et inférence
        log_info("Génération du prompt pour Mistral...")
        prompt = generate_prompt_extraction(text)
        
        log_info("Début inférence Mistral...")
        try:
            # Encodage
            inputs = mistral_tokenizer(prompt, return_tensors="pt").to("cuda")
            
            # Génération avec optimisation mémoire
            with torch.cuda.amp.autocast():
                outputs = mistral_model.generate(
                    inputs.input_ids,
                    max_new_tokens=2048,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=mistral_tokenizer.eos_token_id,
                    eos_token_id=mistral_tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Décodage et parsing du résultat
            output_text = mistral_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Libération mémoire
            del outputs, inputs
            free_gpu_memory()
            
            # Parsing sécurisé du JSON
            result = parse_json_safely(output_text)
            log_info("Extraction avec Mistral terminée avec succès")
            
            # Libération du modèle et tokenizer
            del mistral_model, mistral_tokenizer
            free_gpu_memory()
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            log_info("ERREUR: Mémoire GPU insuffisante pour l'inférence Mistral")
            free_gpu_memory()
            return {"erreur": "Mémoire GPU insuffisante"}
            
        except Exception as e:
            log_info(f"Erreur lors de l'inférence Mistral: {str(e)}")
            return {"erreur": f"Erreur inférence: {str(e)}"}
    
    except Exception as e:
        log_info(f"Erreur lors du chargement du modèle Mistral: {str(e)}")
        return {"erreur": f"Erreur chargement modèle: {str(e)}"}
    
    finally:
        # S'assurer que la mémoire est bien libérée
        free_gpu_memory()

########################################
# PARTIE 3: Extraction avec Donut      #
########################################

# Fonction pour extraire avec Donut
def extract_with_donut(image_path):
    log_info("=== EXTRACTION AVEC DONUT (VISION MODEL) ===")
    
    # Chargement du modèle Donut
    log_info("Chargement du modèle Donut...")
    donut_model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"

    # S'assurer que la mémoire est libre
    free_gpu_memory()
    log_gpu_memory()

    try:
        log_info(f"Chargement processor et modèle Donut: {time.strftime('%H:%M:%S')}")
        donut_processor = DonutProcessor.from_pretrained(donut_model_name)
        donut_model = VisionEncoderDecoderModel.from_pretrained(donut_model_name).to("cuda")
        log_info("Modèle Donut chargé avec succès")
        
        # Traitement de l'image
        log_info("Préparation de l'image pour Donut...")
        try:
            # Charger et optimiser l'image
            image = Image.open(image_path).convert("RGB")
            # Redimensionner pour stabilité et mémoire
            image = image.resize((1280, 960), Image.LANCZOS)
            
            # Préparation du tenseur d'entrée
            pixel_values = donut_processor(image, return_tensors="pt").pixel_values.to("cuda")
            
            # Génération avec paramètres optimisés
            log_info("Début inférence Donut...")
            with torch.cuda.amp.autocast():
                outputs = donut_model.generate(
                    pixel_values,
                    max_length=512,
                    num_beams=1,  # Moins gourmand en RAM
                    early_stopping=True
                )
            
            # Décodage et traitement du résultat
            sequence = donut_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            sequence = sequence.replace("<s>", "").replace("</s>", "")
            
            # Libération mémoire
            del outputs, pixel_values
            free_gpu_memory()
            
            try:
                # Conversion en JSON si possible (utiliser notre parser robuste)
                donut_data = parse_json_safely(sequence)
                log_info("Extraction avec Donut réussie")
                
                # Si c'est un dictionnaire d'erreur, retourner directement
                if "erreur" in donut_data:
                    log_info("Donut n'a pas produit un JSON valide")
                    result = {"erreur": "JSON invalide", "texte_brut": sequence}
                else:
                    # Adaptation du format Donut au format attendu
                    result = {
                        "numéro_facture": donut_data.get("invoice_number", ""),
                        "date_facture": donut_data.get("date", ""),
                        "nom_fournisseur": donut_data.get("company", {}).get("name", ""),
                        "adresse_fournisseur": donut_data.get("company", {}).get("address", ""),
                        "articles": [],
                        "total_ht": donut_data.get("subtotal", 0),
                        "total_tva": donut_data.get("tax", 0),
                        "total_ttc": donut_data.get("total", 0),
                        "mode_règlement": donut_data.get("payment_method", ""),
                        "date_échéance": donut_data.get("due_date", ""),
                        "iban": donut_data.get("iban", ""),
                        "bic": donut_data.get("bic", "")
                    }
                    
                    # Traitement des articles si disponibles
                    if "items" in donut_data:
                        for item in donut_data["items"]:
                            result["articles"].append({
                                "description": item.get("description", ""),
                                "quantité": item.get("count", 1),
                                "prix_unitaire_ht": item.get("price", 0),
                                "montant_ht": item.get("total_price", 0)
                            })
            
            except Exception as e:
                log_info(f"Erreur de traitement du résultat Donut: {str(e)}")
                result = {"erreur": str(e), "texte_brut": sequence[:500]}
            
        except torch.cuda.OutOfMemoryError:
            log_info("ERREUR: Mémoire GPU insuffisante pour l'inférence Donut")
            result = {"erreur": "Mémoire GPU insuffisante"}
            
        except Exception as e:
            log_info(f"Erreur lors du traitement de l'image pour Donut: {str(e)}")
            result = {"erreur": f"Erreur traitement image: {str(e)}"}
        
        # Libération du modèle et processor
        del donut_model, donut_processor
        free_gpu_memory()
        
        return result
    
    except Exception as e:
        log_info(f"Erreur lors du chargement du modèle Donut: {str(e)}")
        return {"erreur": f"Erreur chargement modèle: {str(e)}"}
    
    finally:
        # S'assurer que la mémoire est bien libérée
        free_gpu_memory()

########################################
# PARTIE 4: Fusion des Résultats       #
########################################

# Fonction pour générer le prompt de fusion
def generate_prompt_fusion(ocr_result, donut_result):
    prompt = f"""<s>[INST] Tu es un expert en fusion de données issues de l'extraction d'informations de factures. 

J'ai deux résultats d'extraction obtenus avec deux méthodes différentes, et je veux que tu les fusionnes intelligemment en un seul résultat consolidé.

Pour chaque champ, tu dois :
1. Déterminer quelle source est la plus fiable
2. Attribuer une note de confiance entre 0 et 1 pour chaque champ (1 = confiance maximale)
3. Ajouter une note de confiance globale pour l'ensemble du résultat

Résultat issu d'OCR + LLM :
{json.dumps(ocr_result, ensure_ascii=False, indent=2)}

Résultat issu de Donut (vision-language model) :
{json.dumps(donut_result, ensure_ascii=False, indent=2)}

Format de sortie attendu (JSON uniquement) :
{{
  "résultat_final": {{
    "numéro_facture": "...",
    "date_facture": "...",
    "nom_fournisseur": "...",
    "adresse_fournisseur": "...",
    "articles": [
      {{
        "description": "...",
        "quantité": ...,
        "prix_unitaire_ht": ...,
        "montant_ht": ...
      }}
    ],
    "remise": ...,
    "total_ht": ...,
    "total_tva": ...,
    "total_ttc": ...,
    "mode_règlement": "...",
    "date_échéance": "...",
    "iban": "...",
    "bic": "..."
  }},
  "notes_confiance": {{
    "numéro_facture": ...,
    "date_facture": ...,
    "nom_fournisseur": ...,
    "adresse_fournisseur": ...,
    "articles": ...,
    "remise": ...,
    "total_ht": ...,
    "total_tva": ...,
    "total_ttc": ...,
    "mode_règlement": ...,
    "date_échéance": ...,
    "iban": ...,
    "bic": ...
  }},
  "confiance_globale": ...
}}

Réponds uniquement avec le JSON, sans commentaires additionnels.
[/INST]</s>
"""
    return prompt

# Fonction pour fusionner les résultats
def fusion_resultats(ocr_result, donut_result):
    log_info("=== FUSION DES RÉSULTATS ===")
    
    # Vérifier si l'un des résultats contient une erreur
    if "erreur" in ocr_result and "erreur" in donut_result:
        log_info("Les deux méthodes d'extraction ont échoué, impossible de fusionner")
        return {
            "résultat_final": {"erreur": "Échec des deux méthodes d'extraction"},
            "notes_confiance": {"global": 0},
            "confiance_globale": 0
        }
    
    # Si une seule méthode a fonctionné, on utilise celle-ci avec une confiance réduite
    if "erreur" in ocr_result:
        log_info("OCR+Mistral a échoué, utilisation de Donut uniquement")
        return {
            "résultat_final": donut_result,
            "notes_confiance": {k: 0.6 for k in donut_result.keys() if k != "erreur"},
            "confiance_globale": 0.5
        }
    
    if "erreur" in donut_result:
        log_info("Donut a échoué, utilisation d'OCR+Mistral uniquement")
        return {
            "résultat_final": ocr_result,
            "notes_confiance": {k: 0.7 for k in ocr_result.keys() if k != "erreur"},
            "confiance_globale": 0.6
        }
    
    # Chargement du modèle Mistral pour la fusion
    log_info("Chargement du modèle Mistral pour la fusion...")
    mistral_model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Version v0.1 open source

    # S'assurer que la mémoire est libre
    free_gpu_memory()
    log_gpu_memory()

    try:
        log_info(f"Chargement tokenizer pour fusion: {time.strftime('%H:%M:%S')}")
        fusion_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
        log_info("Tokenizer chargé avec succès")
        
        log_info(f"Chargement modèle pour fusion: {time.strftime('%H:%M:%S')}")
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # Configuration pour quantification 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,              # Quantification 4-bit pour économiser mémoire
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        fusion_model = AutoModelForCausalLM.from_pretrained(
            mistral_model_name,
            quantization_config=bnb_config,  # Utiliser quantification 4-bit
            device_map="auto"                # Laisser transformers optimiser
        )
        log_info("Modèle pour fusion chargé avec succès")
        
        # Génération du prompt et inférence
        log_info("Génération du prompt pour fusion...")
        prompt = generate_prompt_fusion(ocr_result, donut_result)
        
        log_info("Début inférence fusion...")
        try:
            # Encodage
            inputs = fusion_tokenizer(prompt, return_tensors="pt").to("cuda")
            
            # Génération avec optimisation mémoire
            with torch.cuda.amp.autocast():
                outputs = fusion_model.generate(
                    inputs.input_ids,
                    max_new_tokens=3072,  # Plus de tokens pour fusion
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=fusion_tokenizer.eos_token_id,
                    eos_token_id=fusion_tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Décodage et parsing du résultat
            output_text = fusion_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Libération mémoire
            del outputs, inputs
            free_gpu_memory()
            
            # Parsing sécurisé du JSON
            result = parse_json_safely(output_text)
            log_info("Fusion terminée avec succès")
            
            # Si le parsing a échoué, on fait une fusion manuelle simple
            if "erreur" in result:
                log_info("Fusion avec LLM échouée, utilisation d'une fusion simple")
                # Fusion manuelle simple
                fusion_simple = {
                    "résultat_final": {},
                    "notes_confiance": {},
                    "confiance_globale": 0.5
                }
                
                # Fusion simple des champs
                for key in set(list(ocr_result.keys()) + list(donut_result.keys())):
                    if key in ocr_result and key in donut_result:
                        # Si les deux ont le champ, on prend celui d'OCR par défaut
                        fusion_simple["résultat_final"][key] = ocr_result[key]
                        fusion_simple["notes_confiance"][key] = 0.7
                    elif key in ocr_result:
                        fusion_simple["résultat_final"][key] = ocr_result[key]
                        fusion_simple["notes_confiance"][key] = 0.6
                    else:
                        fusion_simple["résultat_final"][key] = donut_result[key]
                        fusion_simple["notes_confiance"][key] = 0.5
                
                result = fusion_simple
            
            # Libération du modèle et tokenizer
            del fusion_model, fusion_tokenizer
            free_gpu_memory()
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            log_info("ERREUR: Mémoire GPU insuffisante pour la fusion")
            # Fusion simple en cas d'erreur mémoire
            fusion_simple = {
                "résultat_final": ocr_result,  # On privilégie OCR+LLM qui est généralement plus précis
                "notes_confiance": {k: 0.6 for k in ocr_result.keys()},
                "confiance_globale": 0.6
            }
            return fusion_simple
            
        except Exception as e:
            log_info(f"Erreur lors de la fusion: {str(e)}")
            # Fusion simple en cas d'erreur
            fusion_simple = {
                "résultat_final": ocr_result,
                "notes_confiance": {k: 0.5 for k in ocr_result.keys()},
                "confiance_globale": 0.5
            }
            return fusion_simple
    
    except Exception as e:
        log_info(f"Erreur lors du chargement du modèle pour fusion: {str(e)}")
        # Fusion simple en cas d'erreur de chargement
        fusion_simple = {
            "résultat_final": ocr_result,
            "notes_confiance": {k: 0.5 for k in ocr_result.keys()},
            "confiance_globale": 0.5
        }
        return fusion_simple
    
    finally:
        # S'assurer que la mémoire est bien libérée
        free_gpu_memory()

########################################
# PARTIE 5: Pipeline principal         #
########################################

def process_invoice(invoice_path):
    log_info(f"=== TRAITEMENT DE {os.path.basename(invoice_path)} ===")
    
    try:
        # ÉTAPE 1: OCR avec Tesseract
        log_info("ÉTAPE 1: OCR avec Tesseract")
        raw_text = ocr_image(invoice_path)
        if not raw_text:
            return {"erreur": "Échec de l'OCR"}
        clean_ocr = clean_text(raw_text)
        log_info(f"OCR réussi: {len(clean_ocr)} caractères")
        
        # ÉTAPE 2: Extraction avec Mistral sur OCR
        log_info("ÉTAPE 2: Extraction avec Mistral")
        ocr_llm_result = extract_with_mistral(clean_ocr)
        log_info("Extraction Mistral terminée")
        
        # Sauvegarde résultat intermédiaire OCR+Mistral
        ocr_result_path = os.path.join(ocr_results_dir, os.path.splitext(os.path.basename(invoice_path))[0] + '_ocr.json')
        with open(ocr_result_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_llm_result, f, ensure_ascii=False, indent=4)
        log_info(f"Résultat OCR+Mistral sauvegardé: {ocr_result_path}")
        
        # ÉTAPE 3: Extraction avec Donut sur image
        log_info("ÉTAPE 3: Extraction avec Donut")
        donut_result = extract_with_donut(invoice_path)
        log_info("Extraction Donut terminée")
        
        # Sauvegarde résultat intermédiaire Donut
        donut_result_path = os.path.join(donut_results_dir, os.path.splitext(os.path.basename(invoice_path))[0] + '_donut.json')
        with open(donut_result_path, 'w', encoding='utf-8') as f:
            json.dump(donut_result, f, ensure_ascii=False, indent=4)
        log_info(f"Résultat Donut sauvegardé: {donut_result_path}")
        
        # ÉTAPE 4: Fusion des résultats
        log_info("ÉTAPE 4: Fusion des résultats")
        final_result = fusion_resultats(ocr_llm_result, donut_result)
        log_info("Fusion terminée")
        
        return final_result
        
    except Exception as e:
        log_info(f"Erreur lors du traitement de la facture: {str(e)}")
        return {
            "résultat_final": {"erreur": str(e)},
            "notes_confiance": {"erreur": 1.0},
            "confiance_globale": 0.0
        }

# Fonction principale
def main():
    log_info("=== DÉBUT DU TRAITEMENT DES FACTURES ===")
    
    # Traitement des factures
    for filename in os.listdir(invoice_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            invoice_path = os.path.join(invoice_dir, filename)
            
            try:
                # Traitement de la facture
                final_result = process_invoice(invoice_path)
                
                # Sauvegarde du résultat final
                result_filename = os.path.splitext(filename)[0] + '.json'
                result_path = os.path.join(final_results_dir, result_filename)
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(final_result, f, ensure_ascii=False, indent=4)
                log_info(f"Résultat final sauvegardé: {result_path}")
                
            except Exception as e:
                log_info(f"Erreur fatale lors du traitement de {filename}: {str(e)}")
    
    log_info("=== TRAITEMENT TERMINÉ ===")

# Point d'entrée
if __name__ == "__main__":
    main() 