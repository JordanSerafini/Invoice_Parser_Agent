import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';
import * as Tesseract from 'tesseract.js';
import * as pdfParse from 'pdf-parse';
import { promises as fsPromises } from 'fs';

@Injectable()
export class OcrService {
  constructor(private configService: ConfigService) {}

  /**
   * Sauvegarde le fichier temporairement et renvoie le chemin
   */
  async saveTemporaryFile(file: Express.Multer.File): Promise<string> {
    const tempDir = path.join(process.cwd(), 'temp');
    if (!fs.existsSync(tempDir)) {
      await fsPromises.mkdir(tempDir, { recursive: true });
    }

    const filePath = path.join(tempDir, file.originalname);
    await fsPromises.writeFile(filePath, file.buffer);
    return filePath;
  }

  /**
   * Nettoie le fichier temporaire
   */
  cleanupTemporaryFile(filePath: string): void {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  }

  /**
   * Extraire le texte d'un fichier PDF ou image
   */
  async extractTextFromFile(
    filePath: string,
    mimeType: string,
  ): Promise<string> {
    if (mimeType === 'application/pdf') {
      return this.extractTextFromPdf(filePath);
    } else {
      return this.extractTextFromImage(filePath);
    }
  }

  /**
   * Convertit une image en texte avec Tesseract OCR
   */
  private async extractTextFromImage(imagePath: string): Promise<string> {
    const {
      data: { text },
    } = await Tesseract.recognize(imagePath, 'fra+eng');
    return text;
  }

  /**
   * Convertit un PDF en texte avec pdf-parse
   */
  private async extractTextFromPdf(pdfPath: string): Promise<string> {
    const pdfBuffer = fs.readFileSync(pdfPath);
    const data = await pdfParse(pdfBuffer);
    return data.text;
  }

  /**
   * Extrait le JSON d'une chaîne de texte
   */
  private extractJsonFromText(text: string): string {
    // Recherche toutes les occurrences potentielles de JSON
    const jsonRegex = /{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*}/g;
    const matches = text.match(jsonRegex);

    if (!matches || matches.length === 0) {
      throw new Error('Aucun JSON trouvé dans la réponse');
    }

    // Essaie de parser chaque match pour trouver un JSON valide
    for (const match of matches) {
      try {
        // Vérifier si c'est un JSON valide en le parsant
        JSON.parse(match);
        // Si on arrive ici, c'est un JSON valide
        return match;
      } catch (e) {
        console.log('Erreur de parsing JSON:', e.message);
        // Ce n'est pas un JSON valide, on continue
        continue;
      }
    }

    // Si aucun JSON valide n'a été trouvé, on essaie de corriger le premier match
    if (matches.length > 0) {
      try {
        // Essayer de nettoyer le JSON pour le rendre valide
        const cleanedJson = this.cleanAndRepairJson(matches[0]);
        return cleanedJson;
      } catch (e) {
        console.log('Erreur de réparation JSON:', e.message);
        throw new Error(`Impossible de réparer le JSON: ${e.message}`);
      }
    }

    throw new Error('Aucun JSON valide trouvé dans la réponse');
  }

  /**
   * Tente de nettoyer et réparer un JSON malformé
   */
  private cleanAndRepairJson(brokenJson: string): string {
    // Créer un objet JSON vide pour la structure
    const template = {
      numéro_facture: '',
      date_facture: '',
      nom_fournisseur: '',
      adresse_fournisseur: '',
      articles: [],
      total_ht: 0,
      total_tva: 0,
      total_ttc: 0,
      mode_règlement: '',
      date_échéance: '',
      iban: '',
      bic: '',
    };

    // Extraire les valeurs avec des expressions régulières
    const extractValue = (key: string, regex: RegExp): string => {
      const match = brokenJson.match(regex);
      return match ? match[1] : '';
    };

    // Essayer d'extraire chaque champ individuellement
    template.numéro_facture = extractValue(
      'numéro_facture',
      /"numéro_facture"\s*:\s*"([^"]*)"/,
    );
    template.date_facture = extractValue(
      'date_facture',
      /"date_facture"\s*:\s*"([^"]*)"/,
    );
    template.nom_fournisseur = extractValue(
      'nom_fournisseur',
      /"nom_fournisseur"\s*:\s*"([^"]*)"/,
    );
    template.adresse_fournisseur = extractValue(
      'adresse_fournisseur',
      /"adresse_fournisseur"\s*:\s*"([^"]*)"/,
    );
    template.mode_règlement = extractValue(
      'mode_règlement',
      /"mode_règlement"\s*:\s*"([^"]*)"/,
    );
    template.date_échéance = extractValue(
      'date_échéance',
      /"date_échéance"\s*:\s*"([^"]*)"/,
    );
    template.iban = extractValue('iban', /"iban"\s*:\s*"([^"]*)"/);
    template.bic = extractValue('bic', /"bic"\s*:\s*"([^"]*)"/);

    // Extraire les totaux (nombres)
    const totalHtMatch = brokenJson.match(/"total_ht"\s*:\s*([\d.]+)/);
    if (totalHtMatch) template.total_ht = parseFloat(totalHtMatch[1]);

    const totalTvaMatch = brokenJson.match(/"total_tva"\s*:\s*([\d.]+)/);
    if (totalTvaMatch) template.total_tva = parseFloat(totalTvaMatch[1]);

    const totalTtcMatch = brokenJson.match(/"total_ttc"\s*:\s*([\d.]+)/);
    if (totalTtcMatch) template.total_ttc = parseFloat(totalTtcMatch[1]);

    // Essayer d'extraire les articles (plus complexe)
    try {
      const articlesMatch = brokenJson.match(/"articles"\s*:\s*(\[.*?\])/s);
      if (articlesMatch) {
        const articlesJson = articlesMatch[1];
        try {
          template.articles = JSON.parse(articlesJson);
        } catch (e) {
          console.log('Erreur de parsing JSON des articles:', e.message);
          // Si impossible de parser les articles, laisser un tableau vide
        }
      }
    } catch (e) {
      console.log('Erreur de parsing JSON des articles:', e.message);
      // Ignorer les erreurs, laisser un tableau vide
    }

    // Convertir en JSON valide
    return JSON.stringify(template);
  }

  /**
   * Analyse le texte avec Hugging Face pour extraire les données structurées
   */
  async analyzeWithMistral(text: string): Promise<string> {
    const HUGGING_FACE_TOKEN =
      this.configService.get<string>('HUGGING_FACE_TOKEN');
    const HF_API_URL =
      'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2';

    if (!HUGGING_FACE_TOKEN) {
      throw new Error("Le token Hugging Face n'est pas configuré");
    }

    const prompt = `<s>[INST] Tu es un expert en extraction d'informations à partir de factures. Je vais te fournir le texte extrait d'une facture, et tu vas m'extraire les données importantes dans un format JSON valide.

Texte de la facture:
${text}

Tu dois extraire ces informations et me les fournir exactement dans ce format JSON (sans aucun texte additionnel):
{
  "numéro_facture": "XXX",
  "date_facture": "XXX",
  "nom_fournisseur": "XXX",
  "adresse_fournisseur": "XXX",
  "articles": [
    {
      "description": "XXX",
      "quantité": X,
      "prix_unitaire_ht": X.XX,
      "montant_ht": X.XX
    }
  ],
  "total_ht": X.XX,
  "total_tva": X.XX,
  "total_ttc": X.XX,
  "mode_règlement": "XXX",
  "date_échéance": "XXX",
  "iban": "XXX",
  "bic": "XXX"
}

Assure-toi de suivre précisément ce format et que ton JSON soit valide. Ne mets absolument aucun texte avant ou après le JSON. [/INST]</s>`;

    try {
      const response = await axios.post(
        HF_API_URL,
        {
          inputs: prompt,
          parameters: {
            max_new_tokens: 1024,
            temperature: 0.1,
            return_full_text: false,
          },
        },
        {
          headers: {
            Authorization: `Bearer ${HUGGING_FACE_TOKEN}`,
            'Content-Type': 'application/json',
          },
        },
      );

      // Obtenir la réponse brute
      const rawResponse = response.data[0]?.generated_text || '';
      console.log('Réponse brute de Hugging Face:', rawResponse);

      try {
        // D'abord, essayer de parser directement la réponse
        return JSON.stringify(JSON.parse(rawResponse));
      } catch (jsonParseError) {
        console.log('Erreur de parsing JSON direct:', jsonParseError.message);

        // Si le parsing direct échoue, essayer d'extraire un JSON valide
        try {
          const extractedJson = this.extractJsonFromText(rawResponse);
          return extractedJson;
        } catch (extractError) {
          console.log("Erreur d'extraction JSON:", extractError.message);

          // Si l'extraction échoue également, essayer une approche plus simple
          try {
            // Rechercher le début et la fin du JSON
            const jsonStartIndex = rawResponse.indexOf('{');
            const jsonEndIndex = rawResponse.lastIndexOf('}') + 1;

            if (jsonStartIndex !== -1 && jsonEndIndex > jsonStartIndex) {
              const jsonStr = rawResponse.substring(
                jsonStartIndex,
                jsonEndIndex,
              );

              // Essayer de nettoyer et réparer le JSON
              const cleanedJson = this.cleanAndRepairJson(jsonStr);
              return cleanedJson;
            } else {
              throw new Error('Impossible de trouver un JSON dans la réponse');
            }
          } catch (repairError) {
            console.log('Erreur de réparation JSON:', repairError.message);
            // En dernier recours, créer un JSON minimal avec un message d'erreur
            const fallbackJson = JSON.stringify({
              error: "Impossible d'extraire un JSON valide de la réponse",
              rawText:
                rawResponse.substring(0, 500) +
                (rawResponse.length > 500 ? '...' : ''),
            });
            return fallbackJson;
          }
        }
      }
    } catch (error) {
      throw new Error(
        `Erreur lors de l'analyse avec Hugging Face: ${
          error.response?.data?.error || error.message
        }`,
      );
    }
  }
}
