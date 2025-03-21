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

    const prompt = `<s>[INST] Voici le texte extrait d'une facture. Tu es un assistant spécialisé dans l'extraction de données de factures.

Texte de la facture:
${text}

Je veux que tu extraies les informations suivantes et que tu réponds UNIQUEMENT au format JSON :
- Numéro de facture
- Date de la facture
- Nom du fournisseur
- Adresse du fournisseur
- Liste des articles avec description, quantité, prix unitaire HT et montant HT
- Total HT
- Total TVA
- Total TTC
- Mode de règlement
- Date d'échéance
- IBAN
- BIC

Réponds uniquement avec un objet JSON valide. [/INST]</s>`;

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

      // Extraction du JSON depuis la réponse
      const rawResponse = response.data[0]?.generated_text || '';

      // Recherche du premier "{" et du dernier "}" pour extraire le JSON
      const jsonStartIndex = rawResponse.indexOf('{');
      const jsonEndIndex = rawResponse.lastIndexOf('}') + 1;

      if (jsonStartIndex === -1 || jsonEndIndex === 0) {
        throw new Error('Format de réponse invalide - JSON non trouvé');
      }

      const jsonStr = rawResponse.substring(jsonStartIndex, jsonEndIndex);

      // Vérification que le JSON est valide
      try {
        JSON.parse(jsonStr);
        return jsonStr;
      } catch (jsonError) {
        throw new Error(`JSON invalide retourné: ${jsonError.message}`);
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
