import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';
import * as Tesseract from 'tesseract.js';
import * as pdfParse from 'pdf-parse';
import { promises as fsPromises } from 'fs';

// Interfaces pour le typage
interface InvoiceArticle {
  description: string;
  quantité: number;
  prix_unitaire_ht: number;
  montant_ht: number;
}

interface InvoiceData {
  numéro_facture: string;
  date_facture: string;
  nom_fournisseur: string;
  adresse_fournisseur: string;
  articles: InvoiceArticle[];
  remise?: number;
  total_ht: number;
  total_tva: number;
  total_ttc: number;
  mode_règlement: string;
  date_échéance: string;
  iban: string;
  bic: string;
  [key: string]: any; // Pour permettre d'autres propriétés
}

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
   * Convertit une image en texte avec Tesseract OCR amélioré
   */
  private async extractTextFromImage(imagePath: string): Promise<string> {
    try {
      // Configuration améliorée de Tesseract
      const options = {};

      // On utilise une méthode alternative pour passer les options avancées
      // qui ne sont pas directement typées dans l'interface WorkerOptions
      (options as any).tessedit_char_whitelist =
        '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:€$£&@#%()<>=/\\|+-_*\'"`!? ';
      (options as any).tessjs_create_pdf = '0';
      (options as any).tessjs_create_hocr = '0';
      (options as any).classify_bln_numeric_mode = '1';

      const {
        data: { text },
      } = await Tesseract.recognize(imagePath, 'fra+eng', options);

      // Nettoyage de base du texte extrait
      const cleanedText = this.cleanOcrText(text);
      return cleanedText;
    } catch (error) {
      console.error("Erreur lors de l'extraction OCR:", error);
      // Tentative avec configuration minimale en cas d'erreur
      try {
        // Essai avec une langue différente
        const {
          data: { text },
        } = await Tesseract.recognize(imagePath, 'eng');
        return this.cleanOcrText(text);
      } catch (error) {
        console.error("Erreur lors de l'extraction OCR:", error);
        // Dernier recours, OCR basique
        const {
          data: { text },
        } = await Tesseract.recognize(imagePath);
        return this.cleanOcrText(text);
      }
    }
  }

  /**
   * Convertit un PDF en texte avec pdf-parse
   */
  private async extractTextFromPdf(pdfPath: string): Promise<string> {
    try {
      const pdfBuffer = fs.readFileSync(pdfPath);
      const options = {
        // Options pour améliorer l'extraction
        pagerender: this.customPageRenderer.bind(this),
        max: 0, // Pas de limite de pages
      };
      const data = await pdfParse(pdfBuffer, options);
      return this.cleanOcrText(data.text);
    } catch (error) {
      console.error("Erreur lors de l'extraction PDF:", error);
      // Fallback à l'extraction simple en cas d'erreur
      const pdfBuffer = fs.readFileSync(pdfPath);
      const data = await pdfParse(pdfBuffer);
      return this.cleanOcrText(data.text);
    }
  }

  /**
   * Rendu personnalisé pour l'extraction PDF
   */
  private customPageRenderer(pageData: any): string {
    // Récupérer le contenu texte
    const renderOptions = {
      normalizeWhitespace: true,
      disableCombineTextItems: false,
    };
    return pageData.getTextContent(renderOptions).then((textContent: any) => {
      let text = '';
      const items = textContent.items;
      let lastY = -1;
      for (const item of items) {
        if (lastY !== item.transform[5] && text) {
          text += '\n';
        }
        text += item.str;
        lastY = item.transform[5];
      }
      return text;
    });
  }

  /**
   * Nettoyage du texte OCR pour améliorer la qualité
   */
  private cleanOcrText(text: string): string {
    if (!text) return '';

    // Remplacer les caractères problématiques
    let cleanedText = text
      .replace(/\r\n/g, '\n')
      .replace(/\t/g, ' ')
      .replace(/\s+/g, ' ') // Supprimer les espaces multiples
      .replace(/(\d),(\d)/g, '$1.$2') // Convertir les virgules en points pour les nombres
      .replace(/[^\x20-\x7E\n\u00C0-\u00FF]/g, '') // Garder uniquement les caractères imprimables et accents
      .trim();

    // Corrections spécifiques pour les numéros de facture, dates, etc.
    cleanedText = this.correctCommonOcrMistakes(cleanedText);

    return cleanedText;
  }

  /**
   * Correction des erreurs OCR courantes
   */
  private correctCommonOcrMistakes(text: string): string {
    return (
      text
        // Correction des numéros (O->0, I->1, etc.)
        .replace(
          /Facture[:\s]+N[°o]?[:\s]*([A-Za-z0-9-_/]+)/gi,
          (match, invoiceNum) => {
            return (
              'Facture N°: ' + invoiceNum.replace(/O/g, '0').replace(/I/g, '1')
            );
          },
        )
        // Correction des dates
        .replace(
          /(\d{1,2})[/\\.-](\d{1,2})[/\\.-](\d{2,4})/g,
          (match, day, month, year) => {
            if (parseInt(month) > 12) {
              // Si le mois est > 12, il s'agit probablement du jour et du mois inversés
              [day, month] = [month, day];
            }
            if (year.length === 2) year = '20' + year; // Compléter l'année si besoin
            return `${day}/${month}/${year}`;
          },
        )
        // Correction des montants
        .replace(/(\d+)[.,](\d+)[ ]?[€eE]/g, '$1.$2 €')
    );
  }

  /**
   * Valide un IBAN
   */
  private isValidIBAN(iban: string): boolean {
    // Retirer les espaces et convertir en majuscules
    iban = iban.replace(/\s/g, '').toUpperCase();
    // Vérification basique du format
    return /^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$/.test(iban);
  }

  /**
   * Valide un BIC
   */
  private isValidBIC(bic: string): boolean {
    // Retirer les espaces et convertir en majuscules
    bic = bic.replace(/\s/g, '').toUpperCase();
    // Vérification basique du format
    return /^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$/.test(bic);
  }

  /**
   * Vérifier la cohérence des montants
   */
  private checkAmountsConsistency(data: InvoiceData): InvoiceData {
    const correctedData = { ...data };

    // Recalculer le total HT à partir des articles
    if (correctedData.articles && correctedData.articles.length > 0) {
      const calculatedTotalHT = correctedData.articles.reduce(
        (sum, article) => sum + article.montant_ht,
        0,
      );

      // Si une remise est présente, l'inclure dans le calcul
      const totalWithDiscount = correctedData.remise
        ? calculatedTotalHT + correctedData.remise
        : calculatedTotalHT;

      // Si l'écart est significatif, corriger le total
      if (Math.abs(totalWithDiscount - correctedData.total_ht) > 0.05) {
        console.log(
          `Incohérence détectée dans le total HT: calculé=${totalWithDiscount}, trouvé=${correctedData.total_ht}`,
        );
      }
    }

    // Vérifier la cohérence TVA + HT = TTC
    const calculatedTTC = correctedData.total_ht + correctedData.total_tva;
    if (Math.abs(calculatedTTC - correctedData.total_ttc) > 0.05) {
      console.log(
        `Incohérence détectée dans le total TTC: calculé=${calculatedTTC}, trouvé=${correctedData.total_ttc}`,
      );
    }

    return correctedData;
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
        // Nettoyage préalable du JSON
        const sanitizedJson = this.sanitizeJson(match);

        // Vérifier si c'est un JSON valide
        const parsedJson = JSON.parse(sanitizedJson) as InvoiceData;

        // Vérifier et corriger les données
        const validatedJson = this.validateAndCorrectData(parsedJson);
        return JSON.stringify(validatedJson);
      } catch (e) {
        console.log('Erreur de parsing JSON:', e.message);
        // Ce n'est pas un JSON valide, on continue
        continue;
      }
    }

    // Si aucun JSON valide n'a été trouvé, on essaie de réparer le premier match
    if (matches.length > 0) {
      try {
        // Essayer de nettoyer le JSON pour le rendre valide
        const cleanedJson = this.cleanAndRepairJson(matches[0]);

        // Vérifier et valider les données
        const parsedJson = JSON.parse(cleanedJson) as InvoiceData;
        const validatedJson = this.validateAndCorrectData(parsedJson);
        return JSON.stringify(validatedJson);
      } catch (e) {
        console.log('Erreur de réparation JSON:', e.message);
        throw new Error(`Impossible de réparer le JSON: ${e.message}`);
      }
    }

    throw new Error('Aucun JSON valide trouvé dans la réponse');
  }

  /**
   * Nettoie et sanitize une chaîne JSON
   */
  private sanitizeJson(text: string): string {
    let jsonText = text.trim();

    // Remplacement des virgules par des points pour les nombres
    jsonText = jsonText.replace(/(\d),(\d)/g, '$1.$2');

    // Supprimer les caractères qui ne sont pas valides en JSON
    jsonText = jsonText.replace(/[\n\r\t\b\f\v]/g, ' ');

    // Réparer les quotes mal formées
    jsonText = jsonText.replace(/(['"])(.*?)['"](?=\s*[:,\]}])/g, '"$2"');

    // Assurer que les clés JSON ont des doubles quotes
    jsonText = jsonText.replace(
      /([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)/g,
      '$1"$2"$3',
    );

    return jsonText;
  }

  /**
   * Valider et corriger les données extraites
   */
  private validateAndCorrectData(data: InvoiceData): InvoiceData {
    const correctedData = { ...data };

    // Corriger le numéro de facture
    if (correctedData.numéro_facture) {
      correctedData.numéro_facture = correctedData.numéro_facture
        .replace(/O/g, '0')
        .replace(/I/g, '1')
        .replace(/\s+/g, '');
    }

    // Vérifier et corriger l'IBAN
    if (correctedData.iban && !this.isValidIBAN(correctedData.iban)) {
      // Si l'IBAN est invalide, tenter de corriger les erreurs communes
      const cleanedIban = correctedData.iban
        .replace(/\s/g, '')
        .toUpperCase()
        .replace(/O/g, '0')
        .replace(/I/g, '1');

      if (this.isValidIBAN(cleanedIban)) {
        correctedData.iban = cleanedIban;
      } else {
        console.log(`IBAN invalide: ${correctedData.iban}`);
      }
    }

    // Vérifier et corriger le BIC
    if (correctedData.bic && !this.isValidBIC(correctedData.bic)) {
      // Si le BIC est invalide, tenter de corriger les erreurs communes
      const cleanedBic = correctedData.bic
        .replace(/\s/g, '')
        .toUpperCase()
        .replace(/O/g, '0')
        .replace(/I/g, '1');

      if (this.isValidBIC(cleanedBic)) {
        correctedData.bic = cleanedBic;
      } else {
        console.log(`BIC invalide: ${correctedData.bic}`);
      }
    }

    // Vérifier la cohérence des articles
    if (correctedData.articles && Array.isArray(correctedData.articles)) {
      for (const article of correctedData.articles) {
        // Vérifier si le montant HT correspond au prix unitaire × quantité
        if (article.quantité && article.prix_unitaire_ht) {
          const calculatedAmount = article.quantité * article.prix_unitaire_ht;
          if (Math.abs(calculatedAmount - article.montant_ht) > 0.01) {
            console.log(
              `Incohérence dans le montant HT de l'article "${article.description}": calculé=${calculatedAmount}, trouvé=${article.montant_ht}`,
            );
          }
        }
      }
    }

    // Vérifier la cohérence des montants HT, TVA, TTC
    return this.checkAmountsConsistency(correctedData);
  }

  /**
   * Tente de nettoyer et réparer un JSON malformé
   */
  private cleanAndRepairJson(brokenJson: string): string {
    // Sanitizer d'abord le JSON
    const sanitizedJson = this.sanitizeJson(brokenJson);

    // Créer un objet JSON vide pour la structure
    const template: InvoiceData = {
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
      const match = sanitizedJson.match(regex);
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
    const totalHtMatch = sanitizedJson.match(/"total_ht"\s*:\s*([\d.-]+)/);
    if (totalHtMatch) template.total_ht = parseFloat(totalHtMatch[1]);

    const totalTvaMatch = sanitizedJson.match(/"total_tva"\s*:\s*([\d.-]+)/);
    if (totalTvaMatch) template.total_tva = parseFloat(totalTvaMatch[1]);

    const totalTtcMatch = sanitizedJson.match(/"total_ttc"\s*:\s*([\d.-]+)/);
    if (totalTtcMatch) template.total_ttc = parseFloat(totalTtcMatch[1]);

    // Extraire la remise si elle existe
    const remiseMatch = sanitizedJson.match(/"remise"\s*:\s*([\d.-]+)/);
    if (remiseMatch) template.remise = parseFloat(remiseMatch[1]);

    // Essayer d'extraire les articles (plus complexe)
    try {
      const articlesMatch = sanitizedJson.match(/"articles"\s*:\s*(\[.*?\])/s);
      if (articlesMatch) {
        const articlesJson = articlesMatch[1];
        try {
          const parsedArticles = JSON.parse(articlesJson) as InvoiceArticle[];
          template.articles = parsedArticles;
        } catch (e) {
          console.log('Erreur de parsing JSON des articles:', e.message);
          // Si impossible de parser les articles, laisser un tableau vide
        }
      } else {
        // Essayer une approche alternative pour extraire les articles en utilisant des patterns
        const articleLines = sanitizedJson.match(/description.*?montant_ht/gs);
        if (articleLines) {
          // Extraire les articles ligne par ligne
          const extractedArticles: InvoiceArticle[] = [];
          for (const line of articleLines) {
            try {
              const description =
                line.match(/"description"\s*:\s*"([^"]*)"/)?.[1] || '';
              const quantité = parseFloat(
                line.match(/"quantité"\s*:\s*([\d.]+)/)?.[1] || '1',
              );
              const prixUnitaire = parseFloat(
                line.match(/"prix_unitaire_ht"\s*:\s*([\d.]+)/)?.[1] || '0',
              );
              const montant = parseFloat(
                line.match(/"montant_ht"\s*:\s*([\d.]+)/)?.[1] || '0',
              );

              if (description) {
                extractedArticles.push({
                  description,
                  quantité,
                  prix_unitaire_ht: prixUnitaire,
                  montant_ht: montant,
                });
              }
            } catch (lineError) {
              console.log('Erreur extraction article:', lineError.message);
            }
          }
          if (extractedArticles.length > 0) {
            template.articles = extractedArticles;
          }
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

    // Pré-traitement du texte pour améliorer l'extraction
    const cleanedText = this.cleanOcrText(text);

    const prompt = `<s>[INST] Tu es un expert en extraction d'informations à partir de factures. Je vais te fournir le texte extrait d'une facture par OCR, et tu vas m'extraire les données importantes dans un format JSON valide.

Texte de la facture:
${cleanedText}

Tu dois extraire exactement ces informations dans ce format JSON strict (sans aucun texte additionnel):
{
  "numéro_facture": "chaîne (ex: FC20250297)",
  "date_facture": "JJ/MM/AAAA",
  "nom_fournisseur": "chaîne exacte",
  "adresse_fournisseur": "chaîne exacte complète",
  "articles": [
    {
      "description": "chaîne exacte",
      "quantité": nombre entier,
      "prix_unitaire_ht": nombre décimal avec point (ex: 120.50),
      "montant_ht": nombre décimal avec point
    }
  ],
  "remise": nombre décimal négatif si présent (sinon 0),
  "total_ht": nombre décimal avec point,
  "total_tva": nombre décimal avec point,
  "total_ttc": nombre décimal avec point,
  "mode_règlement": "chaîne exacte",
  "date_échéance": "JJ/MM/AAAA",
  "iban": "format IBAN exact",
  "bic": "format BIC exact"
}

RÈGLES STRICTES:
1. Fais attention aux valeurs numériques, assure-toi qu'elles sont correctes.
2. Tous les nombres décimaux doivent être formatés avec un point (.) et jamais de virgule (,).
3. Si un champ est absent, mets une chaîne vide "" ou 0 pour les nombres.
4. La remise doit être une valeur négative (ex: -30.00) si présente, sinon 0.
5. Vérifie que le total_ht correspond à la somme des montants HT des articles (+ remise).
6. Vérifie que total_ht + total_tva = total_ttc.
7. Assure-toi que ton JSON est valide et respecte exactement le format demandé.
8. NE METS ABSOLUMENT AUCUN TEXTE AVANT OU APRÈS LE JSON. [/INST]</s>`;

    try {
      // Ajouter un timeout et un mécanisme de retry
      const axiosConfig = {
        headers: {
          Authorization: `Bearer ${HUGGING_FACE_TOKEN}`,
          'Content-Type': 'application/json',
        },
        timeout: 30000, // 30 secondes de timeout
      };

      const requestBody = {
        inputs: prompt,
        parameters: {
          max_new_tokens: 1024,
          temperature: 0.1,
          return_full_text: false,
        },
      };

      // Tentative avec retry
      let response;
      let retryCount = 0;
      const maxRetries = 2;

      while (retryCount <= maxRetries) {
        try {
          response = await axios.post(HF_API_URL, requestBody, axiosConfig);
          break; // Si la requête réussit, sortir de la boucle
        } catch (error) {
          retryCount++;
          if (retryCount > maxRetries) throw error;
          // Attendre avant de réessayer (backoff exponentiel)
          await new Promise((resolve) =>
            setTimeout(resolve, 1000 * retryCount),
          );
        }
      }

      // Obtenir la réponse brute
      const rawResponse = response.data[0]?.generated_text || '';
      console.log('Réponse brute de Hugging Face:', rawResponse);

      try {
        // D'abord, essayer de parser directement la réponse
        const sanitizedResponse = this.sanitizeJson(rawResponse);
        const parsedJson = JSON.parse(sanitizedResponse) as InvoiceData;
        const validatedJson = this.validateAndCorrectData(parsedJson);
        return JSON.stringify(validatedJson);
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
              const parsedJson = JSON.parse(cleanedJson) as InvoiceData;
              const validatedJson = this.validateAndCorrectData(parsedJson);
              return JSON.stringify(validatedJson);
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
