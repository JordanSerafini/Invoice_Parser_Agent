import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
  BadRequestException,
  Get,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { OcrService } from './ocr.service';
import * as fs from 'fs';
import * as path from 'path';

// Types pour les résultats d'analyse
interface InvoiceAnalysisResult {
  fileName: string;
  success: boolean;
  resultFile?: string;
  data?: any;
  error?: string;
}

@Controller('ocr')
export class OcrController {
  constructor(private ocrService: OcrService) {}

  @Post('analyze')
  @UseInterceptors(FileInterceptor('file'))
  async analyzeInvoice(@UploadedFile() file: Express.Multer.File) {
    if (!file) {
      throw new BadRequestException('Aucun fichier trouvé');
    }

    const allowedTypes = [
      'application/pdf',
      'image/png',
      'image/jpeg',
      'image/jpg',
    ];
    if (!allowedTypes.includes(file.mimetype)) {
      throw new BadRequestException(
        'Format non supporté. Utilisez PDF ou image (PNG, JPG)',
      );
    }

    try {
      // Sauvegarder le fichier temporairement
      const filePath = await this.ocrService.saveTemporaryFile(file);

      // Extraire le texte selon le type de fichier
      const extractedText = await this.ocrService.extractTextFromFile(
        filePath,
        file.mimetype,
      );

      if (!extractedText.trim()) {
        this.ocrService.cleanupTemporaryFile(filePath);
        throw new BadRequestException(
          "Aucun texte n'a été extrait du document",
        );
      }

      // Analyser avec Mistral AI
      const analysisResult =
        await this.ocrService.analyzeWithMistral(extractedText);

      // Nettoyer le fichier temporaire
      this.ocrService.cleanupTemporaryFile(filePath);

      return JSON.parse(analysisResult);
    } catch (error) {
      throw new BadRequestException(
        `Erreur lors de l'analyse: ${error.message}`,
      );
    }
  }

  @Get('analyze-all-invoices')
  async analyzeAllInvoices() {
    try {
      const facturesDir = path.join(process.cwd(), 'src', 'Factures');

      if (!fs.existsSync(facturesDir)) {
        throw new BadRequestException('Dossier Factures introuvable');
      }

      const files = fs.readdirSync(facturesDir);
      const pngFiles = files.filter((file) =>
        file.toLowerCase().endsWith('.png'),
      );

      if (pngFiles.length === 0) {
        throw new BadRequestException(
          'Aucun fichier PNG trouvé dans le dossier Factures',
        );
      }

      const results: InvoiceAnalysisResult[] = [];

      for (const pngFile of pngFiles) {
        try {
          const filePath = path.join(facturesDir, pngFile);

          // Extraire le texte de l'image
          const extractedText = await this.ocrService.extractTextFromFile(
            filePath,
            'image/png',
          );

          if (!extractedText.trim()) {
            results.push({
              fileName: pngFile,
              success: false,
              error: "Aucun texte n'a été extrait du document",
            });
            continue;
          }

          // Analyser avec Hugging Face
          const analysisResult =
            await this.ocrService.analyzeWithMistral(extractedText);

          // Sauvegarder le résultat JSON
          const resultFileName = pngFile.replace('.png', '.json');
          const resultFilePath = path.join(facturesDir, resultFileName);
          fs.writeFileSync(resultFilePath, analysisResult);

          results.push({
            fileName: pngFile,
            success: true,
            resultFile: resultFileName,
            data: JSON.parse(analysisResult),
          });
        } catch (error) {
          results.push({
            fileName: pngFile,
            success: false,
            error: error.message,
          });
        }
      }

      return {
        totalProcessed: pngFiles.length,
        successCount: results.filter((r) => r.success).length,
        failureCount: results.filter((r) => !r.success).length,
        results,
      };
    } catch (error) {
      throw new BadRequestException(
        `Erreur lors de l'analyse des factures: ${error.message}`,
      );
    }
  }
}
