import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { OcrController } from './Ocr/ocr.controller';
import { OcrService } from './Ocr/ocr.service';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
    }),
  ],
  controllers: [OcrController],
  providers: [OcrService],
})
export class AppModule {}
