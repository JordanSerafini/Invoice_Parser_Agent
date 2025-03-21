import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.enableCors(); // Activer CORS pour permettre les requêtes depuis le frontend
  await app.listen(3000);
  console.log(
    `L'application est en cours d'exécution sur: http://localhost:3000`,
  );
}
// Ajouter void pour marquer explicitement que nous ignorons la promesse
void bootstrap();
