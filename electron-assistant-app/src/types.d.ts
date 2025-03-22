// Ce fichier doit être traité comme un module
export {};

// Déclarations globales
declare global {
  interface Window {
    // Propriétés pour la fenêtre si nécessaire
  }
}

// Déclarations pour NodeJS
declare namespace NodeJS {
  interface Process {
    readonly platform: string;
  }
}

// Variables globales
declare var process: NodeJS.Process;
declare var __dirname: string; 