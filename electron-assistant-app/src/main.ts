import { app, BrowserWindow, screen } from 'electron';
import * as path from 'path';

// Désactiver complètement l'accélération matérielle
app.disableHardwareAcceleration();

// Configuration pour utiliser le rendu logiciel uniquement
app.commandLine.appendSwitch('disable-gpu');
app.commandLine.appendSwitch('disable-software-rasterizer');
app.commandLine.appendSwitch('in-process-gpu');

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  // Obtenir les dimensions de l'écran
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  
  // Dimensions de notre assistant
  const windowWidth = 300;
  const windowHeight = 400;
  
  // Calculer la position pour qu'il apparaisse en bas à droite
  const xPosition = width - windowWidth - 20; // 20px de marge
  const yPosition = height - windowHeight - 20; // 20px de marge

  // Créer la fenêtre
  mainWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    x: xPosition,
    y: yPosition,
    frame: false,        // Sans bordure
    transparent: true,   // Activer la transparence pour éviter les coins blancs
    backgroundColor: '#00000000', // Fond transparent
    alwaysOnTop: true,   // Toujours au-dessus
    resizable: false,    // Non redimensionnable
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // Charger l'URL de l'application - ajustement du chemin pour trouver le fichier HTML
  const htmlPath = path.join(__dirname, '../src/renderer/index.html');
  mainWindow.loadFile(htmlPath);
  
  // Pour le débogage
  console.log('Chemin du fichier HTML:', htmlPath);

  // Gestion de la fermeture
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Quand Electron est prêt
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    // Sur macOS, c'est courant de recréer une fenêtre quand 
    // l'icône du dock est cliquée et qu'aucune autre fenêtre n'est ouverte
    if (mainWindow === null) createWindow();
  });
});

// Quitter quand toutes les fenêtres sont fermées, sauf sur macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
}); 