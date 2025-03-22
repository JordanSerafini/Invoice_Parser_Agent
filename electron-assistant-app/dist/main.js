"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const path = __importStar(require("path"));
// Désactiver complètement l'accélération matérielle
electron_1.app.disableHardwareAcceleration();
// Configuration pour utiliser le rendu logiciel uniquement
electron_1.app.commandLine.appendSwitch('disable-gpu');
electron_1.app.commandLine.appendSwitch('disable-software-rasterizer');
electron_1.app.commandLine.appendSwitch('in-process-gpu');
let mainWindow = null;
function createWindow() {
    // Obtenir les dimensions de l'écran
    const { width, height } = electron_1.screen.getPrimaryDisplay().workAreaSize;
    // Dimensions de notre assistant
    const windowWidth = 300;
    const windowHeight = 400;
    // Calculer la position pour qu'il apparaisse en bas à droite
    const xPosition = width - windowWidth - 20; // 20px de marge
    const yPosition = height - windowHeight - 20; // 20px de marge
    // Créer la fenêtre
    mainWindow = new electron_1.BrowserWindow({
        width: windowWidth,
        height: windowHeight,
        x: xPosition,
        y: yPosition,
        frame: false, // Sans bordure
        transparent: true, // Activer la transparence pour éviter les coins blancs
        backgroundColor: '#00000000', // Fond transparent
        alwaysOnTop: true, // Toujours au-dessus
        resizable: false, // Non redimensionnable
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
// Écouteur d'événement IPC pour fermer la fenêtre
electron_1.ipcMain.on('close-window', () => {
    if (mainWindow) {
        mainWindow.close();
    }
});
// Quand Electron est prêt
electron_1.app.whenReady().then(() => {
    createWindow();
    electron_1.app.on('activate', () => {
        // Sur macOS, c'est courant de recréer une fenêtre quand 
        // l'icône du dock est cliquée et qu'aucune autre fenêtre n'est ouverte
        if (mainWindow === null)
            createWindow();
    });
});
// Quitter quand toutes les fenêtres sont fermées, sauf sur macOS
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin')
        electron_1.app.quit();
});
//# sourceMappingURL=main.js.map