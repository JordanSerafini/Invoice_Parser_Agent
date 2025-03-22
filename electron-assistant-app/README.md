# Assistant Flottant Electron

Une application desktop simple avec Electron en TypeScript qui affiche un assistant flottant dans le coin en bas à droite de l'écran.

## Fonctionnalités

- Fenêtre toujours au-dessus des autres applications
- Interface transparente et sans bordure
- Possibilité de déplacer la fenêtre en la faisant glisser
- Animation d'apparition
- S'affiche automatiquement au lancement en bas à droite de l'écran

## Installation

```bash
# Cloner le dépôt
git clone [url-du-repo]
cd electron-assistant-app

# Installer les dépendances
npm install

# Compiler le TypeScript
npm run build

# Lancer l'application
npm start
```

## Scripts disponibles

- `npm run build` - Compile les fichiers TypeScript
- `npm run watch` - Compile les fichiers TypeScript en mode watch
- `npm start` - Lance l'application 
- `npm run dev` - Compile et lance l'application (pratique pour le développement)

## Structure du projet

```
electron-assistant-app/
├── dist/               # Fichiers JavaScript compilés
├── src/
│   ├── main.ts         # Point d'entrée du processus principal
│   ├── preload.ts      # Script de preload
│   ├── types.d.ts      # Déclarations de types
│   └── renderer/       # Fichiers frontend
│       ├── css/        # Styles CSS
│       ├── js/         # Scripts JavaScript du renderer
│       └── index.html  # Page HTML
├── package.json        # Configuration npm
└── tsconfig.json       # Configuration TypeScript
```

## Personnalisation

Pour personnaliser l'apparence de l'assistant, modifiez les fichiers CSS dans `src/renderer/css/`. 