// Importer l'API ipcRenderer d'Electron
const { ipcRenderer } = require('electron');

// Gérer le bouton de fermeture
document.getElementById('close-btn').addEventListener('click', () => {
  // Utiliser IPC pour envoyer un message au processus principal pour fermer la fenêtre
  ipcRenderer.send('close-window');
});

// Animation de bienvenue
document.addEventListener('DOMContentLoaded', () => {
  const content = document.querySelector('.assistant-content');
  const footer = document.querySelector('.assistant-footer');
  
  // Ajouter un petit délai pour que l'animation soit visible après le chargement
  setTimeout(() => {
    content.style.opacity = '1';
    footer.style.opacity = '1';
  }, 200);
}); 