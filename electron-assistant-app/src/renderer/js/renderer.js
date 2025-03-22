// Gérer le bouton de fermeture
document.getElementById('close-btn').addEventListener('click', () => {
  // En Electron, window.close() ne fonctionne pas toujours
  // Nous utilisons donc l'API IPC pour envoyer un signal au processus principal
  // Mais comme nous n'avons pas configuré IPC pour cet exemple simple, 
  // nous utilisons window.close() comme solution temporaire
  window.close();
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