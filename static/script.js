const grid = document.getElementById('sampleGrid');
const preview = document.getElementById('selectedImagePreview');
const status = document.getElementById('selectionStatus');
const form = document.getElementById('answerForm');
const input = document.getElementById('letterGuess');
const check = document.getElementById('checkAnswer');
const message = document.getElementById('message');
const wrongAnswerAudio = document.getElementById('wrongAnswerAudio');
function stopWrongAnswerAudio() {
  wrongAnswerAudio.pause();
  wrongAnswerAudio.currentTime = 0;
}
let expected = '';

async function loadImages() {
  try {
    const response = await fetch('/sample-images', { cache: 'no-store' });
    if (!response.ok) throw new Error('Não foi possível carregar as imagens.');
    const data = await response.json();
    const images = Object.entries(data).flatMap(([letter, urls]) => urls.map(url => ({url, letter})));
    if (!images.length) throw new Error('Nenhuma imagem foi encontrada.');
    for (let i = images.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [images[i], images[j]] = [images[j], images[i]];
    }
    grid.replaceChildren();
    images.forEach(({url, letter}, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'sample-card';
      button.setAttribute('aria-pressed', 'false');
      const img = document.createElement('img');
      img.loading = 'lazy';
      img.src = url;
      img.alt = 'Imagem de sinal ' + (index + 1);
      button.appendChild(img);
      button.addEventListener('click', () => {
        stopWrongAnswerAudio();
        expected = letter.trim().toUpperCase();
        preview.src = url;
        preview.hidden = false;
        status.textContent = 'Imagem selecionada';
        input.value = '';
        input.disabled = false;
        check.disabled = false;
        message.hidden = true;
        message.textContent = '';
        grid.querySelectorAll('.sample-card').forEach(card => {
          card.classList.toggle('selected', card === button);
          card.setAttribute('aria-pressed', String(card === button));
        });
        input.focus();
      });
      grid.appendChild(button);
    });
  } catch (error) {
    const notice = document.createElement('p');
    notice.className = 'empty-state error';
    notice.textContent = error.message;
    grid.replaceChildren(notice);
  }
}
input.addEventListener('input', () => { message.hidden = true; });
form.addEventListener('submit', event => {
  event.preventDefault();
  if (!expected || !form.reportValidity()) return;
  const guess = input.value.trim().toUpperCase();
  const correct = guess === expected;
  stopWrongAnswerAudio();
  if (!correct) {
    wrongAnswerAudio.play().catch(() => {
      // O resultado visual continua disponível se o navegador bloquear o áudio.
    });
  }
  message.textContent = correct
    ? 'Você acertou! A letra é ' + expected + '.'
    : 'Você respondeu ' + guess + '. A letra correta é ' + expected + '.';
  message.className = 'message ' + (correct ? 'success' : 'error');
  message.hidden = false;
});
window.addEventListener('DOMContentLoaded', loadImages);
