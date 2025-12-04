// === script.js ===

// Seletores principais
const openCameraBtn = document.getElementById('openCamera');
const captureBtn = document.getElementById('capture');
const retakeBtn = document.getElementById('retake');
const closeCameraBtn = document.getElementById('closeCamera');
const analyzeBtn = document.getElementById('analyzeBtn');
const fileInput = document.getElementById('file');

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const preview = document.getElementById('preview');

const loader = document.getElementById('loader');
const resultSection = document.getElementById('result');
const overlayImg = document.getElementById('overlayImg');
const formatText = document.getElementById('formatText');
const measuresText = document.getElementById('measuresText');
const recsDiv = document.getElementById('recs');
const cutsGallery = document.getElementById('cutsGallery');

let stream = null;
let capturedBlob = null;

// ======== Funções auxiliares ========

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  video.style.display = 'none';
  captureBtn.disabled = true;
  closeCameraBtn.style.display = 'none';
  openCameraBtn.style.display = 'inline-block';
}

function showLoader(msg = "Analisando rosto...") {
  loader.classList.remove('hidden');
  loader.innerHTML = `
    <div class="spinner"></div>
    <p>${msg}</p>
  `;
  analyzeBtn.style.display = 'none';
}

function hideLoader() {
  loader.classList.add('hidden');
  loader.innerHTML = '';
  analyzeBtn.style.display = 'inline-block';
}

// ======== Câmera ========

openCameraBtn.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = stream;
    video.style.display = 'block';
    captureBtn.disabled = false;
    closeCameraBtn.style.display = 'inline-block';
    openCameraBtn.style.display = 'none';
    preview.classList.add('hidden');
    resultSection.classList.add('hidden');
  } catch (err) {
    alert('Não foi possível acessar a câmera: ' + err.message);
  }
});

captureBtn.addEventListener('click', () => {
  if (!stream) return alert('Abra a câmera primeiro.');

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  canvas.width = vw;
  canvas.height = vh;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, vw, vh);

  canvas.toBlob((blob) => {
    capturedBlob = blob;
    preview.src = URL.createObjectURL(blob);
    preview.classList.remove('hidden');
    video.style.display = 'none';
    captureBtn.style.display = 'none';
    retakeBtn.style.display = 'inline-block';
  }, 'image/jpeg', 0.95);
});

retakeBtn.addEventListener('click', () => {
  preview.classList.add('hidden');
  video.style.display = 'block';
  retakeBtn.style.display = 'none';
  captureBtn.style.display = 'inline-block';
  resultSection.classList.add('hidden');
});

closeCameraBtn.addEventListener('click', () => {
  stopCamera();
  preview.classList.add('hidden');
  resultSection.classList.add('hidden');
});

// ======== Upload manual ========

fileInput.addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  capturedBlob = null;
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.classList.remove('hidden');
    video.style.display = 'none';
    resultSection.classList.add('hidden');
  };
  reader.readAsDataURL(f);
  stopCamera();
});

// ======== Analisar rosto ========

analyzeBtn.addEventListener('click', async () => {
  const genero = document.getElementById('genero').value;
  const fd = new FormData();
  fd.append('genero', genero);

  if (capturedBlob) {
    fd.append('file', capturedBlob, 'webcam.jpg');
  } else if (fileInput.files && fileInput.files.length > 0) {
    fd.append('file', fileInput.files[0]);
  } else {
    alert('Por favor, envie uma foto ou tire uma agora.');
    return;
  }

  showLoader();

  try {
    const res = await fetch('/analisar', { method: 'POST', body: fd });
    const data = await res.json();
    hideLoader();

    if (!res.ok) {
      alert(data.erro || 'Erro ao analisar rosto.');
      return;
    }

    // Mostra resultado
    resultSection.classList.remove('hidden');

    const overlayUrl = data.overlay_data_url || data.overlay_url;
    if (overlayUrl) {
      overlayImg.src = overlayUrl;
      overlayImg.classList.remove('hidden');
    }

    formatText.innerHTML = `<strong>Formato:</strong> ${data.formato}`;
    const m = data.medidas || {};
    measuresText.innerHTML = `<strong>Medidas (px):</strong> 
      comprimento ${m.comprimento_px ?? '-'}, 
      testa ${m.testa_px ?? '-'}, 
      bochecha ${m.bochecha_px ?? '-'}, 
      mandíbula ${m.mandibula_px ?? '-'}`;

    recsDiv.innerHTML = '';
    cutsGallery.innerHTML = '';

    const cortes = data.cortes || {};
    for (const g in cortes) {
      const arr = cortes[g];
      if (!arr || arr.length === 0) continue;

      const title = document.createElement('h4');
      title.textContent = g === 'feminino' ? 'Feminino' : 'Masculino';
      recsDiv.appendChild(title);

      arr.forEach(item => {
        const p = document.createElement('p');
        p.innerHTML = `<strong>${item.nome || ''}</strong>: ${item.descricao || ''}`;
        recsDiv.appendChild(p);

        const card = document.createElement('div');
        card.className = 'card';
        const img = document.createElement('img');
        img.src = item.img_url || item.img;
        const nm = document.createElement('div');
        nm.style.fontWeight = '700';
        nm.textContent = item.nome || '';
        card.appendChild(img);
        card.appendChild(nm);
        cutsGallery.appendChild(card);
      });
    }

    analyzeBtn.style.display = 'none';

  } catch (err) {
    hideLoader();
    alert('Erro ao enviar: ' + err.message);
  }
});
