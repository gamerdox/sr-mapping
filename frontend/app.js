let currentData = null;
let showingHeatmap = false;
let loupeActive = false;
let map;
let clickMarker = null;

// Active state tracking to enable instant re-running when model or TTA changes
let activeSourceType = 'location'; // 'location' | 'sample' | 'upload'
let activeLocationParams = { lat: 48.8584, lng: 2.2945, zoom: 16, name: 'Eiffel Tower (Paris, France)' };
let activeSampleId = null;
let activeUploadFile = null;

function getModelParams() {
  const modelSelect = document.getElementById('modelSelect');
  const ttaToggle = document.getElementById('ttaToggle');
  const model = modelSelect ? modelSelect.value : 'fsrcnn';
  const tta = ttaToggle ? ttaToggle.checked : false;
  return { model, tta };
}

function onModelOrTTAChanged() {
  const { model, tta } = getModelParams();

  // Update header badges
  const modelBadge = document.getElementById('activeModelBadge');
  if (modelBadge) {
    modelBadge.innerText = `Model: ${model.toUpperCase()} (2x)`;
    if (model === 'edsr') {
      modelBadge.style.background = '#7c2d12';
      modelBadge.style.color = '#fdba74';
    } else if (model === 'espcn') {
      modelBadge.style.background = '#065f46';
      modelBadge.style.color = '#6ee7b7';
    } else {
      modelBadge.style.background = '#1e3a8a';
      modelBadge.style.color = '#93c5fd';
    }
  }

  const ttaBadge = document.getElementById('activeTtaBadge');
  if (ttaBadge) {
    ttaBadge.style.display = tta ? 'inline-block' : 'none';
  }

  // Show warning banner if deep EDSR is selected
  const edsrWarning = document.getElementById('edsrWarning');
  if (edsrWarning) {
    edsrWarning.style.display = (model === 'edsr') ? 'block' : 'none';
  }

  // Re-run active view with newly selected model/TTA configuration
  if (activeSourceType === 'location' && activeLocationParams) {
    loadLiveLocation(activeLocationParams.lat, activeLocationParams.lng, activeLocationParams.zoom);
  } else if (activeSourceType === 'sample' && activeSampleId !== null) {
    loadSample(activeSampleId);
  } else if (activeSourceType === 'upload' && activeUploadFile) {
    uploadExistingFile(activeUploadFile);
  }
}

function showLoader(msg) {
  const sLoader = document.getElementById('studioLoader');
  const sText = document.getElementById('studioLoaderText');
  const bLoader = document.getElementById('loader');
  if (sLoader && sText) {
    sText.innerText = msg;
    sLoader.style.display = 'flex';
  }
  if (bLoader) {
    bLoader.innerText = msg;
    bLoader.style.display = 'block';
  }
}

function hideLoader() {
  const sLoader = document.getElementById('studioLoader');
  const bLoader = document.getElementById('loader');
  if (sLoader) sLoader.style.display = 'none';
  if (bLoader) bLoader.style.display = 'none';
}

// 1. Initialize Leaflet World Satellite Map
function initMap() {
  const defaultCoords = [-2.609, -60.209]; // Manaus, Brazil
  map = L.map('map', {
    center: defaultCoords,
    zoom: 12,
    minZoom: 2,
    maxZoom: 18
  });

  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Earth Observation Satellite Tiles',
    maxZoom: 18,
  }).addTo(map);

  map.on('click', async (e) => {
    const { lat, lng } = e.latlng;
    const zoom = Math.max(14, map.getZoom());

    if (clickMarker) {
      map.removeLayer(clickMarker);
    }

    clickMarker = L.marker([lat, lng], {
      title: "Selected Location"
    }).addTo(map);

    clickMarker.bindPopup(`<b>Super-Resolving Location:</b><br>Lat: ${lat.toFixed(4)}, Lon: ${lng.toFixed(4)}<br><i>Fetching live satellite tile...</i>`).openPopup();

    await loadLiveLocation(lat, lng, zoom);
  });
}

// 2. Fetch & Super-Resolve from Live Coordinates
async function loadLiveLocation(lat, lng, zoom = 15) {
  activeSourceType = 'location';
  activeLocationParams = { lat, lng, zoom };
  const { model, tta } = getModelParams();

  showLoader(`Running ${model.toUpperCase()}${tta ? ' + 8x TTA' : ''} on Lat: ${lat.toFixed(4)}, Lon: ${lng.toFixed(4)}...`);
  document.querySelectorAll('#sampleButtons button').forEach(b => b.classList.remove('active'));

  try {
    const res = await fetch(`/api/sr/predict-location?lat=${lat}&lon=${lng}&zoom=${zoom}&model=${model}&tta=${tta}`, {
      method: 'POST'
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch satellite tile: ${res.statusText}`);
    }
    currentData = await res.json();
    document.getElementById('targetCard').style.display = 'none';
    document.getElementById('activeSourceTag').innerText = currentData.source;
    renderData(currentData);

    if (clickMarker) {
      clickMarker.bindPopup(`<b>Enhanced in Studio:</b><br>${currentData.source}<br>Model: ${currentData.model}<br>Resolution: ${currentData.input_dimensions} &rarr; ${currentData.output_dimensions}`).openPopup();
    }
  } catch (err) {
    console.error(err);
    alert('Error super-resolving location: ' + err.message);
  } finally {
    hideLoader();
  }
}

function jumpToLandmark(lat, lng, zoom, name) {
  map.setView([lat, lng], zoom);
  if (clickMarker) map.removeLayer(clickMarker);
  clickMarker = L.marker([lat, lng]).addTo(map);
  clickMarker.bindPopup(`<b>${name}</b><br>Super-resolving...`).openPopup();
  activeLocationParams = { lat, lng, zoom, name };
  loadLiveLocation(lat, lng, zoom);
}

// 3. Split Screen Slider Logic
const splitBox = document.getElementById('splitBox');
const splitOverlay = document.getElementById('splitOverlay');
const splitHandle = document.getElementById('splitHandle');
const loupe = document.getElementById('loupe');

function updateSplit(clientX) {
  const rect = splitBox.getBoundingClientRect();
  let pos = clientX - rect.left;
  pos = Math.max(10, Math.min(pos, rect.width - 10));
  const pct = (pos / rect.width) * 100;
  splitOverlay.style.width = pct + '%';
  splitHandle.style.left = pct + '%';
}

let isDragging = false;
splitBox.addEventListener('mousedown', (e) => {
  if (loupeActive) return;
  isDragging = true;
  updateSplit(e.clientX);
});
window.addEventListener('mouseup', () => { isDragging = false; });
window.addEventListener('mousemove', (e) => {
  if (isDragging) updateSplit(e.clientX);
  if (loupeActive) moveLoupe(e);
});
splitBox.addEventListener('touchstart', (e) => { updateSplit(e.touches[0].clientX); });
splitBox.addEventListener('touchmove', (e) => { updateSplit(e.touches[0].clientX); });

// 4. Magnifying Loupe (400% Zoom Lens)
function toggleLoupe() {
  loupeActive = !loupeActive;
  const btn = document.getElementById('loupeBtn');
  if (loupeActive) {
    btn.classList.add('active');
    btn.innerText = 'Disable 4x Loupe';
    splitHandle.style.display = 'none';
  } else {
    btn.classList.remove('active');
    btn.innerText = '4x Zoom Loupe';
    splitHandle.style.display = 'flex';
    loupe.style.display = 'none';
  }
}

function moveLoupe(e) {
  if (!currentData || !loupeActive) return;
  const rect = splitBox.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
    loupe.style.display = 'none';
    return;
  }

  loupe.style.display = 'block';
  loupe.style.left = (x - 70) + 'px';
  loupe.style.top = (y - 70) + 'px';

  const zoomFactor = 4;
  const imgSrc = (x < splitOverlay.offsetWidth) ? currentData.images.bicubic : currentData.images.sr_enhanced;
  loupe.style.backgroundImage = `url("${imgSrc}")`;
  loupe.style.backgroundSize = `${rect.width * zoomFactor}px ${rect.height * zoomFactor}px`;
  loupe.style.backgroundPosition = `-${(x * zoomFactor) - 70}px -${(y * zoomFactor) - 70}px`;
}

// 5. Heatmap Toggle
function toggleHeatmap() {
  if (!currentData) return;
  showingHeatmap = !showingHeatmap;
  const btn = document.getElementById('heatmapBtn');
  const rightLabel = document.getElementById('rightLabel');
  const rightImg = document.getElementById('sliderRightImg');

  if (showingHeatmap) {
    rightImg.src = currentData.images.difference_heatmap;
    btn.classList.add('active');
    btn.innerText = 'View AI Enhanced Image';
    rightLabel.innerText = 'AI Difference Heatmap (Synthesized Edges) ▶';
  } else {
    rightImg.src = currentData.images.sr_enhanced;
    btn.classList.remove('active');
    btn.innerText = 'AI Difference Heatmap';
    rightLabel.innerText = `${currentData.model} ▶`;
  }
}

// 6. Load Dataset Sample
async function loadSample(id) {
  activeSourceType = 'sample';
  activeSampleId = id;
  const { model, tta } = getModelParams();

  document.querySelectorAll('#sampleButtons button').forEach((b, idx) => {
    b.classList.toggle('active', idx === id);
  });
  showLoader(`Running ${model.toUpperCase()}${tta ? ' + 8x TTA' : ''} on Sample ${id}...`);
  document.getElementById('targetCard').style.display = 'flex';
  document.getElementById('activeSourceTag').innerText = `SEN2VENµS Sample ${id} (Manaus, Brazil)`;

  try {
    const res = await fetch(`/api/sr/predict-sample/${id}?model=${model}&tta=${tta}`, { method: 'POST' });
    currentData = await res.json();
    renderData(currentData);
  } catch (err) {
    console.error(err);
    alert('Failed to load sample: ' + err.message);
  } finally {
    hideLoader();
  }
}

// 7. Handle Custom Image Upload
async function handleCustomUpload(input) {
  if (!input.files || !input.files[0]) return;
  activeUploadFile = input.files[0];
  activeSourceType = 'upload';
  await uploadExistingFile(activeUploadFile);
}

async function uploadExistingFile(file) {
  const { model, tta } = getModelParams();
  const formData = new FormData();
  formData.append('file', file);

  document.querySelectorAll('#sampleButtons button').forEach(b => b.classList.remove('active'));
  showLoader(`Running ${model.toUpperCase()}${tta ? ' + 8x TTA' : ''} on ${file.name}...`);
  document.getElementById('activeSourceTag').innerText = `Uploaded: ${file.name}`;

  try {
    const res = await fetch(`/api/sr/predict?return_base64=true&model=${model}&tta=${tta}`, {
      method: 'POST',
      body: formData
    });
    currentData = await res.json();
    document.getElementById('targetCard').style.display = 'none';
    renderData(currentData);
  } catch (err) {
    console.error(err);
    alert('Upload failed: ' + err.message);
  } finally {
    hideLoader();
  }
}

// 8. Render Data into UI
function renderData(data) {
  document.getElementById('sliderLeftImg').src = data.images.bicubic;
  document.getElementById('sliderRightImg').src = showingHeatmap ? data.images.difference_heatmap : data.images.sr_enhanced;

  document.getElementById('imgLR').src = data.images.lr;
  document.getElementById('imgBicubic').src = data.images.bicubic;
  document.getElementById('imgSR').src = data.images.sr_enhanced;
  if (data.images.ground_truth_target) {
    document.getElementById('imgTarget').src = data.images.ground_truth_target;
  }

  const rightLabel = document.getElementById('rightLabel');
  if (showingHeatmap) {
    rightLabel.innerText = 'AI Difference Heatmap (Synthesized Edges) ▶';
  } else {
    rightLabel.innerText = `${data.model} ▶`;
  }

  const cardSRTitle = document.getElementById('cardSRTitle');
  if (cardSRTitle) {
    cardSRTitle.innerText = `3. ${data.model}`;
  }

  document.getElementById('pixelsStat').innerText = `${data.pixel_count_input.toLocaleString()} → ${data.pixel_count_output.toLocaleString()} (4×)`;
  document.getElementById('latencyStat').innerText = `${data.latency_ms} ms`;
  const gain = data.edge_sharpness.sharpness_gain;
  document.getElementById('edgeStat').innerText = `${gain >= 0 ? '+' : ''}${gain}`;

  document.getElementById('lrDimTag').innerText = `${data.input_dimensions}`;
  document.getElementById('bicubicTag').innerText = `Edge: ${data.edge_sharpness.bicubic}`;
  document.getElementById('srTag').innerText = `Edge: ${data.edge_sharpness.model_sr}`;
}

// 9. Download Enhanced Image
function downloadEnhanced() {
  if (!currentData || !currentData.images.sr_enhanced) return;
  const link = document.createElement('a');
  link.href = currentData.images.sr_enhanced;
  link.download = `super_resolved_${Date.now()}.png`;
  link.click();
}

// Startup
window.addEventListener('DOMContentLoaded', () => {
  initMap();
  jumpToLandmark(48.8584, 2.2945, 16, 'Eiffel Tower (Paris, France)');
});

