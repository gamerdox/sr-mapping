let currentData = null;
let showingHeatmap = false;
let showingConfidence = false;
let loupeActive = false;
let currentSpectralMode = 'rgb'; // 'rgb' | 'cir' | 'ndvi'
let map;
let clickMarker = null;

// Active state tracking to enable instant re-running when model or TTA changes
let activeSourceType = 'location'; // 'location' | 'sample' | 'upload'
let activeLocationParams = { lat: 48.8584, lng: 2.2945, zoom: 16, name: 'Eiffel Tower (Paris, France)' };
let activeSampleId = null;
let activeUploadFile = null;

async function initModelDropdown() {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) return;
    const data = await res.json();
    const select = document.getElementById('modelSelect');
    if (!select || !data.models) return;

    select.innerHTML = '';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      const statusNote = m.is_loaded ? '' : ' [Not Loaded]';
      opt.textContent = `${m.badge} ${statusNote}`;
      opt.disabled = !m.is_loaded;
      if (m.is_default && m.is_loaded) opt.selected = true;
      select.appendChild(opt);
    });
  } catch (err) {
    console.warn("Could not populate models dynamically, keeping defaults:", err);
  }
}

function getModelParams() {
  const modelSelect = document.getElementById('modelSelect');
  const ttaToggle = document.getElementById('ttaToggle');
  const scaleSelect = document.getElementById('scaleSelect');
  const model = modelSelect ? modelSelect.value : 'sentinel2srnet';
  const tta = ttaToggle ? ttaToggle.checked : false;
  const scale = scaleSelect ? parseInt(scaleSelect.value, 10) : 4;
  return { model, tta, scale };
}

function onModelOrTTAChanged() {
  const { model, tta, scale } = getModelParams();

  // Update header badges
  const modelBadge = document.getElementById('activeModelBadge');
  if (modelBadge) {
    modelBadge.innerText = `Model: ${model.toUpperCase()} (${scale}x / ${scale === 4 ? '2.5m' : '5m'} GSD)`;
    if (model === 'sentinel2srnet') {
      modelBadge.style.background = '#065f46';
      modelBadge.style.color = '#6ee7b7';
    } else if (model === 'edsr') {
      modelBadge.style.background = '#7c2d12';
      modelBadge.style.color = '#fdba74';
    } else if (model === 'espcn') {
      modelBadge.style.background = '#0284c7';
      modelBadge.style.color = '#bae6fd';
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

let compareBaseline = 'input'; // 'input' | 'bicubic'

function setCompareBaseline(mode) {
  compareBaseline = mode;
  const btnIn = document.getElementById('btnCompareInput');
  const btnBic = document.getElementById('btnCompareBicubic');
  if (btnIn) btnIn.classList.toggle('active', mode === 'input');
  if (btnBic) btnBic.classList.toggle('active', mode === 'bicubic');
  updateActiveImages();
}

// Spectral Remote-Sensing Band Switcher
function setSpectralMode(mode) {
  currentSpectralMode = mode;
  document.querySelectorAll('.spec-btn').forEach(b => {
    if (b.id.startsWith('btnSpec')) b.classList.remove('active');
  });
  if (mode === 'rgb') document.getElementById('btnSpecRGB').classList.add('active');
  if (mode === 'cir') document.getElementById('btnSpecCIR').classList.add('active');
  if (mode === 'ndvi') document.getElementById('btnSpecNDVI').classList.add('active');

  updateActiveImages();
}

function updateActiveImages() {
  if (!currentData || !currentData.images) return;

  const splitBox = document.getElementById('splitBox');
  const sliderLeftImg = document.getElementById('sliderLeftImg');
  if (splitBox && sliderLeftImg) {
    sliderLeftImg.style.width = splitBox.offsetWidth + 'px';
  }

  let leftSrc = (compareBaseline === 'input' && currentData.images.lr_nearest && currentSpectralMode === 'rgb')
    ? currentData.images.lr_nearest
    : currentData.images.bicubic;
  let rightSrc = currentData.images.sr_enhanced;

  if (currentSpectralMode === 'cir' && currentData.images.cir_sr) {
    leftSrc = currentData.images.cir_bicubic;
    rightSrc = currentData.images.cir_sr;
  } else if (currentSpectralMode === 'ndvi' && currentData.images.ndvi_sr) {
    leftSrc = currentData.images.ndvi_bicubic;
    rightSrc = currentData.images.ndvi_sr;
  }

  if (showingConfidence && currentData.images.confidence_heatmap) {
    rightSrc = currentData.images.confidence_heatmap;
  } else if (showingHeatmap && currentData.images.difference_heatmap) {
    rightSrc = currentData.images.difference_heatmap;
  }

  document.getElementById('sliderLeftImg').src = leftSrc;
  document.getElementById('sliderRightImg').src = rightSrc;

  const leftLabel = document.querySelector('.split-label.label-left');
  if (leftLabel) {
    const isSub4m = currentData && currentData.scale_factor === '4x';
    leftLabel.innerText = (compareBaseline === 'input' && currentSpectralMode === 'rgb')
      ? '◀ Sentinel-2 10m (Raw Pixelated Input)'
      : `◀ Bicubic Baseline ${isSub4m ? '2.5m Grid' : '5m'} (Blurry)`;
  }

  const rightLabel = document.getElementById('rightLabel');
  if (showingConfidence) {
    rightLabel.innerText = '🎯 Model Confidence Map (Green=High, Red=Uncertain) ▶';
  } else if (showingHeatmap) {
    rightLabel.innerText = 'AI Difference Heatmap (Synthesized Edges) ▶';
  } else {
    const specTag = currentSpectralMode.toUpperCase();
    const gsdTag = currentData.target_gsd ? `(${currentData.target_gsd} ${specTag})` : `(Crisp 5m ${specTag})`;
    rightLabel.innerText = `${currentData.model} ${gsdTag} ▶`;
  }
}

// Geocoding City Search
function handleSearchKeyDown(e) {
  if (e.key === 'Enter') {
    executeCitySearch();
  }
}

async function executeCitySearch() {
  const input = document.getElementById('citySearchInput');
  const query = input ? input.value.trim() : '';
  if (!query) return;

  showLoader(`Searching location "${query}" via OpenStreetMap...`);

  try {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`;
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!res.ok) throw new Error('Geocoding service unavailable.');
    const data = await res.json();
    if (!data || data.length === 0) {
      alert(`Location "${query}" not found. Try another city name.`);
      hideLoader();
      return;
    }

    const lat = parseFloat(data[0].lat);
    const lon = parseFloat(data[0].lon);
    const displayName = data[0].display_name.split(',')[0];

    map.flyTo([lat, lon], 16, { duration: 1.5 });
    if (clickMarker) map.removeLayer(clickMarker);
    clickMarker = L.marker([lat, lon]).addTo(map);
    clickMarker.bindPopup(`<b>${displayName}</b><br>Fetching live satellite tile...`).openPopup();

    activeLocationParams = { lat, lng: lon, zoom: 16, name: displayName };
    await loadLiveLocation(lat, lon, 16);
  } catch (err) {
    console.error(err);
    alert('Search error: ' + err.message);
    hideLoader();
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
  const { model, tta, scale } = getModelParams();

  showLoader(`Running ${model.toUpperCase()} (${scale}x / ${scale === 4 ? '2.5m' : '5m'} GSD)${tta ? ' + 8x TTA' : ''} on Lat: ${lat.toFixed(4)}, Lon: ${lng.toFixed(4)}...`);
  document.querySelectorAll('#sampleButtons button').forEach(b => b.classList.remove('active'));

  try {
    const res = await fetch(`/api/sr/predict-location?lat=${lat}&lon=${lng}&zoom=${zoom}&model=${model}&tta=${tta}&scale=${scale}`, {
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
  const leftImg = document.getElementById('sliderLeftImg');
  if (leftImg) leftImg.style.width = rect.width + 'px';
  let pos = clientX - rect.left;
  pos = Math.max(10, Math.min(pos, rect.width - 10));
  const pct = (pos / rect.width) * 100;
  splitOverlay.style.width = pct + '%';
  splitHandle.style.left = pct + '%';
}

window.addEventListener('resize', () => {
  if (splitBox && document.getElementById('sliderLeftImg')) {
    document.getElementById('sliderLeftImg').style.width = splitBox.offsetWidth + 'px';
  }
});

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

// 5. Heatmap & Confidence Toggles
function toggleHeatmap() {
  if (!currentData) return;
  showingHeatmap = !showingHeatmap;
  showingConfidence = false;

  const btnH = document.getElementById('heatmapBtn');
  const btnC = document.getElementById('confBtn');
  if (btnC) {
    btnC.classList.remove('active');
    btnC.innerText = '🎯 Confidence Map';
  }

  if (showingHeatmap) {
    btnH.classList.add('active');
    btnH.innerText = 'View Super-Resolved Image';
  } else {
    btnH.classList.remove('active');
    btnH.innerText = 'AI Difference Heatmap';
  }
  updateActiveImages();
}

function toggleConfidence() {
  if (!currentData) return;
  showingConfidence = !showingConfidence;
  showingHeatmap = false;

  const btnC = document.getElementById('confBtn');
  const btnH = document.getElementById('heatmapBtn');
  if (btnH) {
    btnH.classList.remove('active');
    btnH.innerText = 'AI Difference Heatmap';
  }

  if (showingConfidence) {
    btnC.classList.add('active');
    btnC.innerText = 'View Super-Resolved Image';
  } else {
    btnC.classList.remove('active');
    btnC.innerText = '🎯 Confidence Map';
  }
  updateActiveImages();
}

const SAMPLE_LABELS = {
  0: 'SEN2VENµS Sample 0 (Rainforest Canopy - Manaus, Brazil)',
  1: 'SEN2VENµS Sample 1 (Parcel Boundary - Manaus, Brazil)',
  2: 'SEN2VENµS Sample 2 (River / Water Body - Amazon)',
  3: 'SEN2VENµS Sample 3 (Topography & Canopy - Manaus, Brazil)',
  4: 'SEN2VENµS Sample 4 (High Contrast Forest Clearing - Amazon)',
  5: 'Validation Sample 5 (Urban Buildings & Roads - Versailles)',
  6: 'Validation Sample 6 (Agricultural Cropland & Field Boundaries - Bordeaux)',
  7: 'Validation Sample 7 (Coastal & Estuary Edge - Gironde)',
};

// 6. Load Dataset Sample
async function loadSample(id) {
  activeSourceType = 'sample';
  activeSampleId = id;
  const { model, tta, scale } = getModelParams();

  document.querySelectorAll('#sampleButtons button').forEach((b) => {
    b.classList.toggle('active', b.getAttribute('onclick') === `loadSample(${id})`);
  });
  showLoader(`Running ${model.toUpperCase()} (${scale}x / ${scale === 4 ? '2.5m' : '5m'} GSD)${tta ? ' + 8x TTA' : ''} on Sample ${id}...`);
  document.getElementById('targetCard').style.display = 'flex';
  const label = SAMPLE_LABELS[id] || `Benchmark Sample ${id}`;
  document.getElementById('activeSourceTag').innerText = label;

  try {
    const res = await fetch(`/api/sr/predict-sample/${id}?model=${model}&tta=${tta}&scale=${scale}`, { method: 'POST' });
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
  const { model, tta, scale } = getModelParams();
  const formData = new FormData();
  formData.append('file', file);

  document.querySelectorAll('#sampleButtons button').forEach(b => b.classList.remove('active'));
  showLoader(`Running ${model.toUpperCase()} (${scale}x / ${scale === 4 ? '2.5m' : '5m'} GSD)${tta ? ' + 8x TTA' : ''} on ${file.name}...`);
  document.getElementById('activeSourceTag').innerText = `Uploaded: ${file.name}`;

  try {
    const res = await fetch(`/api/sr/predict?return_base64=true&model=${model}&tta=${tta}&scale=${scale}`, {
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
  updateActiveImages();

  document.getElementById('imgLR').src = data.images.lr;
  document.getElementById('imgBicubic').src = data.images.bicubic;
  document.getElementById('imgSR').src = data.images.sr_enhanced;
  if (data.images.ground_truth_target) {
    document.getElementById('imgTarget').src = data.images.ground_truth_target;
  }

  const cardBicTitle = document.getElementById('cardBicTitle');
  if (cardBicTitle) {
    cardBicTitle.innerText = `2. Bicubic Baseline (${data.scale_factor || '4x'})`;
  }

  const cardSRTitle = document.getElementById('cardSRTitle');
  if (cardSRTitle) {
    cardSRTitle.innerText = `3. ${data.model}`;
  }

  // Update validation badge
  const valBadge = document.getElementById('validationBadge');
  if (valBadge && data.validation_status) {
    valBadge.innerText = data.validation_status.split('(')[0].trim();
    if (data.validation_status.includes('Validated (Paired') || data.validation_status.includes('Validated (VENµS')) {
      valBadge.style.background = '#14532d';
      valBadge.style.color = '#86efac';
    } else {
      valBadge.style.background = '#78350f';
      valBadge.style.color = '#fde68a';
    }
  }

  const mult = Math.round(data.pixel_count_output / data.pixel_count_input);
  const targetTag = data.target_gsd || (mult === 16 ? '2.5m Sub-4m' : '5m GSD');
  document.getElementById('pixelsStat').innerText = `${data.pixel_count_input.toLocaleString()} → ${data.pixel_count_output.toLocaleString()} (${mult}× / ${targetTag})`;
  document.getElementById('latencyStat').innerText = `${data.latency_ms} ms`;
  const gain = data.edge_sharpness.sharpness_gain;
  document.getElementById('edgeStat').innerText = `${gain >= 0 ? '+' : ''}${gain}`;

  // Quantitative PSNR / SSIM Fidelity Display
  const metricsStat = document.getElementById('metricsStat');
  const bicubicMetricsTag = document.getElementById('bicubicMetricsTag');
  const srMetricsTag = document.getElementById('srMetricsTag');
  const targetMetricsTag = document.getElementById('targetMetricsTag');

  if (data.quantitative_metrics) {
    const p = data.quantitative_metrics.psnr;
    const s = data.quantitative_metrics.ssim;
    if (metricsStat) metricsStat.innerText = `${p.model_sr_db.toFixed(1)} dB / SSIM: ${s.model_sr.toFixed(3)}`;
    if (bicubicMetricsTag) {
      bicubicMetricsTag.innerText = `PSNR: ${p.bicubic_db.toFixed(1)} dB`;
      bicubicMetricsTag.style.display = 'inline-block';
    }
    if (srMetricsTag) {
      srMetricsTag.innerText = `PSNR: ${p.model_sr_db.toFixed(1)} dB (${p.gain_db >= 0 ? '+' : ''}${p.gain_db.toFixed(2)} dB)`;
      srMetricsTag.style.display = 'inline-block';
    }
    if (targetMetricsTag) {
      targetMetricsTag.innerText = `SSIM: ${s.model_sr.toFixed(3)}`;
      targetMetricsTag.style.display = 'inline-block';
    }
  } else {
    if (metricsStat) metricsStat.innerHTML = '<span style="color:#fde68a;">Unvalidated (Geo-Tile)</span>';
    if (bicubicMetricsTag) bicubicMetricsTag.style.display = 'none';
    if (srMetricsTag) srMetricsTag.style.display = 'none';
    if (targetMetricsTag) targetMetricsTag.style.display = 'none';
  }

  document.getElementById('lrDimTag').innerText = `${data.input_dimensions} (10m Native GSD)`;
  document.getElementById('bicubicTag').innerText = `${data.output_dimensions} | Edge: ${data.edge_sharpness.bicubic}`;
  document.getElementById('srTag').innerText = `${data.output_dimensions} (${data.target_gsd || '2.5m GSD'}) | Edge: ${data.edge_sharpness.model_sr}`;
  const targetTagEl = document.getElementById('targetTag');
  if (targetTagEl) {
    targetTagEl.innerText = `${data.output_dimensions} (Ground Truth Reference)`;
  }
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
window.addEventListener('DOMContentLoaded', async () => {
  await initModelDropdown();
  initMap();
  jumpToLandmark(48.8584, 2.2945, 16, 'Eiffel Tower (Paris, France)');
});

