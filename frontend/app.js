let currentData = null;
let showingHeatmap = false;
let loupeActive = false;
let map;
let clickMarker = null;

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
  document.getElementById('loader').style.display = 'block';
  document.getElementById('loader').innerText = `Fetching live satellite imagery for Lat: ${lat.toFixed(4)}, Lon: ${lng.toFixed(4)}...`;
  document.querySelectorAll('#sampleButtons button').forEach(b => b.classList.remove('active'));

  try {
    const res = await fetch(`/api/sr/predict-location?lat=${lat}&lon=${lng}&zoom=${zoom}`, {
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
      clickMarker.bindPopup(`<b>Enhanced in Studio:</b><br>${currentData.source}<br>Resolution: ${currentData.input_dimensions} &rarr; ${currentData.output_dimensions}`).openPopup();
    }
  } catch (err) {
    console.error(err);
    alert('Error super-resolving location: ' + err.message);
  } finally {
    document.getElementById('loader').style.display = 'none';
  }
}

function jumpToLandmark(lat, lng, zoom, name) {
  map.setView([lat, lng], zoom);
  if (clickMarker) map.removeLayer(clickMarker);
  clickMarker = L.marker([lat, lng]).addTo(map);
  clickMarker.bindPopup(`<b>${name}</b><br>Super-resolving...`).openPopup();
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
    rightLabel.innerText = 'FSRCNN Pretrained AI (Crisp 2x) ▶';
  }
}

// 6. Load Dataset Sample
async function loadSample(id) {
  document.querySelectorAll('#sampleButtons button').forEach((b, idx) => {
    b.classList.toggle('active', idx === id);
  });
  document.getElementById('loader').style.display = 'block';
  document.getElementById('loader').innerText = 'Loading dataset sample patch...';
  document.getElementById('targetCard').style.display = 'flex';
  document.getElementById('activeSourceTag').innerText = `SEN2VENµS Sample ${id} (Manaus, Brazil)`;

  try {
    const res = await fetch(`/api/sr/predict-sample/${id}`, { method: 'POST' });
    currentData = await res.json();
    renderData(currentData);
  } catch (err) {
    console.error(err);
    alert('Failed to load sample: ' + err.message);
  } finally {
    document.getElementById('loader').style.display = 'none';
  }
}

// 7. Handle Custom Image Upload
async function handleCustomUpload(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  const formData = new FormData();
  formData.append('file', file);

  document.querySelectorAll('#sampleButtons button').forEach(b => b.classList.remove('active'));
  document.getElementById('loader').style.display = 'block';
  document.getElementById('loader').innerText = 'Running official pretrained 2x super-resolution...';
  document.getElementById('activeSourceTag').innerText = `Uploaded: ${file.name}`;

  try {
    const res = await fetch('/api/sr/predict?return_base64=true', {
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
    document.getElementById('loader').style.display = 'none';
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
