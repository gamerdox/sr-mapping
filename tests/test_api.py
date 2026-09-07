import io
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    res = client.get('/health')
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'healthy'
    assert 'sentinel2srnet' in data['loaded_models']
    assert 'fsrcnn' in data['loaded_models']
    assert data['default_model'] == 'sentinel2srnet'
    assert data['supports_tta'] is True

def test_get_available_models(client):
    res = client.get('/api/models')
    assert res.status_code == 200
    data = res.json()
    models = {m['id']: m for m in data['models']}
    assert 'sentinel2srnet' in models
    assert models['sentinel2srnet']['is_loaded'] is True
    assert models['sentinel2srnet']['is_default'] is True
    assert models['sentinel2srnet']['engine_type'] == 'pytorch'
    assert 'fsrcnn' in models
    assert models['fsrcnn']['is_loaded'] is True

def test_predict_sample_sentinel2srnet(client):
    res = client.post('/api/sr/predict-sample/0?model=sentinel2srnet')
    assert res.status_code == 200
    data = res.json()
    assert data['sample_id'] == 0
    assert 'Sentinel2SRNet' in data['model']
    assert 'quantitative_metrics' in data
    assert 'psnr' in data['quantitative_metrics']
    assert 'ssim' in data['quantitative_metrics']
    assert 'confidence_heatmap' in data['images']
    assert 'difference_heatmap' in data['images']
    assert 'cir_sr' in data['images']
    assert 'ndvi_sr' in data['images']
    assert 'post_processing' in data

def test_predict_sample_with_tta(client):
    res = client.post('/api/sr/predict-sample/0?model=sentinel2srnet&tta=true')
    assert res.status_code == 200
    data = res.json()
    assert data['tta_enabled'] is True
    assert '8x TTA' in data['model']

def test_predict_sample_fsrcnn_baseline(client):
    res = client.post('/api/sr/predict-sample/0?model=fsrcnn')
    assert res.status_code == 200
    data = res.json()
    assert 'FSRCNN' in data['model']

def test_predict_sample_not_found(client):
    res = client.post('/api/sr/predict-sample/999')
    assert res.status_code == 404

def test_predict_custom_valid_image(client):
    # Create small 64x64 valid PNG
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    res = client.post(
        '/api/sr/predict?return_base64=true&model=sentinel2srnet',
        files={'file': ('test.png', buf.getvalue(), 'image/png')}
    )
    assert res.status_code == 200
    data = res.json()
    assert data['input_dimensions'] == '64x64'
    assert data['output_dimensions'] == '128x128'
    assert 'confidence_heatmap' in data['images']

def test_predict_upload_oversized_file(client):
    # 11 MB payload
    big_bytes = b'0' * (11 * 1024 * 1024)
    res = client.post(
        '/api/sr/predict',
        files={'file': ('big.png', big_bytes, 'image/png')}
    )
    assert res.status_code == 413

def test_predict_upload_invalid_file(client):
    bad_bytes = b'not an image content'
    res = client.post(
        '/api/sr/predict',
        files={'file': ('bad.txt', bad_bytes, 'text/plain')}
    )
    assert res.status_code == 400

def test_predict_multibiome_samples(client):
    for sample_id in [5, 6, 7]:
        res = client.post(f'/api/sr/predict-sample/{sample_id}')
        assert res.status_code == 200
        data = res.json()
        assert data['sample_id'] == sample_id
        assert 'quantitative_metrics' in data
        assert data['quantitative_metrics']['psnr']['model_sr_db'] > 15.0

def test_predict_location_cached(client):
    # Paris coordinates
    res = client.post('/api/sr/predict-location?lat=48.8584&lon=2.2945&zoom=16')
    assert res.status_code == 200
    data = res.json()
    assert 'Paris' in data['source'] or 'Live Satellite' in data['source']
    assert 'confidence_heatmap' in data['images']
    assert data['validation_status'].startswith('Unvalidated')

def test_predict_sample_scale_4x_sub4m(client):
    """Verify that 4x progressive cascaded mode produces 2.5m GSD (Sub-4m Target) and 512x512 resolution."""
    res = client.post('/api/sr/predict-sample/5?model=sentinel2srnet&scale=4')
    assert res.status_code == 200
    data = res.json()
    assert data['scale_factor'] == '4x'
    assert '2.5m' in data['target_gsd']
    assert data['output_dimensions'] == '512x512'
    assert data['edge_sharpness']['sharpness_gain'] > 0
    assert 'confidence_heatmap' in data['images']


