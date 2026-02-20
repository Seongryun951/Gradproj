import getpass
import os
import sys

__USERNAME = getpass.getuser()
# havok_hdd/srjo에서 모델 다운로드 및 불러오기 수행
MODEL_PATH = '/havok_hdd/srjo/models'
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(_BASE_DIR, 'data', 'datasets')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

