import os

# Use absolute paths to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}

# Increase size limit to 500MB
MAX_FILE_SIZE = 500 * 1024 * 1024 

MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'merged_adapter_latest.pt')
MODEL_LOADED = False

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)