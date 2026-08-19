from pathlib import Path

import kagglehub
from kagglehub.config import get_kaggle_credentials

DATA_DIR = Path(__file__).resolve().parent / "data"

# Competition downloads require a Kaggle API token.
# Create one at https://www.kaggle.com/settings/api
if not get_kaggle_credentials():
    kagglehub.login()

path = kagglehub.competition_download(
    "nlp-getting-started",
    output_dir=str(DATA_DIR),
)

print("Path to competition files:", path)
