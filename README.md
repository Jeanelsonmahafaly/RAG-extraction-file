# --- Instructions d'installation et de lancement (Mises à jour) ---
# 1. pip install fastapi uvicorn python-dotenv langchain langchain-community langchain-openai pymupdf python-multipart
#    (Vous n'avez plus besoin d'installer `openai` ni `httpx` explicitement pour cette configuration si vous ne les utilisez pas ailleurs)
# 2. Assurez-vous que votre fichier .env est à la racine de votre projet (un dossier au-dessus de 'src')
#    Exemple de structure de projet:
#    mon_projet/
#    ├── .env
#    └── src/
#        └── fastapi_cv_extraction.py

# 3- install dependance 
# Créer un nouvel environnement
python -m venv new_venv
source new_venv/bin/activate  # Linux/Mac
# ou new_venv\Scripts\activate  # Windows

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4-  Relancer l'application
uvicorn src.extraction_file:app --reload --host 127.0.0.1 --port 8000
