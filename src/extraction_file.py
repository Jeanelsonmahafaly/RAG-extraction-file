import os
import json
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI # Import standard pour LangChain
from openai import OpenAIError # Gardé pour gérer les erreurs spécifiques d'OpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CV Extractor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5000", "http://localhost:4200", "*"],  # ajustez
    allow_methods=["POST"],
    allow_headers=["*"],
)
# --- Chargement des variables d'environnement ---
# Charger les variables d'environnement depuis le .env à la racine du projet.
# Cette logique est correcte si le .env est un répertoire au-dessus du script FastAPI.
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dotenv_path = os.path.join(base_dir, '.env')
load_dotenv(dotenv_path)

# --- Vérification et définition de la clé OpenAI ---
# IMPORTANT : NE PAS ÉCRASER os.environ["OPENAI_API_KEY"] avec une valeur bidon ici.
# La clé doit venir de votre fichier .env ou de l'environnement système.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        f"La variable d'environnement OPENAI_API_KEY doit être définie dans {dotenv_path} ou dans l'environnement système."
    )

# LangChain va automatiquement prendre OPENAI_API_KEY de l'environnement.
# Pas besoin d'initialiser un client OpenAI direct ici si vous n'avez pas de besoins avancés.
# Suppression des lignes :
# transport = httpx.HTTPTransport()
# openai_client = OpenAI(transport=transport)
# car elles causaient le TypeError.

# --- Initialiser le modèle OpenAI (avec fallback si nécessaire) ---
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
try:
    # ChatOpenAI de langchain_openai gère l'API key via l'environnement
    llm = ChatOpenAI(model=model_name, temperature=0) # Suppression de l'argument client=openai_client
except OpenAIError as e:
    print(f"ATTENTION: Modèle '{model_name}' non disponible ou erreur d'initialisation ({e}). Tentative avec 'gpt-3.5-turbo'.")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Pas d'argument client ici non plus
except Exception as e:
    # Gérer d'autres erreurs potentielles lors de l'initialisation du LLM
    raise RuntimeError(f"Erreur inattendue lors de l'initialisation du LLM: {str(e)}")

# --- Préparer le prompt d'extraction en français ---
template = """
Lis et analyse le texte suivant issu d'un fichier: {doc_text}.
Extrait les champs suivants et retourne un dictionnaire JSON valide :

- Nom (chaîne de caractères)
- Prénom (chaîne de caractères)
- Nom de jeune fille (chaîne de caractères)
- Numéro d'employé (chaîne de caractères)
- Titre (chaîne de caractères, ex : « Ingénieur », « Développeur »)
- Date de naissance (chaîne, format JJ/MM/AAAA)
- Lieu de naissance (chaîne de caractères)
- Nationalité (chaîne de caractères)
- Sexe (chaîne de caractères, ex : « Homme », « Femme »)
- Situation familiale (chaîne de caractères, ex : « Célibataire », « Marié(e) »)
- Nombre d'enfants (entier)
- Numéro de sécurité sociale (chaîne de caractères)
- Adresse postale (chaîne de caractères)
- Code postal (chaîne de caractères)
- Adresse e-mail (chaîne de caractères)
- Numéro de téléphone (chaîne de caractères)
- Adresse (chaîne de caractères)

Assure-toi que la sortie est un JSON valide et rien d'autre.
Tout champ manquant → null (ou 0 pour l'entier) ; ne saute aucun champ.
"""

prompt = PromptTemplate(template=template, input_variables=["doc_text"])

# Utiliser LLMChain pour l'extraction
llm_chain = LLMChain(llm=llm, prompt=prompt)

# --- Initialisation de l'app FastAPI ---
app = FastAPI(
    title="API d'extraction d'un fichier PDF",
    description="Envoyez un PDF via le champ 'file' et récupérez un JSON structuré en réponse.",
)

@app.post(
    "/extract",
    summary="Extrait les informations personnelles d'un CV PDF",
    description="Téléversez votre CV (champ 'file') et obtenez un JSON structuré en réponse.",
)
async def extract_cv(file: UploadFile = File(..., description="Fichier PDF du CV")):
    # Vérification du type de fichier
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Le fichier doit être au format PDF.")

    # Sauvegarde temporaire du PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Chargement et lecture du PDF
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()

        # Initialisation du dictionnaire de résultats avec des valeurs par défaut pour une structure cohérente
        result = {
            "Nom": None,
            "Prénom": None,
            "Nom de jeune fille": None,
            "Numéro d'employé": None,
            "Titre": None,
            "Date de naissance": None,
            "Lieu de naissance": None,
            "Nationalité": None,
            "Sexe": None,
            "Situation familiale": None,
            "Nombre d'enfants": 0,
            "Numéro de sécurité sociale": None,
            "Adresse postale": None,
            "Code postal": None,
            "Adresse e-mail": None,
            "Numéro de téléphone": None,
            "Adresse": None,
        }


        # Extraction des données page par page
        for page in pages:
            #text = page.page_content.strip()

            all_text = "\n".join([page.page_content for page in pages if page.page_content.strip()])
            if not all_text: # Ignorer les pages vides après nettoyage
                continue

            try:
                # Appeler le LLM pour chaque page
                #response = llm_chain.invoke({"doc_text": text})
                # Combiner toutes les pages du PDF
                response = llm_chain.invoke({"doc_text": all_text})

                # Accéder à la sortie du LLM, en privilégiant "text" ou "content"
                raw_json = response.get("text", response.get("content", "")).replace("```json", "").replace("```", "").strip()

                # Tenter de parser le JSON
                data = json.loads(raw_json)

                # Fusion des données extraites
                for key, value in data.items():
                        if key in result and value not in (None, "", []):
                            result[key] = value

            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Le LLM renvoie un JSON invalide : {e}"
                )
            except OpenAIError as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Erreur OpenAI : {e}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur interne d’extraction : {e}"
                )

        # Retourner le JSON extrait
        return JSONResponse(content=result)

    finally:
        # S'assurer que le fichier temporaire est supprimé
        os.remove(tmp_path)

