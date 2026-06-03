# Game Recommendation System – React + FastAPI + Sentence Transformers + RAG

Dieses Projekt ist ein NLP-basiertes **Game Recommendation System** für Steam-Spiele. Nutzerinnen und Nutzer geben eine freie Spielpräferenz ein, wählen optionale Filter und erhalten anschließend erklärbare Spielempfehlungen. Zusätzlich enthält das Projekt ein Feedback-System, dynamisches Clustering und eine RAG-Erklärung für einzelne Spiele.

Die aktuelle Version nutzt:

- **React + Vite** für die Weboberfläche
- **FastAPI** als Python-Backend
- **Sentence Transformers** für semantische Suche
- **Okapi BM25** für konkrete Keyword-Treffer
- **Logistic Regression** für `passt / passt nicht`
- **Feedback-Learning** über Daumen hoch / Daumen runter
- **Dynamisches Clustering** der aktuellen Top-Ergebnisse
- **RAG-Erklärungen** mit Retrieval-Kontext und optional lokalem LLM über Ollama

---

## 1. Projektidee

Ziel des Projekts ist es, aus einer freien Nutzereingabe passende Steam-Spiele zu empfehlen.

Beispielhafte Nutzereingabe:

```text
Ich suche ein düsteres Sci-Fi-Spiel mit Story, Atmosphäre und Rätseln.
```

Das System kombiniert mehrere Informationsquellen:

1. Spielbeschreibung und Metadaten
2. Genres, Kategorien und SteamSpy-Tags
3. positive und negative Steam-Bewertungen
4. Preis, Spielzeit und Besitzerangaben
5. gelabelte Trainingsdaten für ein eigenes Klassifikationsmodell
6. bisheriges Nutzerfeedback

Aus diesen Informationen wird ein finaler Empfehlungsscore berechnet.

---

## 2. Datenbasis

Das Projekt verwendet drei zentrale Datendateien im Ordner `data/`:

```text
data/
├── steam.csv
├── steam_description_data.csv
└── labeled_manual.csv
```

### `steam.csv`

Enthält strukturierte Spielinformationen:

- `appid`
- `name`
- `release_date`
- `developer`
- `publisher`
- `platforms`
- `categories`
- `genres`
- `steamspy_tags`
- `positive_ratings`
- `negative_ratings`
- `average_playtime`
- `median_playtime`
- `owners`
- `price`

### `steam_description_data.csv`

Enthält Textdaten:

- `steam_appid`
- `detailed_description`
- `about_the_game`
- `short_description`

Die Verbindung erfolgt über:

```text
steam.csv.appid = steam_description_data.csv.steam_appid
```

### `labeled_manual.csv`

Enthält gelabelte Beispiele für supervised learning:

```text
user_preference, appid, game_name, game_text, label
```

Dabei bedeutet:

```text
label = 1  → Spiel passt zur Nutzerpräferenz
label = 0  → Spiel passt nicht zur Nutzerpräferenz
```

Diese Datei wird zum Trainieren des Logistic-Regression-Klassifikators verwendet.

---

## 3. Projektstruktur

```text
PROJECT_ROOT/
│
├── backend/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── steam.csv
│   ├── steam_description_data.csv
│   └── labeled_manual.csv
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── style.css
│
├── src/
│   ├── prepare_data.py
│   ├── create_labels.py
│   ├── train_classifier.py
│   ├── build_sentence_embeddings.py
│   ├── bm25.py
│   ├── recommender.py
│   ├── feedback.py
│   ├── dynamic_clustering.py
│   ├── rag_answer.py
│   └── evaluate.py
│
├── artifacts/
│   ├── games_processed.csv
│   ├── preference_classifier.pkl
│   ├── game_sentence_embeddings.npy
│   ├── embedding_model_name.txt
│   ├── bm25_index.pkl
│   ├── feedback_events.csv
│   └── recommendation_smoke_test.json
│
├── tests/
│   ├── conftest.py
│   └── test_backend.py
│
├── requirements.txt
├── run_pipeline.py
├── rebuild_all.ps1
├── rebuild_all.bat
├── start_backend.ps1
├── start_frontend.ps1
├── start_backend.bat
├── start_frontend.bat
└── README.md
```

---

## 4. Voraussetzungen

Für die React + FastAPI-Version werden Python und Node.js benötigt.

### Benötigte Software

1. **Python 3.10 oder neuer**
2. **Node.js LTS** inklusive `npm`
3. **PowerShell** oder ein anderes Terminal
4. Optional: **Ollama**, wenn echtes generatives RAG mit lokalem LLM genutzt werden soll

### Python prüfen

```powershell
python --version
```

oder:

```powershell
py --version
```

### Node.js und npm prüfen

```powershell
node -v
npm -v
```

Falls `npm` nicht gefunden wird, muss Node.js LTS installiert werden.

Download:

```text
https://nodejs.org
```

Bei der Installation sollte `Add to PATH` aktiviert sein. Danach PowerShell schließen und neu öffnen.

### Ollama optional prüfen

```powershell
ollama --version
```

Ollama ist nur nötig, wenn die generative RAG-Erklärung mit lokalem LLM verwendet werden soll. Ohne Ollama funktioniert das Projekt trotzdem mit einer stabilen RAG-Fallback-Erklärung.

---

## 5. Installation und Start

Die folgenden Befehle verwenden allgemein formulierte Pfade. Ersetze `PROJECT_ROOT` durch den Ordner, in dem das Projekt entpackt wurde.

Beispiel:

```text
PROJECT_ROOT = Pfad/zum/Projektordner/game_recommender_nlp_project_react_hf_v4_nav_rag
```

### 5.1 Backend vorbereiten

```powershell
cd "PFAD_ZUM_PROJEKTORDNER"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Falls `python` nicht erkannt wird:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 5.2 Pipeline ausführen

```powershell
python run_pipeline.py
```

Die Pipeline erstellt beziehungsweise prüft die benötigten Artefakte:

- bereinigte Spieldaten
- gelabelte Trainingsdaten
- Logistic-Regression-Modell
- Sentence-Transformer-Embeddings
- BM25-Index
- Feedback-Datei
- Smoke-Test

Beim ersten Durchlauf wird das Sentence-Transformer-Modell heruntergeladen. Das kann einige Minuten dauern.

### 5.3 Backend starten

```powershell
python -m uvicorn backend.main:app --reload
```

Backend läuft dann unter:

```text
http://127.0.0.1:8000
```

API-Dokumentation:

```text
http://127.0.0.1:8000/docs
```

### 5.4 Frontend starten

Öffne ein zweites Terminal.

```powershell
cd "PFAD_ZUM_PROJEKTORDNER\frontend"
npm install
npm run dev
```

Die Webseite läuft dann unter:

```text
http://127.0.0.1:5173
```

---

## 6. Schneller Start mit Skripten

Im Projekt liegen zusätzliche Startskripte.

### Backend über PowerShell

```powershell
.\start_backend.ps1
```

### Frontend über PowerShell

```powershell
.\start_frontend.ps1
```

### Backend über Batch

```cmd
start_backend.bat
```

### Frontend über Batch

```cmd
start_frontend.bat
```

Hinweis: Falls PowerShell-Skripte blockiert werden, kann einmalig ausgeführt werden:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 7. Verwendetes Hugging-Face-/Sentence-Transformer-Modell

Für die semantische Suche wird folgendes Modell verwendet:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Im Code steht es unter anderem in:

```text
src/build_sentence_embeddings.py
src/recommender.py
```

Dieses Modell erzeugt Satz- und Text-Embeddings. Dadurch können Nutzereingaben und Spielbeschreibungen semantisch verglichen werden.

Beispiel:

```text
Nutzereingabe:
"Ich suche ein düsteres Sci-Fi-Spiel mit Story."

Spielbeschreibung:
"A narrative science-fiction adventure with atmospheric exploration."
```

Auch wenn nicht exakt dieselben Wörter vorkommen, kann das Modell eine inhaltliche Nähe erkennen.

Beim ersten Durchlauf lädt `sentence-transformers` das Modell automatisch über den Hugging Face Hub herunter und speichert es lokal im Cache. Eine Warnung zu unauthentifizierten Hugging-Face-Requests ist normalerweise unkritisch. Ein Token ist nur nötig, wenn höhere Downloadlimits oder private Modelle verwendet werden sollen.

---

## 8. Logik der Suche

Die Hauptlogik liegt in:

```text
src/recommender.py
```

Die zentrale Funktion ist:

```python
def recommend(
    self,
    user_query: str,
    selected_genres=None,
    selected_categories=None,
    selected_tags=None,
    max_price=None,
    min_rating=0.0,
    top_n=10,
    use_feedback=True,
):
```

### Ablauf

1. Nutzereingabe wird empfangen.
2. Sentence Transformer erzeugt ein Query-Embedding.
3. Das Query-Embedding wird mit allen Spiel-Embeddings verglichen.
4. BM25 berechnet Keyword-Treffer.
5. Filter werden angewendet.
6. Eine Kandidatenmenge wird gebildet.
7. Der Logistic-Regression-Klassifikator berechnet eine Passwahrscheinlichkeit.
8. Rating, Popularität und Filterbonus werden ergänzt.
9. Feedback-Anpassung wird addiert.
10. Die besten Spiele werden sortiert zurückgegeben.

---

## 9. Sentence Transformers für semantische Suche

In `src/build_sentence_embeddings.py` werden alle Spieltexte in Embeddings umgewandelt.

```python
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True)
```

In `src/recommender.py` wird die Nutzereingabe ebenfalls eingebettet:

```python
query_vector = self.embedding_model.encode([query], normalize_embeddings=True)
semantic_similarity = np.dot(self.vectors, query_vector[0])
```

Da die Embeddings normalisiert sind, entspricht das Skalarprodukt praktisch einer Cosine Similarity.

---

## 10. Okapi BM25

BM25 ergänzt die semantische Suche durch konkrete Keyword-Treffer.

Beispiel:

```text
Suche: "co-op zombie survival"
```

BM25 gibt Spielen einen Bonus, wenn konkrete Begriffe wie `co-op`, `zombie` oder `survival` im Spieltext vorkommen.

Im Code:

```python
bm25_raw = self.bm25.score(query)
bm25_score = minmax_normalize(bm25_raw)
```

BM25 ist besonders hilfreich, wenn der Nutzer sehr konkrete Begriffe verwendet.

---

## 11. Supervised Learning mit gelabelten Daten

Die Datei `data/labeled_manual.csv` wird zum Trainieren des eigenen Klassifikationsmodells genutzt.

Die Trainingslogik liegt in:

```text
src/train_classifier.py
```

Aus der gelabelten CSV wird ein Trainingstext gebaut:

```python
df["input_text"] = (
    "User preference: " + df["user_preference"].fillna("") +
    " Game candidate: " + df["game_text"].fillna("")
)
```

Das Ziel ist:

```python
y = df["label"]
```

Das Modell lernt:

```text
Nutzerpräferenz + Spieltext → passt / passt nicht
```

Verwendet wird:

- TF-IDF für Textfeatures
- Logistic Regression für binäre Klassifikation

Das trainierte Modell wird gespeichert als:

```text
artifacts/preference_classifier.pkl
```

In der App wird später nicht mehr direkt die CSV genutzt, sondern das trainierte Modell. Es berechnet für jedes Kandidatenspiel:

```text
classifier_probability
```

Diese Wahrscheinlichkeit fließt in den finalen Score ein.

---

## 12. Finaler Empfehlungsscore

Der finale Score kombiniert mehrere Signale:

```python
df["base_score"] = (
    0.28 * df["text_similarity_score"]
    + 0.27 * df["classifier_probability"]
    + 0.15 * df["bm25_score"]
    + 0.12 * df["bayesian_rating"]
    + 0.06 * df["popularity_score"]
    + 0.07 * df["filter_bonus"]
)
```

Bedeutung:

| Komponente | Gewicht | Bedeutung |
|---|---:|---|
| Sentence-Transformer-Ähnlichkeit | 28 % | Semantische Nähe zwischen Nutzereingabe und Spiel |
| Klassifikator-Wahrscheinlichkeit | 27 % | Gelerntes `passt / passt nicht` |
| BM25 | 15 % | konkrete Keyword-Treffer |
| Bayesian Rating | 12 % | Bewertungsqualität mit Stabilisierung |
| Popularität | 6 % | Anzahl Bewertungen / Bekanntheit |
| Filterbonus | 7 % | Treffer bei Genre, Kategorie und Tags |

Danach wird optional Feedback addiert:

```python
df["final_score"] = (df["base_score"] + df["feedback_adjustment"]).clip(0, 1.2)
```

---

## 13. Feedback-System

Nutzer können Empfehlungen mit Daumen hoch oder Daumen runter bewerten.

Gespeichert wird in:

```text
artifacts/feedback_events.csv
```

Gespeichert werden unter anderem:

- Suchanfrage
- Spiel-ID
- Spielname
- Feedbackwert `+1` oder `-1`
- Rang
- Score
- Similarity
- BM25
- Klassifikator-Wahrscheinlichkeit

Die Feedback-Logik liegt in:

```text
src/feedback.py
```

Der Einfluss wird begrenzt:

```python
adjustment = (
    0.08 * np.tanh(query_values / 3.0)
    + 0.04 * np.tanh(global_values / 6.0)
)
```

Das bedeutet:

- Feedback zur gleichen Suche wirkt stärker.
- Allgemeines Feedback zu einem Spiel wirkt schwächer.
- `tanh()` verhindert, dass Feedback das Ranking komplett dominiert.

---

## 14. Dynamisches Clustering

Das Clustering passiert nach einer Suche und basiert auf den aktuellen Top-Ergebnissen.

Die Logik liegt in:

```text
src/dynamic_clustering.py
```

Die React-Seite `Clustering` nutzt die Top-100-Kandidaten der letzten Suche.

Der Nutzer kann verschiedene Perspektiven auswählen:

1. Inhalt & Genre
2. Spielgefühl / Stimmung
3. Spielmodus
4. Beliebtheit & Bewertung
5. Preis & Spielzeit
6. Geschätzter Grafikaufwand

Das Clustering ist dynamisch:

- Neue Suche → neue Top-100-Kandidaten
- Neue Perspektive → neue Merkmalsbasis
- Clusteranzahl wird sinnvoll angepasst
- Clusterlabels werden nutzerfreundlich benannt

Beispielhafte Cluster:

```text
Sci-Fi Adventure
Puzzle & Mystery
Horror Survival
Relaxing & Cozy
Co-op Games
Budget Picks
Grafisch eher aufwendig
```

---

## 15. RAG-Erklärung

Die Spieldetails enthalten eine RAG-Erklärung.

RAG steht für:

```text
Retrieval-Augmented Generation
```

Im Projekt bedeutet das:

1. Das System wählt ein empfohlenes Spiel aus.
2. Ähnliche Spiele werden als zusätzlicher Kontext abgerufen.
3. Aus Spielinformationen, Scores, Tags, Genres und ähnlichen Spielen wird eine Erklärung erzeugt.

Die Logik liegt in:

```text
src/rag_answer.py
backend/main.py → /api/explain
```

### Standardmodus

Ohne zusätzliches LLM wird eine stabile, datenbasierte RAG-Fallback-Erklärung erzeugt.

Vorteil:

- funktioniert immer
- keine Zusatzinstallation nötig
- keine externen API-Kosten
- Erklärung basiert auf echten Projektdaten

### Optionaler generativer Modus mit Ollama

Wenn ein lokales LLM genutzt werden soll, kann Ollama aktiviert werden.

Modell installieren:

```powershell
ollama pull llama3.2
```

Backend mit Ollama starten:

```powershell
$env:RAG_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.2"
python -m uvicorn backend.main:app --reload
```

Dann generiert das lokale LLM eine natürlichere Erklärung aus dem Retrieval-Kontext.

Falls Ollama nicht erreichbar ist, fällt das System automatisch auf den stabilen Fallback zurück.

---

## 16. Frontend-Navigation

Die React-Webseite ist in mehrere Bereiche aufgeteilt:

### Suche & Empfehlungen

- Freitextsuche
- Filter für Genres, Kategorien und Tags
- Preis- und Bewertungsfilter
- Top-Empfehlungen
- Score-Komponenten
- Feedback-Buttons
- Spieldetails und RAG-Erklärung

### Clustering

- dynamisches Clustering der letzten Suchergebnisse
- Perspektivenauswahl
- Clusterdiagramm
- Clusterkarten
- Detailansichten und ähnliche Spiele

### Feedback-Auswertung

- Anzahl Feedbacks
- positive und negative Bewertungen
- Positivquote
- zuletzt bewertete Spiele
- Top positiv / negativ bewertete Spiele

---

## 17. Tests

Unit Tests liegen im Ordner:

```text
tests/
```

Ausführen:

```powershell
pytest -q
```

Die Tests prüfen zentrale Backend-Funktionen und API-nahe Logik.

---

## 18. Häufige Fehler und Lösungen

### `npm` wird nicht erkannt

Node.js ist nicht installiert oder nicht im PATH.

Lösung:

1. Node.js LTS installieren
2. PowerShell schließen und neu öffnen
3. prüfen:

```powershell
node -v
npm -v
```

### Backend zeigt `Not Found`

Die eigentliche Webseite läuft nicht auf Port 8000, sondern auf Port 5173.

```text
Backend:  http://127.0.0.1:8000
Frontend: http://127.0.0.1:5173
```

### Hugging Face Warnung wegen Token

Eine Meldung wie diese ist meist unkritisch:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

Das Modell kann trotzdem heruntergeladen werden. Ein Token ist nur für höhere Limits oder private Modelle nötig.

### Suche lädt lange beim ersten Start

Beim ersten Start werden Modelle und Artefakte geladen. Danach ist die Suche schneller.

### Weiße Seite im Frontend

Browser hart neu laden:

```text
STRG + F5
```

Falls weiterhin leer:

1. Frontend-Terminal auf Fehlermeldungen prüfen
2. Backend-Terminal auf API-Fehler prüfen
3. Browser Developer Console öffnen

---

## 19. Wichtige Befehle im Überblick

### Backend komplett vorbereiten

```powershell
cd "PFAD_ZUM_PROJEKTORDNER"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
python -m uvicorn backend.main:app --reload
```

### Frontend starten

```powershell
cd "PFAD_ZUM_PROJEKTORDNER\frontend"
npm install
npm run dev
```

### Webseite öffnen

```text
http://127.0.0.1:5173
```

### API-Dokumentation öffnen

```text
http://127.0.0.1:8000/docs
```

### Tests ausführen

```powershell
pytest -q
```

### Optional Ollama-RAG

```powershell
ollama pull llama3.2
$env:RAG_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.2"
python -m uvicorn backend.main:app --reload
```

---

## 20. Kurzfazit

Das Projekt ist ein hybrides Recommendation System. Es kombiniert semantische Suche mit Sentence Transformers, BM25-Keyword-Retrieval, ein eigenes supervised Klassifikationsmodell, Bewertungsdaten, Filterlogik, Nutzerfeedback, dynamisches Clustering und RAG-Erklärungen.

Dadurch entsteht ein System, das nicht nur Spiele empfiehlt, sondern die Empfehlungen auch nachvollziehbar erklärt und interaktiv durch Feedback und Clusteransichten erweitert.
