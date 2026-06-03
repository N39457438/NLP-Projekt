# Game Recommendation System – React + FastAPI + Sentence Transformers

Dieses Projekt ist ein NLP-basiertes Game Recommendation System für Steam-Spiele. Nutzerinnen und Nutzer geben eine freie Spielpräferenz ein, setzen optional Filter und erhalten passende, erklärbare Empfehlungen. Zusätzlich bietet die Anwendung ein dynamisches Clustering der Suchergebnisse, ein Feedback-System sowie eine RAG-basierte Erklärung der Empfehlungen.

Die Anwendung besteht aus einem Python-Backend mit FastAPI und einem React-Frontend. Die semantische Suche nutzt Sentence Transformers von Hugging Face, während weitere Komponenten wie BM25, ein eigener Klassifikator, Bewertungsdaten, Filter und Nutzerfeedback in das finale Ranking einfließen.

---

## 1. Ziel des Projekts

Ziel ist es, aus einer freien Nutzereingabe wie:

```text
Ich suche ein düsteres Sci-Fi-Spiel mit Story, Atmosphäre und Rätseln.
```

passende Steam-Spiele zu finden. Dabei soll das System nicht nur einfache Filter anwenden, sondern den Inhalt der Spielbeschreibungen semantisch verstehen, konkrete Suchbegriffe berücksichtigen, Bewertungen einbeziehen und die Ergebnisse verständlich erklären.

Das Projekt zeigt damit mehrere zentrale NLP- und ML-Komponenten:

- semantische Suche mit Sentence Transformers
- Keyword-Retrieval mit Okapi BM25
- überwachtes Lernen mit Logistic Regression
- dynamisches Clustering der aktuellen Top-Ergebnisse
- Feedback-basierte Anpassung zukünftiger Rankings
- RAG-Erklärung mit Retrieval-Kontext und optionalem lokalem LLM
- moderne Web-App mit React und FastAPI

---

## 2. Grundidee der Suche

Die Suche arbeitet hybrid. Das bedeutet, dass nicht nur ein einzelnes Verfahren verwendet wird, sondern mehrere Signale kombiniert werden.

Ablauf:

1. Der Nutzer gibt einen Suchtext ein.
2. Das Backend wandelt die Eingabe mit einem Sentence Transformer in ein Embedding um.
3. Alle Spiele werden über ihre vorberechneten Embeddings mit der Suchanfrage verglichen.
4. Okapi BM25 bewertet zusätzlich konkrete Keyword-Treffer.
5. Ein trainierter Logistic-Regression-Klassifikator berechnet, ob ein Spiel zur Anfrage passt.
6. Genre-, Kategorie-, Tag-, Preis- und Bewertungsfilter werden berücksichtigt.
7. Bewertungsqualität, Popularität und gespeichertes Nutzerfeedback fließen in den Score ein.
8. Die besten Empfehlungen werden an das React-Frontend zurückgegeben.
9. Aus den Top-100-Ergebnissen kann dynamisch ein Cluster-Dashboard erstellt werden.
10. Für einzelne Spiele wird eine RAG-Erklärung erzeugt.

---

## 3. Verwendete Daten

Das Projekt nutzt drei zentrale CSV-Dateien im Ordner `data/`:

### `steam.csv`

Enthält strukturierte Metadaten zu Steam-Spielen, unter anderem:

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

Enthält Beschreibungstexte der Spiele:

- `steam_appid`
- `detailed_description`
- `about_the_game`
- `short_description`

Die Dateien werden über diese IDs verbunden:

```text
steam.csv: appid
steam_description_data.csv: steam_appid
```

### `labeled_manual.csv`

Enthält gelabelte Trainingsbeispiele für den Klassifikator:

```text
user_preference, appid, game_name, game_text, label
```

Bedeutung des Labels:

```text
1 = Spiel passt zur Nutzerpräferenz
0 = Spiel passt nicht zur Nutzerpräferenz
```

Diese Datei wird beim Training des supervised ML-Modells verwendet.

---

## 4. Projektstruktur

Die wichtigsten Ordner und Dateien:

```text
PROJECT_ROOT/
├── backend/
│   └── main.py
├── frontend/
│   ├── package.json
│   ├── index.html
│   └── src/
├── src/
│   ├── prepare_data.py
│   ├── train_classifier.py
│   ├── build_sentence_embeddings.py
│   ├── bm25.py
│   ├── recommender.py
│   ├── feedback.py
│   ├── dynamic_clustering.py
│   ├── rag_answer.py
│   └── evaluate.py
├── data/
│   ├── steam.csv
│   ├── steam_description_data.csv
│   └── labeled_manual.csv
├── artifacts/
│   ├── games_processed.csv
│   ├── game_sentence_embeddings.npy
│   ├── bm25_index.pkl
│   ├── preference_classifier.pkl
│   └── feedback_events.csv
├── tests/
├── requirements.txt
├── run_pipeline.py
├── rebuild_all.ps1
├── start_backend.ps1
├── start_frontend.ps1
└── README.md
```

---

## 5. Technischer Stack

### Backend

- Python
- FastAPI
- Uvicorn
- pandas / NumPy
- scikit-learn
- sentence-transformers
- joblib

### Frontend

- React
- Vite
- JavaScript
- lucide-react für Icons
- CSS für Dashboard-Layout

### Optional für generatives RAG

- Ollama
- lokales LLM, zum Beispiel `llama3.2`, `mistral` oder `phi3`

---

## 6. Verwendetes Hugging-Face-Modell

Für die semantische Suche wird dieses Sentence-Transformer-Modell verwendet:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Im Code ist es in `src/recommender.py` und `src/build_sentence_embeddings.py` als Standardmodell hinterlegt.

Dieses Modell wandelt Nutzereingaben und Spieltexte in Vektoren um. Dadurch kann das System erkennen, ob eine Suchanfrage und ein Spiel inhaltlich ähnlich sind, auch wenn nicht exakt dieselben Wörter vorkommen.

Beispiel:

```text
Nutzereingabe: story-driven sci-fi game
Spielbeschreibung: futuristic single-player narrative adventure
```

Auch ohne identische Wörter kann die semantische Ähnlichkeit hoch sein.

Beim ersten Start wird das Modell automatisch heruntergeladen und lokal gecacht. Eine Anmeldung bei Hugging Face ist normalerweise nicht nötig. Falls Warnungen zu unauthentifizierten Requests erscheinen, sind diese meist unkritisch. Optional kann ein `HF_TOKEN` gesetzt werden, um höhere Download-Limits zu erhalten.

---

## 7. Erklärung der wichtigsten Methoden

### 7.1 Sentence Transformers

Sentence Transformers erzeugen semantische Text-Embeddings. Im Projekt werden damit sowohl Spieltexte als auch Nutzereingaben in Vektoren umgewandelt. Anschließend wird per Skalarprodukt beziehungsweise Cosine Similarity gemessen, wie ähnlich die Suchanfrage zu den Spielen ist.

Im Projekt ersetzt dieser Ansatz die frühere LSA-Variante. Der Vorteil ist, dass semantische Zusammenhänge besser erkannt werden.

### 7.2 Okapi BM25

BM25 ist ein klassischer Suchalgorithmus aus dem Information Retrieval. Er bewertet, wie gut konkrete Suchbegriffe in einem Dokument vorkommen.

BM25 ist besonders nützlich bei Begriffen wie:

- `zombie`
- `racing`
- `co-op`
- `VR`
- `farming`
- `horror`

Während Sentence Transformers eher die semantische Bedeutung erfassen, stärkt BM25 konkrete Keyword-Treffer.

### 7.3 Logistic Regression

Das Projekt nutzt ein eigenes supervised ML-Modell. Dieses Modell wird mit der Datei `labeled_manual.csv` trainiert.

Trainingslogik:

```text
Input: Nutzerpräferenz + Spieltext
Output: passt / passt nicht
```

Das Modell gibt später für jedes Spiel eine Wahrscheinlichkeit aus:

```text
classifier_probability = Wahrscheinlichkeit, dass das Spiel zur Suche passt
```

Diese Wahrscheinlichkeit fließt in den finalen Score ein.

### 7.4 Rating-Score und Bayesian Rating

Aus positiven und negativen Steam-Bewertungen wird ein Bewertungswert berechnet. Zusätzlich wird ein Bayesian Rating genutzt, damit Spiele mit sehr wenigen Bewertungen nicht zu stark bevorzugt werden.

Beispiel:

```text
Ein Spiel mit 10 positiven und 0 negativen Bewertungen ist nicht automatisch besser
als ein Spiel mit 50.000 positiven und 1.000 negativen Bewertungen.
```

### 7.5 Feedback-System

Nutzer können Empfehlungen mit Daumen hoch oder Daumen runter bewerten. Das Feedback wird in `artifacts/feedback_events.csv` gespeichert.

Es gibt zwei Arten von Einfluss:

- Feedback zur gleichen Suchanfrage wirkt stärker.
- Allgemeines Feedback zu einem Spiel wirkt schwächer.

Der Einfluss wird begrenzt, damit einzelne Klicks das Ranking nicht komplett dominieren.

### 7.6 Dynamisches Clustering

Das Clustering wird nicht global für alle Spiele berechnet, sondern nach jeder Suche aus den aktuellen Top-100-Ergebnissen.

Dadurch passen die Cluster direkt zur aktuellen Nutzeranfrage.

Verfügbare Perspektiven:

- Inhalt & Genre
- Spielgefühl / Stimmung
- Spielmodus
- Beliebtheit & Bewertung
- Preis & Spielzeit
- Geschätzter Grafikaufwand

Je nach Perspektive werden unterschiedliche Merkmale verwendet.

### 7.7 RAG-Erklärung

RAG steht für Retrieval-Augmented Generation.

Im Projekt bedeutet das:

1. Retrieval: Das System sucht das ausgewählte Spiel und ähnliche Spiele als Kontext.
2. Augmentation: Diese Informationen werden als Kontext gesammelt.
3. Generation: Daraus wird eine Erklärung erzeugt.

Standardmäßig wird eine stabile, deterministische RAG-Erklärung erzeugt. Optional kann ein lokales LLM über Ollama genutzt werden, um natürlichere Texte zu generieren.

---

## 8. Ranking-Logik

Der finale Score kombiniert mehrere Werte:

```text
final_score =
    semantische Ähnlichkeit
  + Klassifikator-Wahrscheinlichkeit
  + BM25-Score
  + Bewertungsqualität
  + Popularität
  + Filterbonus
  + Feedback-Anpassung
```

Im Code befindet sich diese Logik in:

```text
src/recommender.py
```

Die wichtigsten Komponenten:

- `text_similarity_score`: semantische Ähnlichkeit über Sentence Transformers
- `classifier_probability`: ML-Wahrscheinlichkeit aus Logistic Regression
- `bm25_score`: Keyword-Treffer über BM25
- `bayesian_rating`: stabilisierte Bewertungsqualität
- `popularity_score`: Popularität auf Basis der Bewertungs- und Besitzerzahlen
- `filter_bonus`: Treffer bei Genre, Kategorie und Tags
- `feedback_adjustment`: Anpassung durch bisheriges Nutzerfeedback

---

## 9. Voraussetzungen zum Starten

Benötigt werden:

### Pflicht

- Python 3.10 oder neuer
- Node.js LTS
- npm
- Internetverbindung beim ersten Start, damit das Sentence-Transformer-Modell geladen werden kann

### Optional

- Ollama für generatives RAG mit lokalem LLM
- Git, falls das Projekt über Git verwaltet wird

Prüfen der Installation:

```powershell
python --version
node -v
npm -v
```

Falls `python` nicht gefunden wird, kann unter Windows oft `py` verwendet werden.

---

## 10. Installation und Start unter Windows

### 10.1 Projekt entpacken

ZIP-Datei entpacken und in den Projektordner wechseln:

```powershell
cd "<PFAD_ZUM_PROJEKT>"
```

Beispiel allgemein:

```powershell
cd "C:\Pfad\zum\game_recommender_nlp_project_react_hf_v4_nav_rag"
```

### 10.2 Virtuelle Python-Umgebung erstellen

```powershell
python -m venv .venv
```

Aktivieren:

```powershell
.\.venv\Scripts\activate
```

Falls PowerShell Skripte blockiert:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Danach erneut aktivieren.

### 10.3 Python-Abhängigkeiten installieren

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 10.4 Pipeline ausführen

```powershell
python run_pipeline.py
```

Dieser Schritt erstellt bzw. prüft die benötigten Artefakte:

- bereinigte Spieldaten
- trainierter Klassifikator
- Sentence-Transformer-Embeddings
- BM25-Index
- Feedback-Datei
- Smoke-Test

Beim ersten Durchlauf kann das Herunterladen des Hugging-Face-Modells etwas dauern.

### 10.5 Backend starten

```powershell
python -m uvicorn backend.main:app --reload
```

Das Backend läuft dann unter:

```text
http://127.0.0.1:8000
```

API-Dokumentation:

```text
http://127.0.0.1:8000/docs
```

### 10.6 Frontend starten

In einem zweiten Terminal:

```powershell
cd "<PFAD_ZUM_PROJEKT>\frontend"
npm install
npm run dev
```

Das Frontend läuft dann unter:

```text
http://127.0.0.1:5173
```

Diese Adresse ist die eigentliche Webseite.

---

## 11. Installation und Start unter macOS/Linux

```bash
cd "<PFAD_ZUM_PROJEKT>"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
python -m uvicorn backend.main:app --reload
```

Zweites Terminal:

```bash
cd "<PFAD_ZUM_PROJEKT>/frontend"
npm install
npm run dev
```

Dann öffnen:

```text
http://127.0.0.1:5173
```

---

## 12. Generatives RAG mit Ollama aktivieren

Standardmäßig funktioniert die RAG-Erklärung ohne zusätzliches LLM. Für generatives RAG kann Ollama verwendet werden.

### 12.1 Ollama installieren

Ollama von der offiziellen Webseite installieren:

```text
https://ollama.com
```

### 12.2 Modell herunterladen

Beispiel:

```powershell
ollama pull llama3.2
```

Alternative kleinere Modelle:

```powershell
ollama pull phi3
ollama pull mistral
```

### 12.3 Backend mit Ollama-RAG starten

```powershell
cd "<PFAD_ZUM_PROJEKT>"
.\.venv\Scripts\activate
$env:RAG_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.2"
python -m uvicorn backend.main:app --reload
```

Frontend wie gewohnt starten:

```powershell
cd "<PFAD_ZUM_PROJEKT>\frontend"
npm run dev
```

Wenn Ollama nicht erreichbar ist, fällt das System automatisch auf die deterministische RAG-Erklärung zurück.

---

## 13. Bedienung der Web-App

Die Oberfläche ist in drei Bereiche aufgeteilt:

### Suche & Empfehlungen

- Suchtext eingeben
- Filter setzen
- Empfehlungen berechnen
- Spieldetails öffnen
- Feedback mit Daumen hoch/runter geben
- RAG-Erklärung ansehen

### Clustering

- basiert auf den Top-100-Ergebnissen der letzten Suche
- Cluster-Perspektive auswählen
- Cluster visuell erkunden
- Clusterkarten öffnen
- ähnliche Spiele betrachten

### Feedback-Auswertung

- Gesamtanzahl der Feedbacks
- positive und negative Bewertungen
- zuletzt bewertete Spiele
- häufig positiv/negativ bewertete Spiele

---

## 14. Wichtige API-Endpunkte

Backend-Endpunkte:

```text
GET  /api/health
GET  /api/filters
POST /api/recommend
POST /api/feedback
GET  /api/feedback/summary
POST /api/clusters
GET  /api/similar/{appid}
POST /api/explain
POST /api/cluster/similar
```

Die automatische API-Dokumentation ist erreichbar unter:

```text
http://127.0.0.1:8000/docs
```

---

## 15. Tests ausführen

```powershell
pytest -q
```

Die Tests prüfen zentrale Backend-Funktionen und sollen sicherstellen, dass wichtige Komponenten grundsätzlich funktionieren.

---

## 16. Häufige Probleme und Lösungen

### Problem: `npm` wird nicht erkannt

Node.js ist nicht installiert oder nicht im PATH.

Lösung:

1. Node.js LTS installieren.
2. Terminal neu öffnen.
3. Prüfen:

```powershell
node -v
npm -v
```

### Problem: Backend zeigt nur JSON im Browser

Das ist normal. `http://127.0.0.1:8000` ist nur das Backend.

Die Webseite läuft unter:

```text
http://127.0.0.1:5173
```

### Problem: Hugging-Face-Warnung zu unauthentifizierten Requests

Das ist meistens unkritisch. Das Modell wird trotzdem geladen. Optional kann ein Hugging-Face-Token gesetzt werden, um höhere Limits zu erhalten.

### Problem: Suche lädt beim ersten Mal lange

Beim ersten Start werden Modelle geladen. Danach ist die Suche schneller, weil Backend und Modell im Speicher bleiben.

### Problem: Artefakte fehlen

Pipeline erneut ausführen:

```powershell
python run_pipeline.py
```

Oder komplett neu bauen:

```powershell
.\rebuild_all.ps1
```

### Problem: Feedback soll zurückgesetzt werden

Datei leeren oder löschen:

```text
artifacts/feedback_events.csv
```

Danach wird sie beim nächsten Start neu angelegt.

---

## 17. Wissenschaftliche Einordnung

Das System ist ein hybrides Recommendation System. Es kombiniert:

- Content-Based Recommendation über Spieltexte, Genres, Kategorien und Tags
- semantische Suche mit Sentence Transformers
- Information Retrieval mit BM25
- supervised Learning mit Logistic Regression
- dynamisches Clustering mit K-Means und regelbasierten Perspektiven
- Feedback-basiertes Re-Ranking
- RAG-Erklärungen auf Basis von Retrieval-Kontext

Dadurch ist das System nicht nur eine einfache Filteranwendung, sondern ein erklärbares NLP- und Machine-Learning-Projekt.

---

## 18. Grenzen des Systems

- Es liegen keine echten Nutzerhistorien vor.
- Das Feedback ist lokal und nutzerübergreifend gespeichert.
- Der Rating-Score basiert auf positiven und negativen Bewertungen, nicht auf Reviewtexten.
- Das optionale generative RAG hängt von Ollama und dem lokalen Modell ab.
- Das Hugging-Face-Modell liefert gute semantische Embeddings, wurde aber nicht speziell auf Steam-Spiele trainiert.

---

## 19. Kurzfassung für Präsentation oder Abgabe

Das Projekt entwickelt ein erklärbares Game Recommendation System für Steam-Spiele. Die Suchanfrage des Nutzers wird mit Sentence Transformers semantisch mit Spielbeschreibungen verglichen. Zusätzlich werden BM25-Keyword-Treffer, ein eigener Logistic-Regression-Klassifikator, Bewertungsdaten, Filter und Nutzerfeedback kombiniert. Die Ergebnisse werden in einer React-Web-App angezeigt. Dynamisches Clustering hilft, die Top-100-Ergebnisse aus verschiedenen Perspektiven zu erkunden. Eine RAG-Komponente erklärt, warum ein Spiel empfohlen wurde.

