# NLP-Projekt
# Game Recommendation System – NLP & Machine Learning

Dieses Projekt implementiert ein **Game Recommendation System** auf Basis klassischer NLP- und Machine-Learning-Verfahren. Nutzer können in einer Streamlit-Web-App ihre Videospielpräferenzen als Freitext eingeben und zusätzlich Genres, Kategorien, Preisgrenzen und Mindestbewertungen auswählen. Das System empfiehlt anschließend passende Steam-Spiele und erklärt die Empfehlungen nachvollziehbar.

Das Projekt wurde bewusst **ohne vortrainierte Transformer-Modelle wie Hugging Face oder SentenceTransformers** umgesetzt. Stattdessen werden klassische, gut erklärbare Verfahren wie **TF-IDF**, **SVD/LSA**, **Logistic Regression**, **Cosine Similarity** und **K-Means Clustering** verwendet.

---

## Projektziel

Ziel des Projekts ist die Entwicklung eines hybriden Empfehlungssystems, das freie Nutzereingaben mit strukturierten und unstrukturierten Spieldaten kombiniert.

Ein Nutzer kann beispielsweise eingeben:

```text
I want a dark sci-fi singleplayer game with story and action.
```

Das System analysiert diese Eingabe und vergleicht sie mit den Spielbeschreibungen, Genres, Kategorien und Tags aus dem Steam-Datensatz. Zusätzlich wird ein trainiertes Klassifikationsmodell verwendet, das bewertet, ob ein Spiel zur Nutzerpräferenz passt.

---

## Verwendete Daten

Das Projekt verwendet zwei Steam-Datensätze:

### `steam.csv`

Diese Datei enthält strukturierte Spieldaten, unter anderem:

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
- `owners`
- `price`

### `steam_description_data.csv`

Diese Datei enthält textuelle Spielinformationen:

- `steam_appid`
- `detailed_description`
- `about_the_game`
- `short_description`

Die beiden Dateien werden über folgende Schlüssel verbunden:

```text
steam.csv: appid
steam_description_data.csv: steam_appid
```

---

## Projektstruktur

```text
game_recommender_nlp_project_no_hf/
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── steam.csv
│   └── steam_description_data.csv
│
├── artifacts/
│   ├── games_processed.csv
│   ├── labeled_training_data.csv
│   ├── tfidf_vectorizer.pkl
│   ├── lsa_model.pkl
│   ├── game_lsa_embeddings.npy
│   ├── preference_classifier.pkl
│   └── kmeans_model.pkl
│
└── src/
    ├── prepare_data.py
    ├── create_labels.py
    ├── train_classifier.py
    ├── build_lsa_embeddings.py
    ├── clustering.py
    ├── recommender.py
    ├── rag_answer.py
    └── evaluate.py
```

---

## Installation

### 1. Projektordner öffnen

In PowerShell oder im VS-Code-Terminal:

```powershell
cd "C:\Users\nikla\OneDrive\Dokumente\Hochschule\6. Semester\NLP\game_recommender_nlp_project_no_hf"
```

### 2. Virtuelle Umgebung erstellen

```powershell
python -m venv .venv
```

Falls `python` nicht funktioniert:

```powershell
py -m venv .venv
```

### 3. Virtuelle Umgebung aktivieren

```powershell
.\.venv\Scripts\activate
```

Danach sollte im Terminal links `(.venv)` stehen.

### 4. Abhängigkeiten installieren

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Projekt ausführen

### 1. Komplette Pipeline starten

```powershell
python run_pipeline.py
```

Diese Datei führt alle wichtigen Schritte automatisch nacheinander aus:

1. Daten laden und bereinigen
2. Daten zusammenführen
3. Trainingslabels erzeugen
4. Klassifikator trainieren
5. TF-IDF- und LSA-Vektoren erzeugen
6. K-Means-Clustering durchführen
7. Evaluation ausgeben

Nach erfolgreicher Ausführung befinden sich alle erzeugten Dateien im Ordner `artifacts/`.

### 2. Streamlit-App starten

```powershell
python -m streamlit run app.py
```

Danach öffnet sich die Web-App normalerweise automatisch im Browser. Falls nicht, kann die angezeigte lokale URL geöffnet werden, zum Beispiel:

```text
http://localhost:8501
```

---

## Funktionsweise des Systems

Das Empfehlungssystem besteht aus mehreren Komponenten, die gemeinsam den finalen Empfehlungsscore berechnen.

### 1. Datenaufbereitung

In `src/prepare_data.py` werden die beiden CSV-Dateien geladen und über `appid` und `steam_appid` verbunden. Anschließend werden HTML-Tags aus den Beschreibungen entfernt, fehlende Werte behandelt und ein gemeinsamer Spieltext erzeugt.

Der kombinierte Text besteht aus:

- Spielname
- Genres
- Kategorien
- SteamSpy-Tags
- Kurzbeschreibung
- ausführlicher Beschreibung

Dieser Text wird als zentrale Grundlage für die NLP-Modelle verwendet.

Zusätzlich werden zwei Scores berechnet:

```text
rating_score = positive_ratings / (positive_ratings + negative_ratings)
```

Der `rating_score` dient als bewertungsbasierter Sentiment-Proxy.

Außerdem wird ein `popularity_score` berechnet, der die Anzahl der Bewertungen berücksichtigt.

---

## Labeling

Für das supervised Learning wird ein gelabelter Trainingsdatensatz verwendet. Die Struktur lautet:

```text
user_preference, appid, game_name, game_text, label
```

Das Label bedeutet:

```text
1 = Spiel passt zur Nutzerpräferenz
0 = Spiel passt nicht zur Nutzerpräferenz
```

Beispiel:

```text
I want a competitive multiplayer shooter + Counter-Strike → 1
I want a relaxing farming simulation game + Counter-Strike → 0
```

Diese Trainingsdaten ermöglichen es dem Klassifikationsmodell, die Passung zwischen Nutzerpräferenz und Spieltext zu lernen.

---

## Verwendete NLP- und ML-Verfahren

### TF-IDF

TF-IDF steht für **Term Frequency - Inverse Document Frequency**. Es wandelt Texte in numerische Vektoren um. Dabei werden Begriffe höher gewichtet, die für ein bestimmtes Dokument charakteristisch sind, während sehr häufige allgemeine Wörter weniger Gewicht erhalten.

Beispiel:

```text
multiplayer, shooter, fps, puzzle, farming, simulation
```

TF-IDF erkennt, welche Begriffe für ein Spiel besonders relevant sind. Außerdem werden N-Gramme verwendet, sodass auch Wortkombinationen wie `single player`, `open world` oder `first person` berücksichtigt werden.

---

### SVD / LSA

Die TF-IDF-Vektoren sind sehr hochdimensional. Mit **TruncatedSVD** werden diese Vektoren auf weniger Dimensionen reduziert. Dieses Verfahren wird im NLP häufig als **Latent Semantic Analysis** bezeichnet.

Dadurch entstehen kompaktere Spielvektoren, die latente Themen abbilden können, zum Beispiel:

- Shooter / Action / FPS
- Puzzle / Physics / First-Person
- Strategy / Management / Simulation
- RPG / Fantasy / Adventure

Diese LSA-Vektoren werden später für die Ähnlichkeitssuche und das Clustering verwendet.

---

### Cosine Similarity

Die **Cosine Similarity** misst, wie ähnlich zwei Vektoren sind. Im Projekt wird die Nutzereingabe mit allen Spielvektoren verglichen.

Beispiel:

```text
User Query: I want a sci-fi singleplayer shooter.
Game: Half-Life 2
→ hohe Ähnlichkeit
```

Die berechnete Ähnlichkeit heißt im System `semantic_similarity`.

---

### Logistic Regression

Die Logistic Regression ist das supervised ML-Modell des Projekts. Sie bekommt als Eingabe eine Kombination aus Nutzerpräferenz und Spieltext:

```text
User preference: ...
Game: ...
```

Das Modell berechnet anschließend die Wahrscheinlichkeit, dass das Spiel zur Nutzerpräferenz passt:

```text
classifier_probability = P(label = 1)
```

Diese Wahrscheinlichkeit wird in der finalen Empfehlungslogik berücksichtigt.

---

### K-Means Clustering

K-Means ist ein unüberwachtes ML-Verfahren. Es gruppiert ähnliche Spiele anhand ihrer LSA-Vektoren.

Mögliche Cluster können thematisch sein, zum Beispiel:

- Action- und Shooter-Spiele
- Puzzle-Spiele
- Strategie- und Management-Spiele
- RPG- und Fantasy-Spiele
- Casual- und Indie-Spiele

Die Cluster dienen zur Analyse der Spielstruktur und können zusätzlich in der Web-App oder Dokumentation interpretiert werden.

---

## Finaler Empfehlungsscore

Für jedes Spiel werden mehrere Werte berechnet:

- `semantic_similarity`
- `classifier_probability`
- `rating_score`
- `popularity_score`
- `filter_bonus`

Diese Werte werden zu einem finalen Score kombiniert:

```text
final_score =
    0.35 * semantic_similarity
  + 0.35 * classifier_probability
  + 0.15 * rating_score
  + 0.10 * popularity_score
  + 0.05 * filter_bonus
```

Die Spiele werden anschließend nach `final_score` absteigend sortiert. Die besten Spiele werden dem Nutzer als Empfehlung angezeigt.

---

## Streamlit-Web-App

Die Web-App befindet sich in `app.py`.

Der Nutzer kann dort:

- eine Freitextpräferenz eingeben,
- Genres auswählen,
- Kategorien auswählen,
- einen maximalen Preis festlegen,
- eine Mindestbewertung auswählen,
- die Anzahl der Empfehlungen bestimmen.

Für jedes empfohlene Spiel zeigt die App unter anderem:

- Spielname
- Kurzbeschreibung
- Genres
- Kategorien
- Entwickler
- Preis
- Rating-Score
- semantische Ähnlichkeit
- Klassifikator-Wahrscheinlichkeit
- finalen Score
- erklärenden Empfehlungstext

---

## RAG-artige Erklärung

Das Projekt verwendet eine einfache, regelbasierte RAG-Logik.

RAG bedeutet:

```text
Retrieval-Augmented Generation
```

Im Projekt bedeutet das:

1. Das System ruft passende Spiele ab.
2. Es sammelt relevante Informationen zum Spiel.
3. Es erzeugt daraus eine Erklärung.

Es wird kein externes LLM verwendet. Die Erklärung basiert ausschließlich auf den berechneten Scores und den vorhandenen Spieldaten.

Beispiel:

```text
Das Spiel wird empfohlen, weil es eine hohe semantische Ähnlichkeit zur Eingabe besitzt,
vom Klassifikator als passend bewertet wurde und eine gute Nutzerbewertung hat.
```

---

## Evaluation

Die Evaluation prüft unter anderem:

- Genauigkeit des Klassifikators
- Precision, Recall und F1-Score
- Confusion Matrix
- Beispielanfragen mit Top-N-Empfehlungen

Mögliche Testanfragen:

```text
I want a competitive multiplayer shooter.
I want a relaxing farming simulation game.
I want a story-driven sci-fi singleplayer game.
I want a puzzle game with first-person gameplay.
I want a strategy game with management and planning.
```

Die Empfehlungen können anschließend manuell auf Plausibilität geprüft werden.

---

## Vorteile des Ansatzes

- Keine Abhängigkeit von Hugging Face oder Transformer-Modellen
- Klassische NLP-Verfahren sind gut erklärbar
- Kombination aus supervised und unsupervised Learning
- Eigener gelabelter Trainingsdatensatz
- Verständliche Empfehlungslogik
- Streamlit-App als benutzerfreundliche Oberfläche
- RAG-artige Erklärung für transparente Empfehlungen

---

## Grenzen des Systems

- TF-IDF erkennt Synonyme schlechter als moderne Transformer-Modelle
- Die Qualität des Klassifikators hängt stark von den Labels ab
- Es gibt keine echten Nutzerinteraktionsdaten
- Der Rating-Score ist nur ein Proxy für Sentiment
- K-Means-Cluster sind nicht automatisch semantisch benannt
- Empfehlungen hängen stark von den vorhandenen Spielbeschreibungen ab

---

## Typische Fehler und Lösungen

### `python wurde nicht gefunden`

Statt `python` kann `py` verwendet werden:

```powershell
py run_pipeline.py
py -m streamlit run app.py
```

### Virtuelle Umgebung kann nicht aktiviert werden

Wenn PowerShell Skripte blockiert:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Danach erneut aktivieren:

```powershell
.\.venv\Scripts\activate
```

### `No module named streamlit`

Dann sind die Pakete nicht installiert oder die virtuelle Umgebung ist nicht aktiv:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Beispielhafter Komplettstart

```powershell
cd "C:\Users\nikla\OneDrive\Dokumente\Hochschule\6. Semester\NLP\game_recommender_nlp_project_no_hf"

python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python run_pipeline.py
python -m streamlit run app.py
```

---

## Fazit

Das Projekt zeigt, wie ein vollständiges Game Recommendation System mit klassischen NLP- und Machine-Learning-Methoden umgesetzt werden kann. Es kombiniert Textanalyse, Klassifikation, Ähnlichkeitssuche, Clustering, Bewertungsdaten und eine Web-App zu einem nachvollziehbaren hybriden Empfehlungssystem.

Besonders wichtig ist, dass die finale Empfehlung nicht nur auf einem einzelnen Verfahren basiert, sondern mehrere Signale kombiniert:

```text
Textähnlichkeit + gelerntes Passungsmodell + Bewertungen + Popularität + Nutzerfilter
```

Dadurch entsteht ein flexibles und erklärbares System, das für ein NLP-Studienprojekt sehr gut geeignet ist.
