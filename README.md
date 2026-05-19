# ML Model Serving Platform — Plan Projektu

## Opis projektu

Platforma do trenowania, wersjonowania i serwowania modeli ML w kontenerach Docker. Projekt obejmuje pełen cykl życia modelu: od eksperymentów, przez deployment jako REST API, po monitoring w produkcji.

**Nazwa repo:** `ml-serving-platform`

---

## Architektura

```
┌─────────────────────────────────────────────────────────┐
│                    docker-compose.yml                   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  MLflow      │  │  Model API   │  │  Prometheus  │   │
│  │  Tracking    │  │  (FastAPI)   │  │  + Grafana   │   │
│  │  :5000       │  │  :8000       │  │  :9090/:3000 │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                  │          │
│         └────────┬────────┘                  │          │
│                  │                           │          │
│         ┌───────┴────────┐                   │          │
│         │   PostgreSQL   │                   │          │
│         │   :5432        │                   │          │
│         └────────────────┘                   │          │
│                                              │          │
│  ┌──────────────┐                            │          │
│  │  Model API   │────── metrics ─────────────┘          │
│  │  (v2 - A/B)  │                                       │
│  │  :8001       │                                       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

**Serwisy:**

| Serwis | Technologia | Port | Rola |
|--------|------------|------|------|
| `mlflow` | MLflow Tracking Server | 5000 | Śledzenie eksperymentów, rejestr modeli |
| `api-v1` | FastAPI + scikit-learn | 8000 | Serwowanie modelu v1 (TF-IDF + LogReg) |
| `api-v2` | FastAPI + HuggingFace | 8001 | Serwowanie modelu v2 (DistilBERT) |
| `db` | PostgreSQL | 5432 | Backend store dla MLflow |
| `prometheus` | Prometheus | 9090 | Zbieranie metryk |
| `grafana` | Grafana | 3000 | Dashboardy monitoringu |

---

## Struktura repo

```
ml-serving-platform/
├── README.md
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: lint + test + build
│
├── training/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train_v1.py                 # TF-IDF + Logistic Regression
│   ├── train_v2.py                 # DistilBERT fine-tuning
│   ├── evaluate.py                 # Ewaluacja na test secie
│   └── data/
│       └── download_data.py        # Skrypt pobierający dataset
│
├── serving/
│   ├── v1/
│   │   ├── Dockerfile              # Multi-stage build
│   │   ├── requirements.txt
│   │   ├── app.py                  # FastAPI app
│   │   ├── model.py                # Logika ładowania modelu z MLflow
│   │   └── tests/
│   │       └── test_api.py
│   ├── v2/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── app.py
│   │   ├── model.py
│   │   └── tests/
│   │       └── test_api.py
│   └── common/
│       ├── schemas.py              # Pydantic modele (request/response)
│       ├── metrics.py              # Prometheus metrics helper
│       └── middleware.py           # Logging, request tracking
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboards/
│           └── model_performance.json
│
└── scripts/
    ├── ab_test.py                  # Skrypt symulujący ruch A/B
    └── drift_check.py              # Prosty monitoring driftu danych
```

---

## Plan realizacji (10 dni roboczych)

### Faza 1: Trening i MLflow (Dni 1–3)

#### Dzień 1 — Setup + Dane
- [ ] Zainicjuj repo Git z `.gitignore`, `README.md`, `LICENSE`
- [ ] Napisz `download_data.py` — pobiera **IMDB Reviews** (sentyment: pozytywny/negatywny)
- [ ] Skonfiguruj `docker-compose.yml` z serwisami: `db` (PostgreSQL) + `mlflow`
- [ ] Zweryfikuj, że MLflow UI działa na `localhost:5000`

**Kluczowe koncepty do zrozumienia:**
- Docker volumes (persystencja danych PostgreSQL)
- Docker networking (serwisy widzą się po nazwie)
- MLflow Tracking Server vs. MLflow Client

#### Dzień 2 — Model v1 (klasyczny ML)
- [ ] Napisz `train_v1.py`:
  - Preprocessing: czyszczenie tekstu, TF-IDF vectorizer
  - Model: `LogisticRegression` ze scikit-learn
  - Logowanie do MLflow: parametry, metryki (accuracy, F1, confusion matrix)
  - Rejestracja modelu w MLflow Model Registry
- [ ] Uruchom trening w kontenerze `training`
- [ ] Sprawdź wyniki w MLflow UI — porównaj kilka runów z różnymi hiperparametrami

**Cel:** Zrozumieć MLflow tracking, artifacts, model registry.

#### Dzień 3 — Model v2 (deep learning)
- [ ] Napisz `train_v2.py`:
  - Fine-tuning `distilbert-base-uncased` z HuggingFace Transformers
  - 2–3 epoki na podzbiorze danych (wystarczy ~5k przykładów)
  - Logowanie do MLflow z tagiem `model_type=transformer`
- [ ] Napisz `evaluate.py` — porównanie v1 vs v2 na tym samym test secie
- [ ] Zaloguj porównanie w MLflow

**Cel:** Pokazać, że umiesz pracować zarówno z klasycznym ML, jak i z DL.

---

### Faza 2: Serving API (Dni 4–6)

#### Dzień 4 — API v1
- [ ] Napisz `serving/v1/app.py`:
  ```python
  # Endpointy:
  POST /predict          # Predykcja sentymentu
  POST /predict/batch    # Batch predykcja
  GET  /health           # Health check
  GET  /model/info       # Wersja modelu, metryki z treningu
  ```
- [ ] Model ładowany z MLflow Model Registry przy starcie kontenera
- [ ] Pydantic schemas dla request/response
- [ ] Napisz `Dockerfile` z **multi-stage build**:
  - Stage 1: instalacja zależności
  - Stage 2: kopiowanie kodu + uruchomienie
- [ ] Dodaj do `docker-compose.yml`, przetestuj

#### Dzień 5 — API v2 + A/B routing
- [ ] Skopiuj strukturę v1, zamień model na DistilBERT
- [ ] Dodaj **request logging** — każdy request zapisywany do pliku/bazy:
  - Timestamp, input text, prediction, confidence, model version, latency
- [ ] Napisz `scripts/ab_test.py`:
  - Wysyła requesty losowo do v1 lub v2
  - Zbiera wyniki do porównania
- [ ] Dodaj do `docker-compose.yml` jako `api-v2`

#### Dzień 6 — Testy + obsługa błędów
- [ ] Napisz testy w `pytest`:
  - Testy endpointów (happy path + edge cases)
  - Test z pustym inputem, za długim tekstem
  - Test health checka
- [ ] Dodaj **graceful error handling**:
  - Custom exception handlers w FastAPI
  - Sensowne komunikaty błędów
  - Timeout na predykcje
- [ ] Dodaj `pre-commit` hooks: `black`, `ruff`, `mypy`

---

### Faza 3: Monitoring + CI/CD (Dni 7–9)

#### Dzień 7 — Prometheus + metryki
- [ ] Dodaj `prometheus-fastapi-instrumentator` do obu API
- [ ] Zdefiniuj custom metryki:
  ```
  prediction_requests_total        (counter, labels: model_version, sentiment)
  prediction_latency_seconds       (histogram, labels: model_version)
  prediction_confidence            (histogram, labels: model_version)
  model_prediction_errors_total    (counter, labels: model_version, error_type)
  ```
- [ ] Skonfiguruj `prometheus.yml` — scrape obu API
- [ ] Dodaj Prometheus do `docker-compose.yml`
- [ ] Zweryfikuj metryki na `localhost:9090`

#### Dzień 8 — Grafana dashboardy
- [ ] Dodaj Grafana do `docker-compose.yml` z auto-provisioningiem
- [ ] Stwórz dashboard `model_performance.json`:
  - Panel 1: Request rate per model (v1 vs v2)
  - Panel 2: Latency p50/p95/p99 per model
  - Panel 3: Confidence distribution
  - Panel 4: Error rate
  - Panel 5: A/B comparison — accuracy over time
- [ ] Uruchom `ab_test.py` i obserwuj dane na dashboardzie

#### Dzień 9 — Data drift + CI/CD
- [ ] Napisz `scripts/drift_check.py`:
  - Porównuje rozkład długości tekstów, częstości słów w nowych requestach vs. dane treningowe
  - Prosty test statystyczny (KS test lub PSI)
  - Alert jeśli drift wykryty (log warning)
- [ ] Napisz GitHub Actions workflow `.github/workflows/ci.yml`:
  ```yaml
  # Trigger: push to main, PR
  # Steps:
  # 1. Lint (ruff)
  # 2. Type check (mypy)
  # 3. Unit tests (pytest)
  # 4. Build Docker images
  # 5. Integration test (docker-compose up + health check)
  ```

---

### Faza 4: Polish + Dokumentacja (Dzień 10)

#### Dzień 10 — README + demo
- [ ] Napisz porządne `README.md`:
  - Opis projektu z architekturą (diagram)
  - Quick start: `docker-compose up` i gotowe
  - Screenshots: MLflow UI, Grafana dashboard, API docs (Swagger)
  - Sekcja "Czego się nauczyłem"
  - Sekcja "Co bym dodał gdybym miał więcej czasu" (Kubernetes, model retraining pipeline, feature store)
- [ ] Nagraj krótki GIF / screencast demo
- [ ] Ostatni review kodu, cleanup
- [ ] Push na GitHub, ustaw repo jako publiczne

---

## Kluczowe umiejętności demonstrowane w projekcie

| Obszar | Co pokazujesz |
|--------|--------------|
| **ML/DS** | Trening, ewaluacja, porównanie modeli, feature engineering |
| **MLOps** | MLflow tracking, model registry, model versioning |
| **Backend** | FastAPI, REST API design, Pydantic, error handling |
| **Docker** | Multi-stage builds, docker-compose, networking, volumes |
| **Monitoring** | Prometheus, Grafana, custom metryki, alerting |
| **Testing** | pytest, integration tests, CI/CD |
| **Software Engineering** | Clean code, typing, linting, Git workflow |

---

## Przydatne zasoby

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Docker Compose Getting Started](https://docs.docker.com/compose/gettingstarted/)
- [Prometheus + Grafana Tutorial](https://prometheus.io/docs/visualization/grafana/)
- [HuggingFace Transformers — Fine-tuning](https://huggingface.co/docs/transformers/training)
- [prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)

---

## Wskazówki

1. **Commituj często** — recruitery patrzą na historię commitów. Małe, opisowe commity > jeden wielki push.
2. **Nie kopiuj kodu** — pisz sam, nawet jeśli wolniej. W rozmowie kwalifikacyjnej będą pytać o szczegóły.
3. **Zacznij od docker-compose** — odpal minimalne serwisy (db + mlflow) pierwszego dnia. Motywacja rośnie, gdy coś działa.
4. **Model nie musi być idealny** — 85% accuracy wystarczy. Tu chodzi o infrastrukturę, nie o SOTA.
5. **README to Twoja wizytówka** — poświęć na niego tyle czasu, ile na kod. Dobry README > dobry model.
