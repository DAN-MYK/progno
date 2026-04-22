# Tennis Match Prediction App — Design

**Status**: Draft complete (Sections 1–7); pending user review
**Date**: 2026-04-22
**Use case**: personal betting tool

## Goal

Desktop-додаток на Tauri, який прогнозує переможця окремого тенісного матчу (pre-match, ATP → потім WTA) і допомагає знаходити value-ставки порівнянням ймовірності моделі з коефіцієнтами букмекерів.

## Context & Decisions Log

### Scope

- **Прогноз**: переможець окремого матчу (не турнір, не раунд).
- **Тури**: Phase 1 — ATP only. WTA — Phase 4.
- **Тип**: pre-match (live betting виключено — потребує платних real-time API і складніших моделей).
- **Аудиторія**: особисте користування (ліцензія CC-BY-NC-SA на даних Sackmann допустима; юридичних обмежень беттингу в Україні немає).

### Метрика успіху

Головний критерій — **не accuracy**, а калібровка + ROI на walk-forward бектесті:

- **Log-loss / Brier score** як основна метрика.
- **Бенчмарк**: модель має стабільно перемагати публічний Sackmann Elo на log-loss. Якщо ні — модель нічого не додає.
- **ROI**: симуляція ставок з fractional (0.25x) Kelly на історичних матчах, де closing odds від Pinnacle відомі (tennis-data.co.uk). Реалістична ціль +1–3% ROI.

### Data sources

- **[Jeff Sackmann `tennis_atp`](https://github.com/JeffSackmann/tennis_atp)** — матчі 1968→зараз, match stats 1991+. Ліцензія CC-BY-NC-SA 4.0 (non-commercial, що підходить для особистого використання).
- **[tennis-data.co.uk](http://www.tennis-data.co.uk/)** — ATP з 2000, WTA з 2007, із closing odds Pinnacle (PSW/PSL) і Bet365. Потрібно лише для Phase 2+ (фіча "opening-closing odds movement" і бектест ROI).
- **Склейка двох джерел**: не тривіальна через різні конвенції іменування гравців і турнірів. Має власну фазу в ETL.

### Критичні ML-принципи

- **No data leakage**: фічі = тільки pre-match агрегати (середнє по N минулих матчах до цього). Post-match статистика (aces, breakpoints won цього матчу) — заборонена як фіча.
- **Walk-forward validation, не random split**: тренуємо на [t0, t1], валідуємо [t1+1, t2], рухаємо вікно. Random split витікає майбутнє в минуле через player-form correlations.
- **Retirement handling**: ~2–3% матчів закінчуються відмовою. Виключаються з тренування (не справжній результат), але враховуються у feature lookup для наступних матчів.
- **Cold start**: нові гравці без історії → fallback на рейтинг і вік; prior у Elo.

### Architectural decisions

| # | Рішення | Причина |
|---|---------|---------|
| 1 | Phase 1 — жодного Python у рантаймі | Elo обчислюється тривіально в Rust; Python потрібен лише для ETL, запускається вручну раз на тиждень/місяць |
| 2 | Артефакти (`.json`, `.parquet`, пізніше model binary) читаються з app-data-директорії | Оновлення моделі = заміна файлу + перезапуск, без rebuild |
| 3 | Phase 3+: **Python sidecar** (FastAPI на localhost) для ML-інференсу + feature engineering | Уся логіка feature engineering в Python, не дублюється в Rust; ONNX додає складності без виграшу для персонального тулу. Деталі sidecar-архітектури — §4.7 |
| 4 | Input — **paste-from-clipboard parser** (не scraper) | ToS сайтів забороняє скрейпінг; HTML крихкий; парсер тексту працює з будь-якого джерела |
| 5 | Frontend — **Svelte** | Малий bundle, простий state, чудово інтегрується з Tauri; React допустимий альтернативи, якщо виникне потреба |

## Phases

Кожна фаза — самодостатній реліз, зупинитися можна на будь-якій.

| Phase | Що | Цінність при релізі |
|-------|-----|---------------------|
| **1** | Tauri skeleton + парсер paste-from-clipboard + ETL ATP → Elo (surface-specific) у Rust → ймовірність перемоги | Elo-прогноз для ваших матчів |
| **2** | Поле для коеф. букмекера → обчислення edge% → fractional Kelly stake recommendation | **Повний беттинг-тул на Elo-бейзлайні** |
| **3** | Python ETL pipeline + feature engineering (no-leakage) + CatBoost + Platt calibration + walk-forward validation + sidecar інтеграція | Покращена модель; UI не змінюється |
| **4** | WTA як окрема модель (окрема ETL, окремий Elo, окремий CatBoost), tab у UI | Подвоєння обсягу ставок |
| **5** | QoL: кнопка "retrain" у додатку, опціональний scraper розкладу, injury toggle | Щоденна зручність |

## Repository Structure

```
progno/
├── training/                    # Python, запускається вручну; не потрібна в проді
│   ├── data/raw/                # Sackmann CSV (git clone), tennis-data.co.uk XLSX
│   ├── pipelines/
│   │   ├── ingest.py            # Склеює Sackmann + tennis-data по match-id
│   │   ├── features.py          # Pre-match фічі: rolling form, H2H, fatigue — БЕЗ leakage
│   │   └── elo.py               # Surface-specific Elo; дампає snapshot у JSON
│   ├── models/
│   │   ├── train.py             # Phase 3+: CatBoost + Platt + walk-forward
│   │   └── export.py            # Зберігає .cbm + calibration.json + model_card.json
│   ├── notebooks/               # Експерименти, візуалізації бектестів
│   └── artifacts/               # Готові файли для додатку
│       ├── elo_state.json       # player_id → {elo_overall, elo_hard, elo_clay, elo_grass}
│       ├── players.parquet      # довідник: id, name, hand, height, DOB
│       ├── match_history.parquet # для lookup rolling form + H2H
│       └── model.cbm            # Phase 3+ (CatBoost native)
│
└── app/
    ├── src-tauri/               # Rust
    │   ├── src/
    │   │   ├── main.rs
    │   │   ├── elo.rs           # f(player_a, player_b, surface) → probability
    │   │   ├── features.rs      # Збирає pre-match фічі з локальних parquet
    │   │   ├── model.rs         # Phase 3+: IPC із Python sidecar
    │   │   ├── parser.rs        # Парсить вставлений текст у список матчів
    │   │   └── kelly.rs         # Phase 2+: EV + fractional Kelly
    │   └── Cargo.toml
    └── src/                     # Svelte UI
        ├── App.svelte
        └── lib/
```

## Workflows

### Dev workflow (ваш, раз на тиждень/місяць)

1. `git -C training/data/raw/tennis_atp pull` — свіжі матчі Sackmann.
2. Завантажити нові XLSX з tennis-data.co.uk (раз на сезон).
3. `python training/pipelines/ingest.py` → оновлені `match_history.parquet`, `players.parquet`.
4. `python training/pipelines/elo.py` → оновлений `elo_state.json`.
5. (Phase 3+) `python training/models/train.py` → оновлений `model.cbm` + `calibration.json` + `model_card.json`, метрики у notebook.
6. Копіюємо `artifacts/*` у app-data директорію додатку; перезапустити Tauri.

### User workflow (ви під час ставок)

1. Відкрити Tauri-додаток.
2. Скопіювати список сьогоднішніх матчів з flashscore/tennisexplorer/будь-де → вставити в додаток.
3. Парсер виводить розпізнані матчі з можливістю виправити. Для кожного: гравець А, Б, покриття (hard/clay/grass), турнір.
4. Для кожного матчу додаток показує: ймовірність моделі, ваш коефіцієнт (ввести з букмекера), implied probability, edge%, рекомендований розмір ставки (0.25x Kelly).
5. (Phase 5) Injury toggle на гравця → виключає матч або додає shrinkage.

---

## Section 2 — Data Pipeline

### 2.1 Inventory полів (з Sackmann data dictionary — верифіковано)

Кожен CSV `atp_matches_YYYY.csv`:

- **Ідентифікатори**: `tourney_id`, `tourney_name`, `tourney_date`, `match_num`
- **Контекст**: `surface` (Hard/Clay/Grass/Carpet), `tourney_level` (G=Grand Slam, M=Masters 1000, A=ATP 250/500, F=Finals, D=Davis Cup, C=Challenger, S=ITF), `draw_size`, `best_of`, `round` (R128…F), `minutes`
- **Гравці** (parallel winner/loser): `_id`, `_seed`, `_entry` (WC/Q/LL), `_name`, `_hand` (R/L/U), `_ht`, `_ioc`, `_age`, `_rank`, `_rank_points`
- **Рахунок**: `score` (текст, напр. `"6-4 3-6 7-5"` або `"6-4 3-6 RET"`)
- **Match stats (1991+)**: `_ace`, `_df`, `_svpt`, `_1stIn`, `_1stWon`, `_2ndWon`, `_SvGms`, `_bpSaved`, `_bpFaced` (parallel winner/loser)

Конвенція ранкінгу (цитата): rank = "as of tourney_date, or most recent ranking date before" — без leakage від майбутнього.

### 2.2 Ingest pipeline

```
data/raw/tennis_atp/atp_matches_*.csv
          ↓ (pandas, типи, парсинг дат)
data/staging/matches_raw.parquet            (усі матчі)
          ↓ (clean)
data/staging/matches_clean.parquet          (виключено retirement/walkover як лейбли)
          ↓ (feature engineering — pre-match only)
artifacts/match_history.parquet             (для lookup у Rust)
artifacts/elo_state.json                    (снепшот Elo)
artifacts/players.parquet                   (довідник)
```

Parquet обрано: збереження типів, швидкий read у Rust через `polars`/`arrow2`, ~10× менше за CSV.

### 2.3 Retirement/walkover handling

- Виключити з **навчальних лейблів**: `score` містить `RET`/`W/O`/`DEF` (~2.5–3% матчів).
- **Не виключати з історії гравця**: зіграні сети враховуються в fatigue і формі.
- Зберігаємо прапорець `is_complete` і `completed_sets`.

### 2.4 Data leakage — конкретні запобіжники

| Джерело leakage | Запобіжник |
|-----------------|------------|
| Post-match stats цього матчу як фічі | Тільки історичні rolling-агрегати |
| Rank, оновлений цим матчем | Sackmann rank — `as of tourney_date`, безпечно; асерт у ETL |
| Elo, оновлений цим матчем | Послідовний розрахунок по `tourney_date`; для матчу M беремо Elo **перед** M |
| Майбутні матчі в rolling form | Вікно `[match_date - N, match_date)`; асерт на час |
| `winner_name` як ознака | Вхід моделі завжди (player_a, player_b) з випадковою перестановкою; лейбл інвертується |
| Closing odds як фіча | Не використовуємо як вхід моделі, тільки як бенчмарк і для ROI |

### 2.5 Walk-forward протокол

Expanding window, retrain раз на рік:

```
Train: [2000, 2015] → Validate: 2016
Train: [2000, 2016] → Validate: 2017
...
Train: [2000, 2022] → Test: 2023
Train: [2000, 2023] → Test: 2024
```

- Валідація — гіперпараметри.
- Тест — фінальна оцінка, без переналаштування.
- Burn-in: 2000–2004 для нагріву Elo, без тренування.
- Жодного random split / KFold.

### 2.6 Join Sackmann ↔ tennis-data.co.uk (Phase 3+)

Канонічного рішення немає. Ступеневий match:

- **Ключ**: `(tourney_week, player_A_norm, player_B_norm)`, де `tourney_week` — понеділок тижня з `tourney_date`; ім'я нормалізоване до `"LASTNAME F"`, ASCII через `unidecode`.
- **Стратегія**: точний match → `rapidfuzz` з порогом 90+ → ручний `data/manual/name_map.csv` для решти.
- Очікуваний yield ~98–99%.

### 2.7 Cold start

- Elo start = 1500 (538 convention).
- Rolling form < 5 матчів — фіча = медіана популяції + булева `low_history_flag`.
- Surface-specific Elo до першого матчу на покритті — загальний Elo.
- Null rank → значення 2000 (нижче top-1500).

### 2.8 Відкрите (обговоримо в наступних секціях)

- Вікна rolling form (10/25/50 матчів? 3/6/12 міс?) — перевіримо на валідації.
- `rank` як окрема фіча при наявності Elo (ймовірно дублює).
- Concept drift через 25 років: expanding vs sliding 10-year window — вирішимо після першого прогону.

### 2.9 Ризики

- **Join yield**: невідомий до першого запуску. <95% → обмежений ROI-бектест.
- **Sackmann cadence**: оновлюється нерегулярно (після Grand Slam). Для свіжих матчів може бракувати даних — UX показує "data as of DATE".

## Section 3 — Features

### 3.1 Elo family (Phase 1)

Ймовірність:
```
P(A beats B) = 1 / (1 + 10^((elo_B - elo_A) / 400))
```

K-factor (538):
```
K(n) = 250 / (n + 5)^0.4     n = matches played before this match
```

Контекстні модифікатори K (multiplicative, з Ultimate Tennis Statistics):
- Tournament level: G=1.0, M=0.85, A=0.75, F=0.90, D=0.70, C=0.50, S=0.40
- Round: F=1.0, SF=0.95, QF=0.90, R16-earlier=0.85
- Best_of: BO5=1.0, BO3=0.90

Surface-specific — **4 окремі рейтинги** на гравця: `elo_overall`, `elo_hard`, `elo_clay`, `elo_grass`. Carpet оновлює лише overall.
Composite для прогнозу: `0.5 × elo_surface + 0.5 × elo_overall` (вага — hyperparameter). При <20 матчах на покритті — вага surface → 0.

### 3.2 Rolling form (Phase 3)

| Фіча | Вікно |
|------|-------|
| `win_rate_overall` | Останні 50 матчів |
| `win_rate_surface` | Останні 20 матчів на цьому покритті |
| `win_rate_12m` | Останні 12 місяців |
| `win_rate_vs_top20` | Останні 30 матчів проти rank ≤ 20 |

Cold start (<5 матчів) → медіана популяції + `low_history_flag=1`.

### 3.3 Fatigue (Phase 3)

| Фіча | Опис |
|------|------|
| `days_since_last_match` | Cap 30 |
| `sets_last_14d` | Сумарні сети |
| `matches_last_30d` | Кількість матчів |
| `minutes_last_7d` | Хвилини на корті (null → sets × 45) |
| `surface_switch` | 1 якщо попередній матч був на іншому покритті |

### 3.4 Serve/return efficiency (Phase 3, 1991+)

Rolling mean за 25 останніх матчів:

| Фіча | Формула |
|------|---------|
| `first_serve_in_pct` | `1stIn / svpt` |
| `first_serve_won_pct` | `1stWon / 1stIn` |
| `second_serve_won_pct` | `2ndWon / (svpt - 1stIn - df)` |
| `ace_rate` | `ace / svpt` |
| `df_rate` | `df / svpt` |
| `bp_saved_pct` | `bpSaved / bpFaced` |
| `return_points_won_pct` | Похідне через cross-join з opponent's serve_won |

Всі агрегати — по **минулих** матчах.

### 3.5 Head-to-head (Phase 3)

Shrinkage, щоб уникнути overfit на малих вибірках:
```
h2h_score = (wins_A_vs_B + prior × prior_mean) / (total + prior)
prior = 5, prior_mean = 0.5
```
+ `h2h_sample_size` як окрема фіча.

### 3.6 Контекст матчу

One-hot: surface (Hard/Clay/Grass), `best_of_5`, round (Final/Semi/Early), tourney_level (GS/Masters/Other), `is_indoor` (через manual mapping турнір → indoor/outdoor — Sackmann не містить явно).

### 3.7 Player meta

`age_A/B`, `age_diff`, `height_A/B`, `height_diff`, `lefty_vs_righty` (бінарна — відома edge ліворуких проти правшів).

### 3.8 Common-opponent (Phase 3.5, опціонально)

Knottenbelt/Sipko підхід: агрегована різниця стат проти спільних опонентів. Дорого в обчисленні; додаємо лише якщо дає gain на валідації.

### 3.9 Representation

**Pair-diff encoding**: числові фічі — як `f_A - f_B`. Категоріальні — one-hot без diff. Усуває потребу в симетризації.

**Підсумкова розмірність ~30–35 фіч** (Sipko=22, Stanford проекти 30–50 — наш діапазон стандартний).

### 3.10 Не включаємо

| Не | Чому |
|----|------|
| Current match stats | Leakage |
| Closing odds | Модель копіюватиме ринок |
| Player IDs/names | Overfit, concept drift |
| Seed/entry | Дублює Elo |
| Nationality | Шум (крім Davis Cup) |

### 3.11 Ризики

- Конкретні вікна (50/20/25) — калібруємо на валідації.
- Composite вага 0.5 — гіперпараметр, може бути 0.35.
- Common-opponent — великий потенціал, але витрати; вирішуємо після базової моделі.

## Section 4 — Model

### 4.1 Stack по фазах

| Phase | Модель | Мета |
|-------|--------|------|
| 1 | Pure Elo (§3.1) | Baseline, має бити 50% і rank-based |
| 2 | Elo + Kelly/EV overlay | Беттинг-тул на Elo |
| 3 | CatBoost + Platt calibration | Перевершити Elo на log-loss, +ROI |
| 3.5 | Common-opponent, hyperparam search | Incremental, лише якщо виграш |

### 4.2 Чому CatBoost

- XGBoost/LightGBM з коробки погано каліброві (ECE ~0.18); потребують Platt (до 0.04).
- CatBoost через ordered boosting має нижчий prediction shift і кращу нативну калібровку — критично для беттингу (продаємо ймовірність, не бінарне рішення).
- Native categorical handling (`surface`, `round`, `tourney_level` без one-hot).
- Менше hyperparameter tuning.

Fallback: LightGBM + Platt якщо CatBoost дає гірші метрики.

### 4.3 Acceptance criteria (non-negotiable)

Модель релізується тільки якщо:
- Log-loss нижчий за Pure Elo **і** Sackmann public Elo на тестових роках.
- ECE <0.03 post-calibration.
- ROI ≥ 0 на walk-forward test з 0.25× Kelly проти Pinnacle closing (CLV-based — див. нижче).

### 4.4 Hyperparameters (стартові)

```python
CatBoostClassifier(
    iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    loss_function='Logloss', eval_metric='Logloss',
    early_stopping_rounds=50,
    cat_features=['surface', 'round', 'tourney_level', 'hand_A', 'hand_B'],
    random_seed=42,
)
```

Random search 30 ітерацій на валідації: `depth` (4–8), `learning_rate` (0.01–0.1), `l2_leaf_reg` (1–10).

### 4.5 Калібрування

- Тренуємо CatBoost на train.
- Окремий hold-out рік для калібровки.
- **Platt scaling** (1-param logistic): `P_cal = sigmoid(a * logit(P_raw) + b)`.
- Isotonic — резерв; потребує більше даних.
- Тест — зафіксовані `{a, b}`, не переналаштовуються.

### 4.6 Метрики

**ML**: log-loss (primary), Brier, ECE (10 binів), reliability diagram.
**Betting**: ROI з 0.25× Kelly на Pinnacle closing, Sharpe-like (ROI/std), **CLV** (`mean(1/P_model - 1/P_pinnacle)` на зіграних ставках) — стабільніший за ROI на коротких періодах.

### 4.7 Serving architecture (Phase 3+, Tauri 2 sidecar pattern — верифіковано)

```
Tauri Rust
  ├── UI IPC (файли, вікно)
  └── spawns Python sidecar (tauri sidecar config)
                 ↓
         FastAPI 127.0.0.1:PORT  (рандомний вільний порт через argv)
                 ↓
         CatBoostClassifier.load_model("model.cbm")
                 ↓
         /health  /predict  /model_info
```

**Бандлинг**: PyInstaller one-file з catboost/fastapi/pydantic/uvicorn.

**UI ↔ Sidecar комунікація — через Rust-проксі команди** (не напряму з webview). Tauri 2 CSP блокує cross-origin за замовчуванням; Rust-команда простіша і безпечніша.

### 4.8 Artifact format

```
artifacts/model_v<version>.cbm           # CatBoost
artifacts/calibration_v<version>.json    # Platt {a, b}
artifacts/model_card_v<version>.json     # train date, years, metrics, features list
```

Model_card обов'язковий.

### 4.9 API contract

**Рішення**: feature engineering у Python sidecar, не в Rust. Sidecar читає `match_history.parquet` / `elo_state.json` сам. Rust передає лише ID і контекст.

```
POST /predict
{
  "matches": [{
    "player_a_id": 104745,
    "player_b_id": 106421,
    "surface": "Clay",
    "tourney_level": "M",
    "round": "QF",
    "best_of": 3,
    "indoor": false,
    "tourney_date": "2026-04-23"
  }]
}

Response:
{
  "model_version": "2026-04-22",
  "predictions": [{
    "prob_a_wins": 0.6234,
    "prob_a_wins_uncalibrated": 0.6812,
    "elo_prob_a_wins": 0.5891,
    "confidence_flag": "ok"       // "ok" | "low_history" | "insufficient_data"
  }]
}
```

### 4.10 Performance targets

- Cold start sidecar: <3s.
- Latency на 10 матчів: <200ms.
- Memory: <500MB.

### 4.11 Ризики

- PyInstaller + CatBoost бінарник ~300MB. Прийнятно для персоналки.
- Порт колізії → `port=0`, передати Rust через stdout handshake.
- Sidecar crash → Tauri supervisor через `tauri-plugin-shell`; UI показує toast.

## Section 5 — UI/UX

### 5.1 Принципи

1. **Чесність з невпевненістю** — прогноз завжди з датою моделі, confidence-прапорцем, поруч з Elo-бейзлайном.
2. **Один екран, мінімум кліків** — це персональний тул.
3. **Прозорість** — Elo і CatBoost видно поруч.
4. **Disclaimer "не порада"** — у footer.

### 5.2 Phase 1 wireframe (Elo, MVP)

```
┌─ Progno ─────────────────────────────────────────────────────┐
│  Paste today's matches                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Alcaraz vs Sinner - Madrid - Clay                      │  │
│  │ Djokovic vs Zverev - Madrid - Clay                     │  │
│  └────────────────────────────────────────────────────────┘  │
│  [Parse]                                                     │
│                                                              │
│  Match 1: Alcaraz vs Sinner   Surface: Clay ▾   BO3 ▾        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Alcaraz  62.3%  ████████████▌                        │    │
│  │ Sinner   37.7%  ███████▌                             │    │
│  │ Elo: Alcaraz 2145 (clay) / Sinner 2087 (clay)        │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  Model: Elo baseline · Data as of 2026-04-20 · ATP only      │
│  Not financial advice.                                       │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 Phase 2 wireframe (Elo + EV/Kelly)

Top bar: `Bankroll: $1000 [Edit]   ·   Kelly fraction: 0.25× [Edit]`.

Додатково в картці:

```
Your odds:  Alcaraz [1.85]    Sinner [2.00]

┌── Alcaraz ──────────┐  ┌── Sinner ──────────┐
│ Implied: 54.1%      │  │ Implied: 50.0%     │
│ Edge:    +8.2%  ✓   │  │ Edge:   -12.3%  ✗  │
│ Kelly:   3.5% BR    │  │ Kelly:   0% BR     │
│ Stake:   $35        │  │                    │
└─────────────────────┘  └────────────────────┘
```

Формули (`kelly.rs`):
```
implied_p   = 1 / decimal_odds
edge        = model_p - implied_p
full_kelly  = (model_p * odds - 1) / (odds - 1)
frac_kelly  = max(0, 0.25 * full_kelly)
stake       = bankroll * frac_kelly
```
`edge ≤ 0` → stake=0, картка приглушена.

### 5.4 Phase 3 wireframe (CatBoost + confidence breakdown)

Confidence flag (іконка): `ok` / `low_history` / `insufficient_data`.

Toggle "Show model breakdown" розкриває агрегований contribution by feature family:

```
Elo (surface composite)    57.5%  ← baseline
+ Recent form (last 12m)   +3.2%
+ H2H shrinkage            +0.8%
+ Surface specialization   +0.5%
+ Fatigue                  +0.3%
= CatBoost calibrated      62.3%
```

### 5.5 Phase 5 (injury toggle, retrain)

- Іконка 🩹 біля кожного гравця → shrinkage `P_adj = 0.7 × P_model + 0.3 × 0.5`, бордер жовтий.
- Settings → `[Retrain model now]` → spawn Python pipeline, прогрес-бар (30 s ETL + 2–5 хв тренування).

### 5.6 Persistence

| Що | Де |
|----|-----|
| Bankroll, Kelly fraction, UI prefs | `app_data/settings.json` через `tauri-plugin-store` |
| Історія прогнозів і ставок | `app_data/bets.sqlite` (`rusqlite`) |
| ML артефакти | `app_data/artifacts/` (замінюються при ретрейні) |
| Player lookup | `app_data/artifacts/players.parquet` |

SQLite для bets, а не JSON — щоб пізніше рахувати ROI по фільтрах.

### 5.7 Парсер тексту

Regex tokenizer підтримує:
- `Player A vs Player B`
- `Player A - Player B`
- `Player A, Player B - Tournament - Surface`
- `13:00  Alcaraz - Sinner  (Madrid, Clay)`

Fuzzy match імен проти `players.parquet`. Невпевнені — жовті картки з ручним вибором.

### 5.8 Frontend stack

- **Svelte 5** + Svelte stores.
- `@tauri-apps/api` для Rust команд.
- Tailwind для стилів.
- `@tauri-apps/plugin-store` для preferences.

### 5.9 Error states

- Sidecar down → banner "ML service stopped. Falling back to Elo. [Restart]".
- <5 матчів у вікні гравця → warning на картці.
- Parser не розпізнав → жовта картка з ручним вводом.
- `data_as_of` >14 днів → banner "Model data is stale. Consider retraining."

### 5.10 Не включаємо в MVP

| Не | Чому |
|----|------|
| Live-odds feed | Pre-match scope |
| Bracket forecasting | Окремий проект |
| Соц. фічі, leaderboard | Персональний тул |
| Cloud sync | Додає інфраструктуру |

### 5.11 Відкрите

- Темна тема за замовчуванням, toggle на світлу.
- Svelte vs React — рекомендую Svelte, без критичного trade-off.

## Section 6 — Retrain workflow + testing

### 6.1 Cadence

| Тригер | Частота |
|--------|---------|
| CLI `just retrain` | Тижнево (після більшості турнірів тижня) |
| UI "Retrain now" (Phase 5) | За потребою |
| Автоматичний крон | Не в MVP — Future (§7.2) |

### 6.2 Pipeline (`justfile`)

```
just update-data     # git pull Sackmann + download tennis-data.co.uk
just ingest          # raw → staging parquet (чистка, retirement flags)
just features        # staging → artifacts/ (features, Elo state)
just train           # CatBoost + Platt, export model.cbm + model_card
just validate        # walk-forward metrics, ECE, ROI; FAIL on gate
just publish         # копіює artifacts у app_data, оновлює symlink `current`
just retrain         # все разом
```

### 6.3 Versioning

```
artifacts/v2026-04-22_14-30/
├── model.cbm
├── calibration.json
├── model_card.json     # train window, metrics, features, git SHA
├── elo_state.json
└── match_history.parquet

artifacts/current → symlink на останній passed
```

Failed runs зберігаються; `just rollback v...` перемикає symlink.

### 6.4 Acceptance gate (`just validate`)

FAIL якщо:
- Log-loss на test > Pure Elo baseline.
- Log-loss на test > публічний Sackmann Elo.
- ECE post-calibration > 0.03.
- ROI на test < -1%.

При FAIL — `publish` не виконується, користувач залишається на попередній моделі.

### 6.5 Testing pyramid

**Python unit** (`training/tests/`, pytest):
- Retirement detection з `score` text.
- Score parser на відомі формати.
- Elo update — еталонні числа.
- Rolling form — synthetic player, нуль leakage.
- Platt calibration — synthetic data → очікувані `{a, b}`.
- Pair-diff encoding.

**Python ML-specific**:
- **Temporal leakage property test**: random date D → жодна фіча не залежить від матчів >= D.
- **Deterministic training** з фіксованим seed.
- **Monotonicity**: `elo_A - elo_B` ↑ → `P(A)` ↑.
- **Calibration in tolerance**: кожен 10%-бін `|actual - predicted| < 0.05`.
- **Cold start** не ламає pipeline.

**Rust unit** (`src-tauri/tests/`):
- Elo: `1500 vs 1500 → 0.5 ± ε`.
- Kelly: відомі `(p, odds, bankroll)` → очікуваний stake.
- Kelly neg edge → stake=0.
- Parser формат → `[(a, b, tourney, surface)]`.
- Parquet lookup.

**Integration**:
- Synthetic 100-matches, 10-players pipeline `ingest→features→train→validate`.
- Rust запускає sidecar, `/health` + `/predict` на stub-model.

**End-to-end smoke** (after `just retrain`):
- 10 hardcoded відомих активних гравців → inference → no NaN, ймовірності в [0.05, 0.95].
- Топ-20 `elo_overall` має містити відомих лідерів; інакше ETL зламане → pipeline не публікується.

### 6.6 Monitoring (`artifacts/v<ver>/report.md`)

- Топ-20 Elo overall + per-surface.
- Walk-forward metrics table.
- Reliability diagram (PNG).
- ROI per month на test.
- Feature importance (top-20).
- Delta vs попередня версія: Δlog-loss, ΔROI. `>0.005` погіршення → warn.

### 6.7 Reproducibility

- `random_seed=42` всюди.
- Python деп — `uv` lockfile.
- Rust — `Cargo.lock`.
- Sackmann git SHA у `model_card.json`.
- CatBoost version pinned.

### 6.8 Структура репо (розширена)

```
progno/
├── justfile
├── training/
│   ├── src/progno_train/
│   ├── tests/
│   ├── pyproject.toml
│   └── uv.lock
└── app/
    ├── src-tauri/
    │   ├── src/
    │   ├── tests/
    │   └── Cargo.toml
    ├── src/                  # Svelte
    ├── package.json
    └── bun.lockb
```

### 6.9 Ризики

- **Повторний gate fail** після нових даних — перевірити ETL (сміття?), cold-start (нові гравці?), розглянути sliding window.
- **Sackmann помилки в даних** — моніторити через smoke test на топ-гравцях.
- **Manual Elo override** (тому що гравець повернувся з травми) — **заборонити собі**; інакше знищується бектест.

## Section 7 — Future improvements

### 7.1 Принципи додавання

Кожне покращення проходить §6.4 gate: log-loss ≤ baseline, ECE не зростає, ROI/CLV покращується. Без цього — feature creep. Особливо небезпечно для LLM-фіч.

### 7.2 Автоматизація

| Пріоритет | Покращення | Ризик |
|-----------|-----------|-------|
| **P1** | Крон-ретрейн щотижня з gate (systemd timer) | Нотифікація на fail |
| **P1** | Auto-git-pull Sackmann щодня (02:00) | Низький |
| **P2** | Desktop notification на +EV (edge > 5%) | Reflex-bet; toggle у settings |
| **P2** | Auto-fetch розкладу (scraper, feature-flag) | ToS; HTML drift |
| **P3** | Auto-fetch odds через агрегатор API | $$$; етично серйозніше |
| **SKIP** | Auto-розміщення ставок через API букмекера | Regulatory + ризик багу |

### 7.3 LLM/ШІ через OpenRouter

#### Корисне

**1. News/injury retrieval (P2 — головний кандидат)**

Структуровані дані не містять новин про травми/форму/особисте. LLM з web search:
- Для кожного pre-match: запит про новини, травми, withdrawals за 30 днів.
- Повертає `{ injury_flag, recent_news[], sources[], confidence }`.
- UI показує блок "News check" з джерелами.
- **НЕ** корегує ймовірність — користувач вирішує натиснути injury toggle.

Чому OK: retrieval + summarization, не передбачення.

**2. Natural-language explanation (P3)**

Кнопка "Explain" → LLM перекладає SHAP breakdown у речення. Нуль впливу на ймовірність.

**3. Sanity check assistant (P3)**

"Check this pick" — LLM читає прогноз + news + поточні новини → flag очевидні проблеми (player withdrew hour ago).

**4. Parser augmentation (P2)**

Fallback на LLM коли regex не впорався: `"Extract tennis matches from this text as JSON"`. Cheap (Haiku/Gemini Flash).

#### НЕ треба

- **LLM як probability adjuster** — знищує калібровку.
- **LLM як primary predictor** — гірше за Elo.
- **LLM для автоматичного розміщення ставок** — ніколи.
- **LLM-generated player embeddings з тексту** — шум, лише якщо A/B на walk-forward стабільно кращий.

### 7.4 ML-покращення

| P | Що |
|---|----|
| **P1** | Ensemble Elo + CatBoost + LogReg (weighted) |
| **P1** | Refit calibration кожен retrain (temporal drift) |
| **P2** | Surface transfer learning (shrink до learned prior) |
| **P2** | Live in-play Markov chain + correction model (Phase 6) |
| **P3** | Neural player embeddings (sequence model) |
| **P3** | Multi-head: match winner + set + over/under games |

### 7.5 Беттинг-workflow

| P | Що |
|---|----|
| **P1** | CLV tracking dashboard (місяць/surface/market) |
| **P1** | Bet history з ROI breakdown |
| **P2** | Line shopping між кількома книгами (API) |
| **P2** | Stop-loss guards (pause після drawdown) |
| **P3** | Hedge calculator для live-хеджування |

### 7.6 Data

| P | Що |
|---|----|
| **P1** | Match Charting Project point-by-point (серв speed, rally patterns) |
| **P1** | Challenger/ITF для cold start новачків |
| **P2** | WTA повна окрема модель |
| **P3** | Weather/altitude — edge у Мадриді, Майамі |

### 7.7 UX

| P | Що |
|---|----|
| **P1** | Export прогнозів у CSV |
| **P2** | What-if slider для "injury probability" |
| **P3** | Mobile read-only companion view |

### 7.8 Інфраструктура (якщо виходитиме за персональне)

| P | Що |
|---|----|
| Якщо колаб | CI на GitHub Actions |
| Якщо експерименти | MLflow tracking |
| Якщо scale | Feature store, model registry |

### 7.9 Явно НІКОЛИ

- Automated betting API — regulatory + bug risk + емоційна пастка.
- Tipping service — вимагає KYC/ліцензію.
- Copy-trading — витрати з'їдають edge.

### 7.10 Рекомендований порядок після MVP

1. Крон-ретрейн з gate (P1) — ~1 день.
2. News/injury LLM check (P2) — ~2 дні, найцінніший LLM-use.
3. Bet history + CLV dashboard (P1) — критично для самооцінки.
4. Ensemble Elo + CatBoost (P1) — швидкий ML gain.
5. Решта — за потребою.
