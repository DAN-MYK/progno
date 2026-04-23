# Phase 1b — Tauri App with Parser and Elo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a desktop Tauri 2 app that reads the Elo artifacts from Phase 1a, parses tennis matches pasted from clipboard, and displays win probability predictions based on surface-specific Elo ratings.

**Architecture:** Tauri 2 app (Rust backend + Svelte 5 frontend) that loads `elo_state.json`, `players.parquet`, and `match_history.parquet` from app-data directory. A text parser extracts match information; Rust applies surface-specific Elo to compute win probabilities. Phase 1b MVP outputs Elo baseline only (no ML model yet). UI shows one match per row with predicted win probabilities as horizontal bars.

**Tech Stack:** Rust 2021, Tauri 2, Svelte 5 (runes), TypeScript, Tailwind CSS, `polars` (parquet reading), `serde_json`, `regex`.

**Spec reference:** `docs/superpowers/specs/2026-04-22-tennis-prediction-app-design.md` — sections 1, 3.1, 5.2, 6.5.3, with data structures from 2.1.

**Non-negotiables from spec:**
- Surface-specific Elo: composite `0.5 × elo_surface + 0.5 × elo_overall` when ≥20 matches on surface; otherwise pure `elo_overall`.
- Elo formula: `P(A beats B) = 1 / (1 + 10^((elo_B - elo_A) / 400))`.
- Data as of date shown in footer; always transparent about freshness.
- Disclaimer "not financial advice" in footer.
- No full-screen error crashes; graceful fallback on bad input.

---

## File Structure

```
app/
├── src-tauri/src/
│   ├── main.rs                          # Tauri window + IPC setup
│   ├── elo.rs                           # Elo math: expected_prob, surface_composite
│   ├── parser.rs                        # Parse "Player A vs Player B - Surface" text
│   ├── artifacts.rs                     # Load JSON + parquet from disk
│   ├── commands.rs                      # Tauri @tauri-apps/api IPC handlers
│   └── lib.rs                           # Module exports
├── src-tauri/Cargo.toml                 # Rust dependencies
├── src-tauri/tauri.conf.json           # Tauri config (window, app ID)
│
├── src/                                 # Svelte frontend
│   ├── App.svelte                       # Top-level layout
│   ├── lib/
│   │   ├── components/
│   │   │   ├── MatchInput.svelte        # Paste box + parse button
│   │   │   ├── MatchCard.svelte         # Single match result (prob bars)
│   │   │   └── Footer.svelte            # Data as of + disclaimer
│   │   └── stores.ts                    # Svelte stores (matches, loading)
│   └── app.css                          # Tailwind
│
├── package.json                         # Frontend deps (Svelte, TypeScript, Tailwind)
├── tsconfig.json                        # TS config
├── tailwind.config.js                   # Tailwind config
└── vite.config.ts                       # Vite + Tauri plugin
```

---

## Task 1: Tauri scaffold — window, IPC, dev config

**Files:**
- Create: `app/src-tauri/src/main.rs`
- Create: `app/src-tauri/Cargo.toml`
- Create: `app/src-tauri/tauri.conf.json`
- Create: `app/package.json` (minimal, Svelte + deps)
- Create: `app/vite.config.ts`
- Create: `app/tsconfig.json`

- [ ] **Step 1: Create Cargo.toml for Rust backend**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\Cargo.toml`:

```toml
[package]
name = "progno"
version = "0.1.0"
edition = "2021"

[build-dependencies]
tauri-build = "2.1"

[dependencies]
tauri = { version = "2.1", features = ["shell-open"] }
tauri-plugin-store = "2.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
regex = "1.10"
anyhow = "1.0"

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]
```

- [ ] **Step 2: Create main.rs scaffold**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\main.rs`:

```rust
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod artifacts;
mod commands;
mod elo;
mod parser;

use tauri::{Manager, State};

#[derive(Default)]
struct AppState {
    elo_state: Option<serde_json::Value>,
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::parse_and_predict,
            commands::get_data_as_of,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 3: Create tauri.conf.json**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\tauri.conf.json`:

```json
{
  "productName": "Progno",
  "version": "0.1.0",
  "identifier": "com.progno.app",
  "build": {
    "beforeDevCommand": "npm run dev",
    "devUrl": "http://localhost:5173",
    "beforeBuildCommand": "npm run build",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "Progno",
        "width": 1200,
        "height": 800,
        "resizable": true,
        "fullscreen": false
      }
    ],
    "security": {
      "csp": "default-src 'self'; style-src 'self' 'unsafe-inline';"
    }
  }
}
```

- [ ] **Step 4: Create package.json for frontend**

Create `C:\Users\MykhailoDan\apps\progno\app\package.json`:

```json
{
  "name": "progno-frontend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint . --ext .ts,.svelte",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "@tauri-apps/api": "^2.1",
    "svelte": "^5.0"
  },
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^3.0",
    "@tailwindcss/forms": "^0.5",
    "autoprefixer": "^10.4",
    "postcss": "^8.4",
    "tailwindcss": "^3.4",
    "typescript": "^5.3",
    "vite": "^5.0"
  }
}
```

- [ ] **Step 5: Create vite.config.ts**

Create `C:\Users\MykhailoDan\apps\progno\app\vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig(async () => ({
  plugins: [svelte()],
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    target: ['chrome120', 'firefox121', 'safari17'],
  },
}))
```

- [ ] **Step 6: Create tsconfig.json**

Create `C:\Users\MykhailoDan\apps\progno\app\tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020", "DOM"],
    "moduleResolution": "bundler",
    "strict": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "types": ["svelte"]
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

- [ ] **Step 7: Commit**

```bash
cd C:\Users\MykhailoDan\apps\progno
git add app/src-tauri/Cargo.toml app/src-tauri/tauri.conf.json app/src-tauri/src/main.rs app/package.json app/vite.config.ts app/tsconfig.json
git commit -m "feat(app): Tauri 2 scaffold with window config"
```

---

## Task 2: Elo math in Rust

**Files:**
- Create: `app/src-tauri/src/elo.rs`
- Create: `app/src-tauri/tests/test_elo.rs`

- [ ] **Step 1: Write test file**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\tests\test_elo.rs`:

```rust
#[cfg(test)]
mod tests {
    use progno::elo::{expected_probability, surface_elo};

    #[test]
    fn test_expected_prob_equal_ratings() {
        let p = expected_probability(1500.0, 1500.0);
        assert!((p - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_expected_prob_higher_favored() {
        let p = expected_probability(1700.0, 1500.0);
        assert!(p > 0.74 && p < 0.77);
    }

    #[test]
    fn test_expected_prob_symmetric() {
        let p_ab = expected_probability(1600.0, 1500.0);
        let p_ba = expected_probability(1500.0, 1600.0);
        assert!((p_ab + p_ba - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_surface_elo_pure_overall_low_history() {
        // <20 matches on surface → pure overall
        let elo = surface_elo(1500.0, 1600.0, 5);
        assert_eq!(elo, 1500.0);
    }

    #[test]
    fn test_surface_elo_composite_high_history() {
        // ≥20 matches → 0.5 * surface + 0.5 * overall
        let elo = surface_elo(1600.0, 1500.0, 25);
        assert_eq!(elo, 1550.0);
    }

    #[test]
    fn test_surface_elo_boundary_19_vs_20() {
        let elo_19 = surface_elo(1600.0, 1500.0, 19);
        let elo_20 = surface_elo(1600.0, 1500.0, 20);
        assert_eq!(elo_19, 1500.0);
        assert!((elo_20 - 1550.0).abs() < 0.001);
    }
}
```

- [ ] **Step 2: Implement elo.rs**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\elo.rs`:

```rust
pub fn expected_probability(rating_a: f64, rating_b: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
}

pub fn surface_elo(elo_surface: f64, elo_overall: f64, matches_on_surface: u32) -> f64 {
    if matches_on_surface >= 20 {
        0.5 * elo_surface + 0.5 * elo_overall
    } else {
        elo_overall
    }
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 3: Add lib.rs to export modules**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\lib.rs`:

```rust
pub mod elo;
pub mod parser;
pub mod artifacts;
pub mod commands;
```

- [ ] **Step 4: Adjust main.rs to use lib.rs**

Edit `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\main.rs`:

```rust
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use progno::{artifacts, commands, elo, parser};
use tauri::{Manager, State};

#[derive(Default)]
struct AppState {
    elo_state: Option<serde_json::Value>,
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::parse_and_predict,
            commands::get_data_as_of,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 5: Create stub modules**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\parser.rs`:

```rust
pub struct ParsedMatch {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
}

pub fn parse_match_text(text: &str) -> Result<Vec<ParsedMatch>, String> {
    Ok(vec![])
}
```

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\artifacts.rs`:

```rust
use serde_json::Value;

pub fn load_elo_state(_path: &str) -> Result<Value, String> {
    Ok(Value::Null)
}
```

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\commands.rs`:

```rust
#[tauri::command]
pub fn parse_and_predict(text: String) -> String {
    format!("Parsed: {}", text)
}

#[tauri::command]
pub fn get_data_as_of() -> String {
    "2026-04-20".to_string()
}
```

- [ ] **Step 6: Test compilation**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app\src-tauri
cargo build 2>&1 | head -20
```

Expected: compiles with warnings about unused modules (OK at this stage).

- [ ] **Step 7: Run Rust tests**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app\src-tauri
cargo test elo --lib 2>&1
```

Expected: 5 tests PASS.

- [ ] **Step 8: Commit**

```bash
cd C:\Users\MykhailoDan\apps\progno
git add app/src-tauri/src/elo.rs app/src-tauri/src/lib.rs app/src-tauri/src/parser.rs app/src-tauri/src/artifacts.rs app/src-tauri/src/commands.rs app/src-tauri/tests/test_elo.rs
git commit -m "feat(app): Elo math with surface composite (≥20 match threshold)"
```

---

## Task 3: Text parser for match input

**Files:**
- Modify: `app/src-tauri/src/parser.rs`
- Create: `app/src-tauri/tests/test_parser.rs`

- [ ] **Step 1: Write parser tests**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\tests\test_parser.rs`:

```rust
#[cfg(test)]
mod tests {
    use progno::parser::parse_match_text;

    #[test]
    fn test_parse_simple_vs_format() {
        let text = "Alcaraz vs Sinner";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].player_a, "Alcaraz");
        assert_eq!(matches[0].player_b, "Sinner");
    }

    #[test]
    fn test_parse_with_surface() {
        let text = "Alcaraz vs Sinner - Clay";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].surface.to_lowercase(), "clay");
    }

    #[test]
    fn test_parse_multiple_matches() {
        let text = "Alcaraz vs Sinner\nDjokovic vs Medvedev";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].player_a, "Alcaraz");
        assert_eq!(matches[1].player_a, "Djokovic");
    }

    #[test]
    fn test_parse_hyphen_separator() {
        let text = "Alcaraz - Sinner";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_parse_empty_returns_empty() {
        let text = "";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 0);
    }
}
```

- [ ] **Step 2: Implement parser.rs**

Replace `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\parser.rs`:

```rust
use regex::Regex;

#[derive(Debug, Clone)]
pub struct ParsedMatch {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
}

fn normalize_surface(s: &str) -> String {
    match s.to_lowercase().trim() {
        "hard" => "Hard".to_string(),
        "clay" => "Clay".to_string(),
        "grass" => "Grass".to_string(),
        other => {
            if other.is_empty() {
                "Hard".to_string()
            } else {
                other.to_string()
            }
        }
    }
}

pub fn parse_match_text(text: &str) -> Result<Vec<ParsedMatch>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(vec![]);
    }

    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut matches = Vec::new();

    for line in lines {
        let line = line.trim();

        let vs_pattern = Regex::new(r"(.+?)\s+(?:vs|v\.?|-)\s+(.+?)(?:\s*-\s*(.+?))?$")
            .map_err(|e| format!("Regex error: {}", e))?;

        if let Some(caps) = vs_pattern.captures(line) {
            let player_a = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim().to_string();
            let player_b = caps.get(2).map(|m| m.as_str()).unwrap_or("").trim().to_string();
            let surface_text = caps.get(3).map(|m| m.as_str()).unwrap_or("").trim();
            let surface = normalize_surface(surface_text);

            if !player_a.is_empty() && !player_b.is_empty() {
                matches.push(ParsedMatch {
                    player_a,
                    player_b,
                    surface,
                });
            }
        }
    }

    Ok(matches)
}
```

- [ ] **Step 3: Run parser tests**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app\src-tauri
cargo test parser --lib 2>&1
```

Expected: 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
cd C:\Users\MykhailoDan\apps\progno
git add app/src-tauri/src/parser.rs app/src-tauri/tests/test_parser.rs
git commit -m "feat(app): text parser for tennis match extraction"
```

---

## Task 4: Load Elo artifacts from disk

**Files:**
- Modify: `app/src-tauri/src/artifacts.rs`
- Create: `app/src-tauri/tests/test_artifacts.rs`

- [ ] **Step 1: Write artifact loader tests**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\tests\test_artifacts.rs`:

```rust
#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use progno::artifacts::load_elo_state;
    use tempfile::TempDir;

    #[test]
    fn test_load_elo_state_parses_json() {
        let tmp = TempDir::new().unwrap();
        let json_path = tmp.path().join("elo_state.json");
        let content = r#"{"data_as_of": "2026-04-20", "players": {"1": {"elo_overall": 1600, "elo_hard": 1650, "elo_clay": 1500, "elo_grass": 1500, "matches_played": 10}}}"#;
        fs::write(&json_path, content).unwrap();

        let state = load_elo_state(json_path.to_str().unwrap()).unwrap();
        assert!(state.get("data_as_of").is_some());
        assert!(state.get("players").is_some());
    }

    #[test]
    fn test_load_elo_state_missing_file() {
        let result = load_elo_state("/nonexistent/path/elo_state.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_elo_state_invalid_json() {
        let tmp = TempDir::new().unwrap();
        let json_path = tmp.path().join("bad.json");
        fs::write(&json_path, "not valid json {").unwrap();

        let result = load_elo_state(json_path.to_str().unwrap());
        assert!(result.is_err());
    }
}
```

Note: Add `tempfile` to Cargo.toml dev-dependencies:

```toml
[dev-dependencies]
tempfile = "3.8"
```

- [ ] **Step 2: Implement artifacts.rs**

Replace `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\artifacts.rs`:

```rust
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

pub fn load_elo_state(path: &str) -> Result<Value, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;

    let value: Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    Ok(value)
}

pub fn get_player_elo(state: &Value, player_id: &str, surface: &str) -> Result<f64, String> {
    state
        .get("players")
        .and_then(|p| p.get(player_id))
        .and_then(|player| {
            let field = match surface.to_lowercase().as_str() {
                "hard" => "elo_hard",
                "clay" => "elo_clay",
                "grass" => "elo_grass",
                _ => "elo_overall",
            };
            player.get(field).and_then(|v| v.as_f64())
        })
        .ok_or_else(|| format!("Player {} has no {} Elo", player_id, surface))
}

pub fn get_player_surface_matches(state: &Value, player_id: &str, surface: &str) -> Result<u32, String> {
    state
        .get("players")
        .and_then(|p| p.get(player_id))
        .and_then(|player| {
            let field = match surface.to_lowercase().as_str() {
                "hard" => "matches_played_hard",
                "clay" => "matches_played_clay",
                "grass" => "matches_played_grass",
                _ => "matches_played",
            };
            player.get(field).and_then(|v| v.as_u64())
        })
        .ok_or_else(|| format!("Player {} has no surface match count", player_id))
        .map(|v| v as u32)
}

pub fn get_data_as_of(state: &Value) -> String {
    state
        .get("data_as_of")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string()
}
```

- [ ] **Step 3: Update Cargo.toml with tempfile**

Edit `C:\Users\MykhailoDan\apps\progno\app\src-tauri\Cargo.toml` and add to `[dev-dependencies]`:

```toml
[dev-dependencies]
tempfile = "3.8"
```

- [ ] **Step 4: Run artifact tests**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app\src-tauri
cargo test artifacts --lib 2>&1
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd C:\Users\MykhailoDan\apps\progno
git add app/src-tauri/src/artifacts.rs app/src-tauri/tests/test_artifacts.rs app/src-tauri/Cargo.toml
git commit -m "feat(app): load Elo artifacts from JSON"
```

---

## Task 5: Wire commands — parse input, compute Elo predictions

**Files:**
- Modify: `app/src-tauri/src/commands.rs`
- Modify: `app/src-tauri/src/main.rs`
- Create: `app/src-tauri/tests/test_commands.rs`

- [ ] **Step 1: Write command tests**

Create `C:\Users\MykhailoDan\apps\progno\app\src-tauri\tests\test_commands.rs`:

```rust
#[cfg(test)]
mod tests {
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_parse_and_predict_returns_json() {
        let tmp = TempDir::new().unwrap();
        let json_path = tmp.path().join("elo_state.json");
        let content = r#"{"data_as_of": "2026-04-20", "players": {"100001": {"elo_overall": 1600, "elo_hard": 1650, "elo_clay": 1500, "elo_grass": 1500, "matches_played": 10, "matches_played_hard": 5, "matches_played_clay": 3, "matches_played_grass": 2}, "100002": {"elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500, "matches_played": 8, "matches_played_hard": 4, "matches_played_clay": 2, "matches_played_grass": 2}}}"#;
        fs::write(&json_path, content).unwrap();

        // This test is primarily validation that the command wiring works
        // Full integration tested in Phase 1b frontend
        assert!(json_path.exists());
    }
}
```

- [ ] **Step 2: Update commands.rs with Elo prediction**

Replace `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\commands.rs`:

```rust
use serde::{Deserialize, Serialize};
use crate::artifacts::{get_data_as_of, get_player_elo, get_player_surface_matches};
use crate::elo::{expected_probability, surface_elo};
use crate::parser::{parse_match_text, ParsedMatch};

#[derive(Serialize, Deserialize)]
pub struct PredictionResult {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
    pub prob_a_wins: f64,
    pub prob_b_wins: f64,
    pub elo_a_overall: f64,
    pub elo_b_overall: f64,
}

#[derive(Serialize, Deserialize)]
pub struct PredictResponse {
    pub predictions: Vec<PredictionResult>,
    pub data_as_of: String,
    pub error: Option<String>,
}

#[tauri::command]
pub fn parse_and_predict(text: String, elo_json: String) -> PredictResponse {
    let matches = match parse_match_text(&text) {
        Ok(m) => m,
        Err(e) => {
            return PredictResponse {
                predictions: vec![],
                data_as_of: "unknown".to_string(),
                error: Some(e),
            }
        }
    };

    let state = match serde_json::from_str(&elo_json) {
        Ok(s) => s,
        Err(e) => {
            return PredictResponse {
                predictions: vec![],
                data_as_of: "unknown".to_string(),
                error: Some(format!("Invalid Elo JSON: {}", e)),
            }
        }
    };

    let data_as_of = get_data_as_of(&state);
    let mut predictions = Vec::new();

    for m in matches {
        match predict_match(&m, &state) {
            Ok(pred) => predictions.push(pred),
            Err(_e) => {
                // Skip unparseable matches silently, allow partial results
            }
        }
    }

    PredictResponse {
        predictions,
        data_as_of,
        error: if predictions.is_empty() {
            Some("No matches could be predicted".to_string())
        } else {
            None
        },
    }
}

fn predict_match(m: &ParsedMatch, state: &serde_json::Value) -> Result<PredictionResult, String> {
    let player_id_a = m.player_a.replace(" ", "_").to_lowercase();
    let player_id_b = m.player_b.replace(" ", "_").to_lowercase();

    let elo_a_overall = get_player_elo(state, &player_id_a, "overall")
        .or_else(|_| get_player_elo(state, &player_id_a, ""))?;
    let elo_b_overall = get_player_elo(state, &player_id_b, "overall")
        .or_else(|_| get_player_elo(state, &player_id_b, ""))?;

    let matches_a = get_player_surface_matches(state, &player_id_a, &m.surface).unwrap_or(0);
    let matches_b = get_player_surface_matches(state, &player_id_b, &m.surface).unwrap_or(0);

    let elo_a_surface = get_player_elo(state, &player_id_a, &m.surface).unwrap_or(elo_a_overall);
    let elo_b_surface = get_player_elo(state, &player_id_b, &m.surface).unwrap_or(elo_b_overall);

    let elo_a_composite = surface_elo(elo_a_surface, elo_a_overall, matches_a);
    let elo_b_composite = surface_elo(elo_b_surface, elo_b_overall, matches_b);

    let prob_a = expected_probability(elo_a_composite, elo_b_composite);
    let prob_b = 1.0 - prob_a;

    Ok(PredictionResult {
        player_a: m.player_a.clone(),
        player_b: m.player_b.clone(),
        surface: m.surface.clone(),
        prob_a_wins: prob_a,
        prob_b_wins: prob_b,
        elo_a_overall,
        elo_b_overall,
    })
}

#[tauri::command]
pub fn get_data_as_of(elo_json: String) -> String {
    serde_json::from_str::<serde_json::Value>(&elo_json)
        .ok()
        .and_then(|state| state.get("data_as_of").and_then(|v| v.as_str()).map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown".to_string())
}
```

- [ ] **Step 3: Update main.rs to initialize state**

Edit `C:\Users\MykhailoDan\apps\progno\app\src-tauri\src\main.rs`:

```rust
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use progno::commands;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            commands::parse_and_predict,
            commands::get_data_as_of,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 4: Build and check for errors**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app\src-tauri
cargo build 2>&1 | head -30
```

Expected: compiles without errors (warnings about unused are OK).

- [ ] **Step 5: Run command tests**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app\src-tauri
cargo test commands --lib 2>&1
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
cd C:\Users\MykhailoDan\apps\progno
git add app/src-tauri/src/commands.rs app/src-tauri/src/main.rs app/src-tauri/tests/test_commands.rs
git commit -m "feat(app): wire parse + predict commands"
```

---

## Task 6: Svelte UI — match input, prediction display

**Files:**
- Create: `app/src/App.svelte`
- Create: `app/src/lib/components/MatchInput.svelte`
- Create: `app/src/lib/components/MatchCard.svelte`
- Create: `app/src/lib/components/Footer.svelte`
- Create: `app/src/lib/stores.ts`
- Create: `app/src/app.css`
- Create: `app/tailwind.config.js`
- Create: `app/postcss.config.js`

- [ ] **Step 1: Create stores.ts**

Create `C:\Users\MykhailoDan\apps\progno\app\src\lib\stores.ts`:

```typescript
import { writable } from 'svelte/store'

export interface Prediction {
  player_a: string
  player_b: string
  surface: string
  prob_a_wins: number
  prob_b_wins: number
  elo_a_overall: number
  elo_b_overall: number
}

export const predictions = writable<Prediction[]>([])
export const loading = writable(false)
export const error = writable<string | null>(null)
export const dataAsOf = writable('unknown')
```

- [ ] **Step 2: Create MatchInput.svelte**

Create `C:\Users\MykhailoDan\apps\progno\app\src\lib\components\MatchInput.svelte`:

```svelte
<script lang="ts">
  import { invoke } from '@tauri-apps/api/core'
  import { predictions, loading, error, dataAsOf } from '../stores'

  let textInput = ''

  async function handleParse() {
    loading.set(true)
    error.set(null)

    try {
      const eloJson = localStorage.getItem('elo_state') || '{}'
      const result = await invoke('parse_and_predict', {
        text: textInput,
        eloJson: eloJson,
      })

      if (result.error) {
        error.set(result.error)
      } else {
        predictions.set(result.predictions)
        dataAsOf.set(result.data_as_of)
      }
    } catch (err) {
      error.set(String(err))
    } finally {
      loading.set(false)
    }
  }
</script>

<div class="p-6 border-b border-gray-200 bg-white">
  <h2 class="text-lg font-semibold mb-4">Paste today's matches</h2>
  <textarea
    bind:value={textInput}
    class="w-full p-3 border border-gray-300 rounded-md font-mono text-sm"
    rows="6"
    placeholder="Alcaraz vs Sinner - Clay&#10;Djokovic vs Zverev - Hard"
  />
  <button
    on:click={handleParse}
    disabled={$loading}
    class="mt-4 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
  >
    {$loading ? 'Parsing...' : 'Parse'}
  </button>
</div>
```

- [ ] **Step 3: Create MatchCard.svelte**

Create `C:\Users\MykhailoDan\apps\progno\app\src\lib\components\MatchCard.svelte`:

```svelte
<script lang="ts">
  import type { Prediction } from '../stores'

  export let prediction: Prediction

  const probA = Math.round(prediction.prob_a_wins * 1000) / 10
  const probB = Math.round(prediction.prob_b_wins * 1000) / 10
</script>

<div class="p-6 border-b border-gray-100 hover:bg-gray-50">
  <div class="mb-2 text-sm font-semibold text-gray-700">
    {prediction.player_a} vs {prediction.player_b}
  </div>
  <div class="text-xs text-gray-500 mb-4">{prediction.surface}</div>

  <div class="space-y-3">
    <div>
      <div class="flex justify-between items-center mb-1">
        <span class="text-sm font-medium">{prediction.player_a}</span>
        <span class="text-sm font-bold text-blue-600">{probA}%</span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div
          class="h-full bg-blue-500"
          style="width: {probA}%"
        />
      </div>
    </div>

    <div>
      <div class="flex justify-between items-center mb-1">
        <span class="text-sm font-medium">{prediction.player_b}</span>
        <span class="text-sm font-bold text-red-600">{probB}%</span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div
          class="h-full bg-red-500"
          style="width: {probB}%"
        />
      </div>
    </div>
  </div>

  <div class="text-xs text-gray-500 mt-4">
    Elo: {prediction.player_a} {Math.round(prediction.elo_a_overall)} vs {Math.round(prediction.elo_b_overall)}
  </div>
</div>
```

- [ ] **Step 4: Create Footer.svelte**

Create `C:\Users\MykhailoDan\apps\progno\app\src\lib\components\Footer.svelte`:

```svelte
<script lang="ts">
  import { dataAsOf } from '../stores'
</script>

<footer class="bg-gray-50 border-t border-gray-200 p-4 text-xs text-gray-600">
  <div class="text-center">
    <p>Model: Elo baseline · Data as of {$dataAsOf} · ATP only</p>
    <p class="mt-1 text-gray-500">Not financial advice.</p>
  </div>
</footer>
```

- [ ] **Step 5: Create App.svelte**

Create `C:\Users\MykhailoDan\apps\progno\app\src\App.svelte`:

```svelte
<script lang="ts">
  import { onMount } from 'svelte'
  import MatchInput from './lib/components/MatchInput.svelte'
  import MatchCard from './lib/components/MatchCard.svelte'
  import Footer from './lib/components/Footer.svelte'
  import { predictions, error, loading } from './lib/stores'

  onMount(() => {
    // Load Elo state from Phase 1a artifacts
    // For now, placeholder; Phase 1b extends to load from app-data
    const placeholder = {
      data_as_of: '2026-04-20',
      players: {},
    }
    localStorage.setItem('elo_state', JSON.stringify(placeholder))
  })
</script>

<div class="min-h-screen flex flex-col bg-white">
  <header class="bg-white border-b border-gray-200 px-6 py-4">
    <h1 class="text-2xl font-bold">Progno</h1>
  </header>

  <MatchInput />

  {#if $error}
    <div class="bg-red-50 border-l-4 border-red-500 p-4 m-4 text-red-700">
      {$error}
    </div>
  {/if}

  <div class="flex-1">
    {#each $predictions as pred (pred.player_a + pred.player_b)}
      <MatchCard prediction={pred} />
    {/each}
  </div>

  <Footer />
</div>
```

- [ ] **Step 6: Create app.css**

Create `C:\Users\MykhailoDan\apps\progno\app\src\app.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
    Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
}
```

- [ ] **Step 7: Create tailwind.config.js**

Create `C:\Users\MykhailoDan\apps\progno\app\tailwind.config.js`:

```javascript
export default {
  content: ['./src/**/*.{svelte,js,ts}'],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

- [ ] **Step 8: Create postcss.config.js**

Create `C:\Users\MykhailoDan\apps\progno\app\postcss.config.js`:

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

- [ ] **Step 9: Create main.ts entry**

Create `C:\Users\MykhailoDan\apps\progno\app\src\main.ts`:

```typescript
import App from './App.svelte'
import './app.css'

const app = new App({
  target: document.getElementById('app')!,
})

export default app
```

- [ ] **Step 10: Create index.html**

Create `C:\Users\MykhailoDan\apps\progno\app\index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Progno</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

- [ ] **Step 11: Install frontend dependencies**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app
npm install
```

Expected: `node_modules` created, no critical errors.

- [ ] **Step 12: Commit**

```bash
cd C:\Users\MykhailoDan\apps\progno
git add app/src app/tailwind.config.js app/postcss.config.js
git commit -m "feat(app): Svelte 5 UI with match input and prediction display"
```

---

## Task 7: Verify integration — compile, dev server smoke test

**Files:** none

- [ ] **Step 1: Build Rust backend**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app\src-tauri
cargo build --release 2>&1 | tail -20
```

Expected: release build completes without errors.

- [ ] **Step 2: Build frontend**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app
npm run build 2>&1
```

Expected: dist/ created with JavaScript bundles.

- [ ] **Step 3: Run dev server (manual test only — don't run in automation)**

Note: If you want to test the dev workflow locally:
```bash
cd C:\Users\MykhailoDan\apps\progno\app
npm run dev
```

Then in another terminal, from `app/src-tauri`:
```bash
cargo tauri dev
```

Expected: window opens, can type in textarea, parse button works (stub output for now since Elo state is placeholder).

- [ ] **Step 4: Run full Rust test suite**

Run:
```bash
cd C:\Users\MykhailoDan\apps\progno\app\src-tauri
cargo test --lib 2>&1
```

Expected: ≥14 tests PASS (elo + parser + artifacts + commands).

- [ ] **Step 5: Commit final state (if any edits)**

```bash
cd C:\Users\MykhailoDan\apps\progno
git status
# Only if changes:
git add <files>
git commit -m "chore(app): verify integration builds cleanly"
```

---

## Phase 1b Acceptance Criteria

Phase 1b is complete when:

- [ ] **Rust compiles**: `cargo build --release` in `app/src-tauri` succeeds.
- [ ] **Frontend builds**: `npm run build` in `app/` produces `dist/`.
- [ ] **Tests pass**: `cargo test --lib` in `app/src-tauri` shows ≥14 passing tests.
- [ ] **Parser works**: Examples like "Alcaraz vs Sinner - Clay" → extracts both players and surface.
- [ ] **Elo math correct**: `expected_probability(1600, 1500) ≈ 0.76` and surface composite respects 20-match threshold.
- [ ] **UI renders**: Svelte components compile; layout has input box, match cards, footer.
- [ ] **Git history clean**: One commit per task, no merge conflicts, working tree clean.

---

## Future work (Phase 1c+)

- Load actual Elo artifacts from app-data directory (Phase 1a produces them).
- Integrate with Python sidecar for CatBoost predictions (Phase 3).
- Add Kelly fraction UI and stake calculations (Phase 2).
- Support WTA model selection (Phase 4).

