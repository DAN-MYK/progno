use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::Manager;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BetRecord {
    pub id: String,
    pub date: String,
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
    pub tournament: Option<String>,
    /// "a" | "b"
    pub bet_on: String,
    pub our_prob: f64,
    pub odds: f64,
    pub stake: f64,
    /// "win" | "loss" | "void" — None means pending
    pub result: Option<String>,
    /// profit (positive) or loss (negative) after result is set
    pub pnl: Option<f64>,
}

impl BetRecord {
    fn computed_pnl(&self) -> Option<f64> {
        self.result.as_deref().map(|r| match r {
            "win" => self.stake * (self.odds - 1.0),
            "loss" => -self.stake,
            _ => 0.0, // void
        })
    }
}

fn bets_path(app_data_dir: &PathBuf) -> PathBuf {
    app_data_dir.join("progno_bets.json")
}

fn load_bets(path: &PathBuf) -> Result<Vec<BetRecord>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let raw = std::fs::read_to_string(path).context("read bets file")?;
    serde_json::from_str(&raw).context("parse bets JSON")
}

fn save_bets(path: &PathBuf, bets: &[BetRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(bets)?;
    std::fs::write(path, raw).context("write bets file")
}

#[tauri::command]
pub fn add_bet(
    record: BetRecord,
    app_handle: tauri::AppHandle,
) -> Result<BetRecord, String> {
    let dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    let path = bets_path(&dir);
    let mut bets = load_bets(&path).map_err(|e| e.to_string())?;
    bets.push(record.clone());
    save_bets(&path, &bets).map_err(|e| e.to_string())?;
    Ok(record)
}

#[tauri::command]
pub fn get_bets(app_handle: tauri::AppHandle) -> Result<Vec<BetRecord>, String> {
    let dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    load_bets(&bets_path(&dir)).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn update_bet_result(
    id: String,
    result: String,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    let path = bets_path(&dir);
    let mut bets = load_bets(&path).map_err(|e| e.to_string())?;

    let bet = bets
        .iter_mut()
        .find(|b| b.id == id)
        .ok_or_else(|| format!("Bet {id} not found"))?;

    bet.result = Some(result);
    bet.pnl = bet.computed_pnl();

    save_bets(&path, &bets).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_bet(id: String, app_handle: tauri::AppHandle) -> Result<(), String> {
    let dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    let path = bets_path(&dir);
    let mut bets = load_bets(&path).map_err(|e| e.to_string())?;
    let before = bets.len();
    bets.retain(|b| b.id != id);
    if bets.len() == before {
        return Err(format!("Bet {id} not found"));
    }
    save_bets(&path, &bets).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_bet(id: &str, result: Option<&str>) -> BetRecord {
        BetRecord {
            id: id.to_string(),
            date: "2026-04-25".to_string(),
            player_a: "Alcaraz".to_string(),
            player_b: "Sinner".to_string(),
            surface: "Clay".to_string(),
            tournament: None,
            bet_on: "a".to_string(),
            our_prob: 0.65,
            odds: 1.80,
            stake: 50.0,
            result: result.map(str::to_string),
            pnl: None,
        }
    }

    #[test]
    fn test_pnl_win() {
        let mut b = make_bet("1", Some("win"));
        b.pnl = b.computed_pnl();
        assert!((b.pnl.unwrap() - 40.0).abs() < 0.01); // 50 * (1.80 - 1.0)
    }

    #[test]
    fn test_pnl_loss() {
        let mut b = make_bet("1", Some("loss"));
        b.pnl = b.computed_pnl();
        assert!((b.pnl.unwrap() + 50.0).abs() < 0.01);
    }

    #[test]
    fn test_pnl_void() {
        let mut b = make_bet("1", Some("void"));
        b.pnl = b.computed_pnl();
        assert_eq!(b.pnl.unwrap(), 0.0);
    }

    #[test]
    fn test_pnl_pending() {
        let b = make_bet("1", None);
        assert!(b.computed_pnl().is_none());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = bets_path(&dir.path().to_path_buf());
        let bet = make_bet("abc", None);
        save_bets(&path, &[bet.clone()]).unwrap();
        let loaded = load_bets(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "abc");
    }

    #[test]
    fn test_load_missing_file() {
        let dir = TempDir::new().unwrap();
        let path = bets_path(&dir.path().to_path_buf());
        let loaded = load_bets(&path).unwrap();
        assert!(loaded.is_empty());
    }
}
