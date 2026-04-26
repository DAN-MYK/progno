use serde::{Deserialize, Serialize};
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
    pub pnl: Option<f64>,
}

impl BetRecord {
    fn computed_pnl(&self) -> Option<f64> {
        self.result.as_deref().map(|r| match r {
            "win" => self.stake * (self.odds - 1.0),
            "loss" => -self.stake,
            _ => 0.0,
        })
    }
}

fn db_path(app_handle: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    let dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Cannot get app data dir: {e}"))?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create app data dir: {e}"))?;
    Ok(dir.join("progno_bets.db"))
}

fn open(path: &std::path::Path) -> Result<rusqlite::Connection, String> {
    let conn = rusqlite::Connection::open(path)
        .map_err(|e| format!("Cannot open bets DB: {e}"))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS bets (
            id          TEXT    PRIMARY KEY,
            date        TEXT    NOT NULL,
            player_a    TEXT    NOT NULL,
            player_b    TEXT    NOT NULL,
            surface     TEXT    NOT NULL,
            tournament  TEXT,
            bet_on      TEXT    NOT NULL,
            our_prob    REAL    NOT NULL,
            odds        REAL    NOT NULL,
            stake       REAL    NOT NULL,
            result      TEXT,
            pnl         REAL
        );",
    )
    .map_err(|e| format!("Cannot init bets table: {e}"))?;
    Ok(conn)
}

/// One-time migration: if old progno_bets.json exists and the DB is empty, import records.
fn maybe_migrate_json(conn: &rusqlite::Connection, dir: &std::path::Path) {
    let json_path = dir.join("progno_bets.json");
    if !json_path.exists() {
        return;
    }
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM bets", [], |r| r.get(0))
        .unwrap_or(0);
    if count > 0 {
        return;
    }
    let raw = match std::fs::read_to_string(&json_path) {
        Ok(s) => s,
        Err(_) => return,
    };
    let records: Vec<BetRecord> = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return,
    };
    for r in &records {
        let _ = conn.execute(
            "INSERT OR IGNORE INTO bets
             (id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
            rusqlite::params![
                r.id, r.date, r.player_a, r.player_b, r.surface, r.tournament,
                r.bet_on, r.our_prob, r.odds, r.stake, r.result, r.pnl,
            ],
        );
    }
    eprintln!("[bets] migrated {} records from JSON", records.len());
}

fn row_to_bet(row: &rusqlite::Row<'_>) -> rusqlite::Result<BetRecord> {
    Ok(BetRecord {
        id:         row.get(0)?,
        date:       row.get(1)?,
        player_a:   row.get(2)?,
        player_b:   row.get(3)?,
        surface:    row.get(4)?,
        tournament: row.get(5)?,
        bet_on:     row.get(6)?,
        our_prob:   row.get(7)?,
        odds:       row.get(8)?,
        stake:      row.get(9)?,
        result:     row.get(10)?,
        pnl:        row.get(11)?,
    })
}

#[tauri::command]
pub fn add_bet(record: BetRecord, app_handle: tauri::AppHandle) -> Result<BetRecord, String> {
    let path = db_path(&app_handle)?;
    let conn = open(&path)?;
    maybe_migrate_json(&conn, path.parent().unwrap_or(std::path::Path::new(".")));
    conn.execute(
        "INSERT OR REPLACE INTO bets
         (id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl)
         VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
        rusqlite::params![
            record.id, record.date, record.player_a, record.player_b,
            record.surface, record.tournament, record.bet_on, record.our_prob,
            record.odds, record.stake, record.result, record.pnl,
        ],
    )
    .map_err(|e| format!("Failed to insert bet: {e}"))?;
    Ok(record)
}

#[tauri::command]
pub fn get_bets(app_handle: tauri::AppHandle) -> Result<Vec<BetRecord>, String> {
    let path = db_path(&app_handle)?;
    let conn = open(&path)?;
    maybe_migrate_json(&conn, path.parent().unwrap_or(std::path::Path::new(".")));
    let mut stmt = conn
        .prepare(
            "SELECT id,date,player_a,player_b,surface,tournament,bet_on,
                    our_prob,odds,stake,result,pnl
             FROM bets ORDER BY date DESC, rowid DESC",
        )
        .map_err(|e| format!("prepare: {e}"))?;
    let records = stmt
        .query_map([], row_to_bet)
        .map_err(|e| format!("query: {e}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("collect: {e}"))?;
    Ok(records)
}

#[tauri::command]
pub fn update_bet_result(
    id: String,
    result: String,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let path = db_path(&app_handle)?;
    let conn = open(&path)?;
    let (stake, odds): (f64, f64) = conn
        .query_row("SELECT stake, odds FROM bets WHERE id=?1", [&id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .map_err(|e| format!("Bet {id} not found: {e}"))?;
    let mut dummy = BetRecord {
        id: id.clone(), date: String::new(), player_a: String::new(),
        player_b: String::new(), surface: String::new(), tournament: None,
        bet_on: String::new(), our_prob: 0.0, odds, stake,
        result: Some(result.clone()), pnl: None,
    };
    dummy.pnl = dummy.computed_pnl();
    conn.execute(
        "UPDATE bets SET result=?1, pnl=?2 WHERE id=?3",
        rusqlite::params![result, dummy.pnl, id],
    )
    .map_err(|e| format!("Failed to update bet: {e}"))?;
    Ok(())
}

#[tauri::command]
pub fn delete_bet(id: String, app_handle: tauri::AppHandle) -> Result<(), String> {
    let path = db_path(&app_handle)?;
    let conn = open(&path)?;
    let n = conn
        .execute("DELETE FROM bets WHERE id=?1", [&id])
        .map_err(|e| format!("Failed to delete bet: {e}"))?;
    if n == 0 {
        return Err(format!("Bet {id} not found"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmp_conn(dir: &std::path::Path) -> rusqlite::Connection {
        let path = dir.join("test.db");
        open(&path).unwrap()
    }

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
        assert!((b.pnl.unwrap() - 40.0).abs() < 0.01);
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
    fn test_sqlite_insert_and_query() {
        let dir = TempDir::new().unwrap();
        let conn = tmp_conn(dir.path());
        let bet = make_bet("abc", None);
        conn.execute(
            "INSERT INTO bets (id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
            rusqlite::params![
                bet.id, bet.date, bet.player_a, bet.player_b, bet.surface,
                bet.tournament, bet.bet_on, bet.our_prob, bet.odds, bet.stake,
                bet.result, bet.pnl,
            ],
        ).unwrap();
        let mut stmt = conn.prepare(
            "SELECT id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl FROM bets"
        ).unwrap();
        let loaded: Vec<BetRecord> = stmt.query_map([], row_to_bet).unwrap()
            .collect::<Result<_, _>>().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "abc");
    }

    #[test]
    fn test_sqlite_delete() {
        let dir = TempDir::new().unwrap();
        let conn = tmp_conn(dir.path());
        let bet = make_bet("xyz", None);
        conn.execute(
            "INSERT INTO bets (id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
            rusqlite::params![
                bet.id, bet.date, bet.player_a, bet.player_b, bet.surface,
                bet.tournament, bet.bet_on, bet.our_prob, bet.odds, bet.stake,
                bet.result, bet.pnl,
            ],
        ).unwrap();
        let n = conn.execute("DELETE FROM bets WHERE id=?1", ["xyz"]).unwrap();
        assert_eq!(n, 1);
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM bets", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_json_migration() {
        let dir = TempDir::new().unwrap();
        let bet = make_bet("migrated", None);
        let json = serde_json::to_string_pretty(&vec![bet]).unwrap();
        std::fs::write(dir.path().join("progno_bets.json"), json).unwrap();

        let conn = tmp_conn(dir.path());
        maybe_migrate_json(&conn, dir.path());

        let count: i64 = conn.query_row("SELECT COUNT(*) FROM bets", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);
    }
}
