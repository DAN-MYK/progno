use std::os::unix::fs::symlink;
use std::path::PathBuf;

fn main() {
    // For onedir PyInstaller sidecar: symlink _internal/ next to the sidecar binary
    // that tauri copies into target/{triple}/{profile}/.
    // OUT_DIR = …/{triple}/{profile}/build/{pkg}/out — four parents up = profile dir.
    if let (Ok(out_dir), Ok(manifest_dir)) = (
        std::env::var("OUT_DIR"),
        std::env::var("CARGO_MANIFEST_DIR"),
    ) {
        // OUT_DIR = …/debug/build/{pkg}/out  →  3 parents up = debug/
        let profile_dir = PathBuf::from(&out_dir)
            .parent().and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf());
        if let Some(dir) = profile_dir {
            let dest = dir.join("_internal");
            // CARGO_MANIFEST_DIR = app/src-tauri; two parents up = repo root
            let repo_root = PathBuf::from(&manifest_dir)
                .parent().and_then(|p| p.parent())
                .map(|p| p.to_path_buf())
                .unwrap_or_default();
            let src = repo_root.join("sidecar/dist/progno-sidecar/_internal");
            let _ = std::fs::remove_file(&dest);
            let _ = symlink(&src, &dest);
        }
    }
    tauri_build::build()
}
