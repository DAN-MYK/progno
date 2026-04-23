// Tests moved to src/commands.rs as unit tests (#[cfg(test)] mod tests).
// Integration tests require the library to compile without cfg(test),
// which pulls in Tauri DLLs that are unavailable at test runtime on this host.
