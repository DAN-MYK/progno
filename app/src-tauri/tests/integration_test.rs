// Integration scenarios moved to src/commands.rs as unit tests (#[cfg(test)] mod tests).
// Linking progno as a library pulls in Tauri DLLs that are unavailable at test runtime on Windows.
