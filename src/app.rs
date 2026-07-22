mod loader;

use loader::{ArrayData, FileSummary};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OpenedFile {
    summary: FileSummary,
    array: ArrayData,
    selected_entry: Option<String>,
}

#[tauri::command]
async fn load_array(path: String, entry: Option<String>) -> Result<ArrayData, String> {
    tauri::async_runtime::spawn_blocking(move || {
        loader::load_array(Path::new(&path), entry.as_deref())
    })
    .await
    .map_err(|err| format!("Array loading task failed: {err}"))?
}

fn open_path(path: PathBuf) -> Result<OpenedFile, String> {
    let summary = loader::inspect_file(&path)?;
    let selected_entry = if summary.kind == "npz" {
        summary.entries.first().cloned()
    } else {
        None
    };
    let array = loader::load_array(Path::new(&summary.path), selected_entry.as_deref())?;
    Ok(OpenedFile {
        summary,
        array,
        selected_entry,
    })
}

#[tauri::command]
async fn open_initial() -> Result<Option<OpenedFile>, String> {
    let Some(path) = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .find(|path| is_numpy_path(path))
    else {
        return Ok(None);
    };
    tauri::async_runtime::spawn_blocking(move || open_path(path).map(Some))
        .await
        .map_err(|err| format!("Initial loading task failed: {err}"))?
}

#[tauri::command]
async fn open_file(path: String) -> Result<OpenedFile, String> {
    tauri::async_runtime::spawn_blocking(move || open_path(PathBuf::from(path)))
        .await
        .map_err(|err| format!("File loading task failed: {err}"))?
}

#[tauri::command]
async fn pick_and_open() -> Result<Option<OpenedFile>, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("NumPy data", &["npy", "npz"])
            .pick_file()
        else {
            return Ok(None);
        };
        open_path(path).map(Some)
    })
    .await
    .map_err(|err| format!("File picker task failed: {err}"))?
}

fn is_numpy_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("npy" | "npz")
    )
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            load_array,
            open_initial,
            open_file,
            pick_and_open
        ])
        .run(tauri::generate_context!())
        .expect("failed to run npyz-viewer");
}
