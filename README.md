# npyz-viewer

Tauri + Rust で動く、NumPy `.npy` / `.npz` ファイル用のデスクトップビューアです。
フロントエンドに Node.js や npm は不要です。

## Features

- `.npy` / `.npz` のファイル選択、ドラッグ＆ドロップ、コマンドライン起動
- NPZ 内の配列を同じウィンドウで素早く切り替え
- dtype / shape / 要素数の表示
- ページング付きテーブル、値検索
- 数式による計算列（`$1`, `$2`, `sin`, `atan2`, `sqrt`, `pow` など）
- mean / std / quartile / skew / kurtosis / RMS と Pearson 相関行列
- line / scatter / histogram / box plot、linear / log 軸
- 1〜5 次多項式、指数、対数、べき乗 fitting
- R² / adjusted R² / RMSE / MAE、fitted curve と残差プロット
- 現在のページを CSV としてクリップボードへコピー
- bool、整数、float16/32/64、複素数、日時、byte/Unicode、void、scalar structured dtype に対応
- object dtype は安全のため pickle を実行せず、shape とプレースホルダーを表示

## Run

```bash
cargo run --release
cargo run --release -- /path/to/file.npy
cargo run --release -- /path/to/file.npz
```

Linux では Tauri の実行に WebKitGTK 4.1 が必要です。開発パッケージ名は
Debian/Ubuntu 系では通常 `libwebkit2gtk-4.1-dev` です。

Linuxでは、ディストリビューションに合わせてTauri版を自動選択できます。

```bash
./scripts/build_linux_npyz_viewer.sh
```

- WebKitGTK 4.1 + GLib 2.70以上: 標準のTauri 2版
- WebKitGTK 4.0: Rocky/RHEL 8向けTauri 1互換版

Rocky Linux 8では先に次をインストールしてください。

```bash
sudo dnf install webkit2gtk3-devel gtk3-devel openssl-devel
sudo dnf group install "Development Tools"
```

互換版は `src/loader.rs` と `ui/` を標準版と共有し、Tauriランチャーのみ
`compat/rocky8/` に分離しています。

## Test

```bash
cargo fmt --all -- --check
cargo check --locked
cargo test --locked
node --check ui/app.js
```

## Install scripts

- Linux: `./scripts/install_linux_npyz_viewer.sh`
- macOS: `./scripts/install_macos_npyz_viewer.sh`
- Windows: `powershell -ExecutionPolicy Bypass -File .\scripts\install_windows_npyz_viewer.ps1`

## Source layout

- `src/app.rs`: Tauri commands and application entry point
- `src/loader.rs`: NPY/NPZ parsing and serialization
- `compat/rocky8/`: WebKitGTK 4.0 / Tauri 1 compatibility shell
- `ui/`: HTML/CSS/JavaScript frontend
- `tauri.conf.json`: Tauri window and security configuration
- `npyzviewer.py`, `npyz2txt.py`, `npy2npz.py`: helper scripts
