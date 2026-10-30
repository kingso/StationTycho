mod camera;
mod geometry;
mod lighting;
mod lod;
mod manifest;
// Canvas surfaces only exist on wasm32; gating lets `cargo test` build on the host.
#[cfg(target_arch = "wasm32")]
mod renderer;
mod streaming;
mod tiles;

use wasm_bindgen::prelude::*;
use web_sys::console;

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    // Set up panic hook so Rust panics show up nicely in the browser developer console
    console_error_panic_hook::set_once();
    console::log_1(&"StationTycho engine initialized.".into());
    Ok(())
}

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Welcome to Station Tycho, {}.", name)
}
