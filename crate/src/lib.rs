use wasm_bindgen::prelude::*;
use web_sys::console;

// Runs automatically when the WASM module is instantiated in the browser.
#[wasm_bindgen(start)]
pub fn start() {
    console::log_1(&"StationTycho engine initialized.".into());
}

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Station Tycho online. Welcome, {name}.")
}
