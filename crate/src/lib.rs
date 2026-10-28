mod camera;
mod geometry;
// Canvas surfaces only exist on wasm32; gating lets `cargo test` build on the host.
#[cfg(target_arch = "wasm32")]
mod renderer;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
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

#[wasm_bindgen]
pub async fn ping_r2_asset(asset_name: &str) -> Result<JsValue, JsValue> {
    let window = web_sys::window().expect("no global `window` exists");
    let url = format!("/assets/{}", asset_name);
    console::log_1(&format!("Requesting lunar asset from R2 edge: {}", url).into());

    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(&url)).await?;
    let resp: web_sys::Response = resp_value.dyn_into()?;

    console::log_1(&format!("R2 Response Status: {}", resp.status()).into());
    Ok(JsValue::from_f64(resp.status() as f64))
}
