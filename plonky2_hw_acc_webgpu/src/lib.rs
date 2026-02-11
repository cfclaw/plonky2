pub mod context;
pub mod prover;
pub mod utils;

/// Log a message to the browser console (WASM) or stderr (native).
pub fn log_msg(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(msg));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        eprintln!("{}", msg);
    }
}
