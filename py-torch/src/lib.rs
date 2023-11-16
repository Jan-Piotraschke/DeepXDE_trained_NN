use autocxx::prelude::*;
// wasm-pack uses wasm-bindgen to provide a bridge between the types of JavaScript and Rust
use wasm_bindgen::prelude::*;

// Including the C++ header file
include_cpp! {
    #include "input.h"
    safety!(unsafe_ffi)
    generate!("do_math")
}

/// Calls a function from a C++ library and prints the result.
/// This function is exposed to other crates if they use this library.
#[wasm_bindgen]
pub fn call_cpp_function(arg1: u32, arg2: u32) {
    let result = ffi::do_math(arg1, arg2);
    println!("Result: {}", result);
}

#[wasm_bindgen]
extern {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, py-torch!");
}
