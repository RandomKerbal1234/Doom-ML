use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let levels_dir = manifest_dir.join("src").join("levels");
    println!("cargo:rerun-if-changed={}", levels_dir.display());

    let mut level_paths = fs::read_dir(&levels_dir)
        .unwrap_or_else(|error| panic!("could not read {}: {error}", levels_dir.display()))
        .map(|entry| entry.expect("could not read level directory entry").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("lvl"))
        .collect::<Vec<_>>();
    level_paths.sort();
    if level_paths.is_empty() {
        panic!("no .lvl files found in {}", levels_dir.display());
    }

    let mut generated = String::from("static EMBEDDED_LEVELS: &[(&str, &str)] = &[\n");
    for path in level_paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("level filename must be valid UTF-8");
        generated.push_str(&format!(
            "    ({filename:?}, include_str!({:?})),\n",
            path.to_string_lossy()
        ));
    }
    generated.push_str("];\n");

    let output_path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("embedded_levels.rs");
    fs::write(output_path, generated).expect("could not write embedded level manifest");
}
