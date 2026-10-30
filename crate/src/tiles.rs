use crate::geometry::Face;

/// Tile edge length in pixels; neighbouring tiles share their border row/column (PLAN.md §4.2).
pub const TILE_SIZE: u32 = 257;

/// Dataset version prefix in R2 keys (v1 is the synthetic M3 test set).
pub const DATASET_VERSION: &str = "v2";

/// Tile layer in R2.
#[derive(Clone, Copy)]
pub enum Layer {
    /// RGBA8 sRGB, row-major; pixel (x, y) sits at face coordinates u = -1 + 2x/256, v = -1 + 2y/256.
    Albedo,
    /// Little-endian `u16` heights with the same pixel layout as `Albedo`.
    Dem,
}

/// Worker URL path for a tile, e.g. `/assets/v1/dem/px/0/0/0.bin`.
pub fn tile_path(layer: Layer, face: Face, level: u32, x: u32, y: u32) -> String {
    let (dir, ext) = match layer {
        Layer::Albedo => ("albedo", "rgba"),
        Layer::Dem => ("dem", "bin"),
    };
    format!("/assets/{DATASET_VERSION}/{dir}/{}/{level}/{x}/{y}.{ext}", face.key())
}

/// Worker URL path of the dataset manifest.
pub fn manifest_path() -> String {
    format!("/assets/{DATASET_VERSION}/meta/manifest.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_follow_r2_layout() {
        assert_eq!(tile_path(Layer::Dem, Face::Px, 0, 0, 0), "/assets/v2/dem/px/0/0/0.bin");
        assert_eq!(tile_path(Layer::Albedo, Face::Nz, 3, 5, 2), "/assets/v2/albedo/nz/3/5/2.rgba");
        assert_eq!(manifest_path(), "/assets/v2/meta/manifest.json");
    }
}
