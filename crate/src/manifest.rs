use std::collections::HashSet;

use serde::Deserialize;

use crate::geometry::Face;
use crate::lod::TileId;
use crate::tiles::{DATASET_VERSION, TILE_SIZE};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawManifest {
    version: String,
    tile_size: u32,
    levels: Vec<RawLevel>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawLevel {
    level: u8,
    geometric_error: f64,
    /// `[face, x, y, minHeight, maxHeight]` per tile.
    tiles: Vec<(String, u32, u32, f64, f64)>,
}

/// The parts of `v{N}/meta/manifest.json` the engine uses (PLAN.md §4.2).
#[derive(Debug)]
pub struct Manifest {
    /// Terrain geometric error in metres, indexed by level.
    pub geometric_error_m: Vec<f64>,
    available: HashSet<TileId>,
}

impl Manifest {
    pub fn parse(json: &str) -> Result<Self, String> {
        let raw: RawManifest = serde_json::from_str(json).map_err(|e| format!("manifest: {e}"))?;
        if raw.version != DATASET_VERSION {
            return Err(format!("manifest is {}, engine reads {DATASET_VERSION}", raw.version));
        }
        if raw.tile_size != TILE_SIZE {
            return Err(format!("manifest tile size {} != {TILE_SIZE}", raw.tile_size));
        }
        let mut available = HashSet::new();
        let mut geometric_error_m = Vec::with_capacity(raw.levels.len());
        for (index, level) in raw.levels.iter().enumerate() {
            if level.level as usize != index {
                return Err(format!("manifest level {} listed at position {index}", level.level));
            }
            geometric_error_m.push(level.geometric_error);
            for (face, x, y, _, _) in &level.tiles {
                available.insert(tile_id(level.level, face, *x, *y)?);
            }
        }
        if let Some(face) = Face::ALL.into_iter().find(|&face| !available.contains(&TileId::root(face))) {
            return Err(format!("manifest has no level-0 tile for {}", face.key()));
        }
        Ok(Self { geometric_error_m, available })
    }

    pub fn contains(&self, tile: TileId) -> bool {
        self.available.contains(&tile)
    }

    pub fn max_level(&self) -> u8 {
        self.geometric_error_m.len().saturating_sub(1) as u8
    }

    #[cfg(test)]
    pub fn tile_count(&self) -> usize {
        self.available.len()
    }
}

fn tile_id(level: u8, face: &str, x: u32, y: u32) -> Result<TileId, String> {
    let bad = || format!("manifest level {level}: bad tile {face} {x} {y}");
    let face = Face::from_key(face).ok_or_else(bad)?;
    let side = 1u32.checked_shl(u32::from(level)).ok_or_else(bad)?;
    if x >= side || y >= side {
        return Err(bad());
    }
    Ok(TileId { face, level, x, y })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOTS: &str = r#"["px",0,0,-1,1], ["nx",0,0,-1,1], ["py",0,0,-1,1], ["ny",0,0,-1,1], ["pz",0,0,-1,1], ["nz",0,0,-1,1]"#;

    fn manifest(version: &str, levels: &str) -> String {
        format!(r#"{{"version": "{version}", "tileSize": 257, "dem": {{"scale": 0.5}}, "levels": [{levels}]}}"#)
    }

    #[test]
    fn parses_availability_and_errors() {
        let json = manifest(
            "v2",
            &format!(
                r#"{{"level": 0, "geometricError": 6826.5, "tiles": [{ROOTS}]}},
                   {{"level": 1, "geometricError": 3634.3, "tiles": [["px",1,0,-5,5], ["nz",0,1,-5,5]]}}"#
            ),
        );
        let m = Manifest::parse(&json).unwrap();
        assert_eq!(m.geometric_error_m, [6826.5, 3634.3]);
        assert_eq!(m.max_level(), 1);
        assert_eq!(m.tile_count(), 8);
        assert!(m.contains(TileId { face: Face::Px, level: 1, x: 1, y: 0 }));
        assert!(m.contains(TileId { face: Face::Nz, level: 1, x: 0, y: 1 }));
        assert!(!m.contains(TileId { face: Face::Px, level: 1, x: 0, y: 0 }));
    }

    #[test]
    fn rejects_inconsistent_manifests() {
        let roots = format!(r#"{{"level": 0, "geometricError": 1, "tiles": [{ROOTS}]}}"#);
        assert!(Manifest::parse(&manifest("v1", &roots)).unwrap_err().contains("v1"));
        assert!(Manifest::parse(&manifest("v2", r#"{"level": 0, "geometricError": 1, "tiles": [["px",0,0,0,0]]}"#))
            .unwrap_err()
            .contains("level-0"));
        let bad = |tile: &str| {
            manifest("v2", &format!(r#"{roots}, {{"level": 1, "geometricError": 1, "tiles": [{tile}]}}"#))
        };
        assert!(Manifest::parse(&bad(r#"["px",1,1,0,0]"#)).is_ok());
        assert!(Manifest::parse(&bad(r#"["qx",0,0,0,0]"#)).is_err());
        assert!(Manifest::parse(&bad(r#"["px",2,0,0,0]"#)).is_err());
        assert!(Manifest::parse(&bad(r#"["px",0]"#)).is_err());
        assert!(Manifest::parse(&bad(r#""px 0 0 0 0""#)).is_err());
        assert!(Manifest::parse(&manifest("v2", r#"{"level": 1, "geometricError": 1, "tiles": []}"#)).is_err());
        assert!(Manifest::parse("not json").is_err());
    }

    /// The manifest the pipeline built, when present (it lives in the git-ignored `data/`).
    #[test]
    fn parses_the_built_v2_manifest() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/tiles/v2/meta/manifest.json");
        let Ok(json) = std::fs::read_to_string(path) else { return };
        let m = Manifest::parse(&json).unwrap();
        assert_eq!(m.max_level(), 7);
        assert_eq!(m.tile_count(), 6 + 24 + 96 + 384 + 16 + 16 + 48 + 124);
    }
}
