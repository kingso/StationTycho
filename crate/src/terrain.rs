use std::collections::HashMap;

use glam::DVec3;

use crate::geometry::{tile_face_bounds, unit_to_cube, MOON_RADIUS};
use crate::lod::{tile_containing, TileId, MAX_TERRAIN_HEIGHT, TILE_SEGMENTS};
use crate::manifest::DemQuantization;
use crate::tiles::TILE_SIZE;

/// DEM texels between neighbouring grid vertices.
const GRID_STEP: usize = ((TILE_SIZE - 1) / TILE_SEGMENTS) as usize;
const GRID_SIDE: usize = TILE_SEGMENTS as usize + 1;

/// The quantized DEM values under a tile's grid vertices, row-major (`GRID_SIDE²`), from a 257×257 `u16` LE tile.
pub fn grid_heights(dem: &[u8]) -> Vec<u16> {
    let texel = |x: usize, y: usize| {
        let i = (y * TILE_SIZE as usize + x) * 2;
        u16::from_le_bytes([dem[i], dem[i + 1]])
    };
    (0..GRID_SIDE).flat_map(|j| (0..GRID_SIDE).map(move |i| texel(i * GRID_STEP, j * GRID_STEP))).collect()
}

/// Height in DEM units at tile-local `(s, t)` in `[0, 1]²` on the rendered triangles: each cell is split along
/// its `(i, j)`–`(i + 1, j + 1)` diagonal, like `geometry::grid_indices`.
pub fn interpolate(grid: &[u16], s: f64, t: f64) -> f64 {
    let segments = TILE_SEGMENTS as f64;
    let (x, y) = ((s.clamp(0.0, 1.0) * segments), (t.clamp(0.0, 1.0) * segments));
    let (i, j) = ((x as usize).min(GRID_SIDE - 2), (y as usize).min(GRID_SIDE - 2));
    let (fx, fy) = (x - i as f64, y - j as f64);
    let h = |di: usize, dj: usize| f64::from(grid[(j + dj) * GRID_SIDE + i + di]);
    let (a, b, c, d) = (h(0, 0), h(1, 0), h(0, 1), h(1, 1));
    if fx >= fy {
        a + fx * (b - a) + fy * (d - b)
    } else {
        a + fy * (c - a) + fx * (d - c)
    }
}

/// Terrain heights of the tiles being drawn, so picking and the camera follow exactly what is on screen.
pub struct DrawnTerrain<'a> {
    tiles: HashMap<TileId, &'a [u16]>,
    max_level: u8,
    dem: DemQuantization,
}

impl<'a> DrawnTerrain<'a> {
    pub fn new(tiles: impl IntoIterator<Item = (TileId, &'a [u16])>, dem: DemQuantization) -> Self {
        let tiles: HashMap<_, _> = tiles.into_iter().collect();
        let max_level = tiles.keys().map(|t| t.level).max().unwrap_or(0);
        Self { tiles, max_level, dem }
    }

    /// Height above the reference sphere in metres under unit direction `dir`, if a drawn tile covers it.
    pub fn height_at(&self, dir: DVec3) -> Option<f64> {
        let (face, u, v) = unit_to_cube(dir);
        (0..=self.max_level).rev().find_map(|level| {
            let tile = tile_containing(face, u, v, level);
            let grid = self.tiles.get(&tile)?;
            let (u0, u1, v0, v1) = tile_face_bounds(tile.level, tile.x, tile.y)?;
            let raw = interpolate(grid, (u - u0) / (u1 - u0), (v - v0) / (v1 - v0));
            Some(raw * self.dem.scale + self.dem.offset)
        })
    }
}

/// First point where the ray from `origin` along unit `dir` meets the terrain, or `None` if it misses. Heights come
/// from `height_at` (metres above the reference sphere; `None` counts as 0). Steps are half the ray's current
/// height above the ground, which cannot jump through slopes up to about 63°; the crossing is then bisected.
pub fn ray_terrain(origin: DVec3, dir: DVec3, height_at: impl Fn(DVec3) -> Option<f64>) -> Option<DVec3> {
    const MIN_STEP: f64 = 0.01;
    const MAX_STEPS: usize = 100_000;
    let outer = MOON_RADIUS + MAX_TERRAIN_HEIGHT;
    let b = origin.dot(dir);
    let disc = b * b - (origin.length_squared() - outer * outer);
    if disc < 0.0 {
        return None;
    }
    let (enter, exit) = (-b - disc.sqrt(), -b + disc.sqrt());
    if exit < 0.0 {
        return None;
    }
    let gap = |t: f64| {
        let p = origin + dir * t;
        p.length() - MOON_RADIUS - height_at(p.normalize()).unwrap_or(0.0)
    };
    let mut t = enter.max(0.0);
    let mut g = gap(t);
    if g <= 0.0 {
        return Some(origin + dir * t);
    }
    for _ in 0..MAX_STEPS {
        let next = t + (0.5 * g).max(MIN_STEP);
        if next > exit {
            return None;
        }
        let g_next = gap(next);
        if g_next <= 0.0 {
            let (mut above, mut below) = (t, next);
            for _ in 0..50 {
                let mid = 0.5 * (above + below);
                if gap(mid) > 0.0 {
                    above = mid;
                } else {
                    below = mid;
                }
            }
            return Some(origin + dir * below);
        }
        (t, g) = (next, g_next);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{cube_to_unit, latlon_to_unit, Face};

    const DEM: DemQuantization = DemQuantization { scale: 0.5, offset: -9200.0 };

    fn dem_tile(value: impl Fn(usize, usize) -> u16) -> Vec<u8> {
        let n = TILE_SIZE as usize;
        (0..n * n).flat_map(|k| value(k % n, k / n).to_le_bytes()).collect()
    }

    #[test]
    fn grid_takes_the_texel_under_each_vertex() {
        let grid = grid_heights(&dem_tile(|x, y| (x + 1000 * y) as u16 % 60_000));
        assert_eq!(grid.len(), 65 * 65);
        assert_eq!(grid[0], 0);
        assert_eq!(grid[1], 4);
        assert_eq!(grid[64], 256);
        assert_eq!(grid[65], 4000);
        assert_eq!(grid[65 * 65 - 1], (256 + 256_000) as u16 % 60_000);
    }

    #[test]
    fn interpolation_follows_the_rendered_triangles() {
        // One raised vertex at grid (1, 1): its height spreads only over the triangles that use it.
        let mut grid = vec![100u16; 65 * 65];
        grid[65 + 1] = 200;
        let at = |i: f64, j: f64| interpolate(&grid, i / 64.0, j / 64.0);
        assert_eq!(at(1.0, 1.0), 200.0);
        assert_eq!(at(2.0, 2.0), 100.0);
        // Cell (0, 0) is split along (0, 0)-(1, 1): the raised corner is in both halves.
        assert!((at(0.75, 0.25) - 125.0).abs() < 1e-9);
        assert!((at(0.25, 0.75) - 125.0).abs() < 1e-9);
        // Cell (1, 0)'s diagonal runs (1, 0)-(2, 1), away from (1, 1): its lower half does not see it.
        assert_eq!(at(1.75, 0.25), 100.0);
        assert!((at(1.25, 0.75) - 150.0).abs() < 1e-9);
        // Outside [0, 1] clamps to the edge.
        assert_eq!(interpolate(&grid, -1.0, 2.0), 100.0);
    }

    #[test]
    fn drawn_terrain_uses_the_tile_covering_each_point() {
        let flat = |value: u16| vec![value; 65 * 65];
        let (coarse, fine) = (flat(20_000), flat(20_100));
        let coarse_tile = TileId { face: Face::Px, level: 1, x: 0, y: 0 };
        let fine_tile = TileId { face: Face::Px, level: 2, x: 2, y: 2 };
        let terrain = DrawnTerrain::new([(coarse_tile, &coarse[..]), (fine_tile, &fine[..])], DEM);
        // (-0.5, -0.5) is in the level-1 tile, (0.25, 0.25) in the level-2 one; (0.75, 0.75) is in no drawn tile.
        assert_eq!(terrain.height_at(cube_to_unit(Face::Px, -0.5, -0.5)), Some(20_000.0 * 0.5 - 9200.0));
        assert_eq!(terrain.height_at(cube_to_unit(Face::Px, 0.25, 0.25)), Some(20_100.0 * 0.5 - 9200.0));
        assert_eq!(terrain.height_at(cube_to_unit(Face::Px, 0.75, 0.75)), None);
        assert_eq!(terrain.height_at(cube_to_unit(Face::Nz, 0.0, 0.0)), None);
    }

    #[test]
    fn rays_hit_flat_terrain_at_its_height() {
        let dir = latlon_to_unit(-0.756, -0.198);
        let origin = dir * (MOON_RADIUS + 50_000.0);
        let hit = ray_terrain(origin, -dir, |_| Some(-2_500.0)).unwrap();
        assert!((hit.length() - (MOON_RADIUS - 2_500.0)).abs() < 1e-3);
        assert!(hit.normalize().angle_between(dir) < 1e-12);
    }

    #[test]
    fn grazing_rays_stop_at_a_hill_in_front() {
        let ground = latlon_to_unit(0.0, 0.0);
        let east = DVec3::Y;
        let origin = ground * (MOON_RADIUS + 1_000.0);
        // Looking east, 0.05 rad down; a 2 km wall starts 5 km away, where the ray is still about 760 m up.
        let dir = (east - ground * 0.05).normalize();
        let wall_angle = 5_000.0 / MOON_RADIUS;
        let terrain = |p: DVec3| Some(if p.angle_between(ground) > wall_angle { 2_000.0 } else { 0.0 });
        let hit = ray_terrain(origin, dir, terrain).unwrap();
        // The hit is where the ray crosses the wall's edge (5 km of ground arc; about 5,008 m along the raised,
        // tilted ray).
        let angle = hit.angle_between(ground);
        assert!((angle - wall_angle).abs() * MOON_RADIUS < 0.05, "{} m from the wall edge", (angle - wall_angle) * MOON_RADIUS);
        assert!((hit.distance(origin) - 5_008.4).abs() < 1.0, "{}", hit.distance(origin));
        // Without the wall, the same ray reaches the flat ground about 23 km away (the Moon curves away beneath it).
        let flat = ray_terrain(origin, dir, |_| Some(0.0)).unwrap().distance(origin);
        assert!((flat - 23_060.0).abs() < 100.0, "{flat}");
    }

    #[test]
    fn rays_into_space_miss() {
        let dir = latlon_to_unit(0.3, 1.2);
        let origin = dir * (MOON_RADIUS + 1_000.0);
        assert!(ray_terrain(origin, dir, |_| Some(0.0)).is_none());
        // Passing beside the Moon entirely.
        assert!(ray_terrain(DVec3::new(0.0, 2.0 * MOON_RADIUS, 0.0), DVec3::X, |_| Some(0.0)).is_none());
    }
}
