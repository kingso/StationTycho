use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use glam::DVec3;

use crate::camera::OrbitCamera;
use crate::geometry::{cube_to_unit, tile_face_bounds, unit_to_cube, Face, MOON_RADIUS};
use crate::tiles::TILE_SIZE;

/// Grid cells along each tile edge; every tile is drawn as `TILE_SEGMENTS × TILE_SEGMENTS` quads (PLAN.md §3.4).
pub const TILE_SEGMENTS: u32 = 64;
/// The grid takes every `2^GRID_LEVEL_SHIFT`-th DEM texel, so its vertex spacing is that of the full-resolution
/// grid `GRID_LEVEL_SHIFT` levels up.
const GRID_LEVEL_SHIFT: i32 = ((TILE_SIZE - 1) / TILE_SEGMENTS).trailing_zeros() as i32;
/// Most tiles drawn in one frame; sizes the per-tile uniform buffer.
pub const MAX_VISIBLE_TILES: usize = 1024;
/// Highest terrain above the reference sphere: the top of the `u16` DEM range (PLAN.md §4.2).
pub const MAX_TERRAIN_HEIGHT: f64 = 10_800.0;
/// Deepest level the selector will ever reach, whatever the config says.
const MAX_LEVEL_LIMIT: u8 = 24;
/// `levels[].geometricError` of the v2 manifest in metres, until the engine reads the manifest.
const V2_TERRAIN_ERROR_M: [f64; 8] = [6826.5, 3634.3, 2034.0, 991.8, 631.4, 423.9, 312.4, 104.1];
/// Largest grid-cell half-diagonal relative to `angular_radius / TILE_SEGMENTS`, because the cube-to-sphere
/// mapping makes cells uneven. Checked by `mesh_error_bounds_every_cell`.
const CELL_SIZE_MARGIN: f64 = 1.3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileId {
    pub face: Face,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl TileId {
    pub fn root(face: Face) -> Self {
        Self { face, level: 0, x: 0, y: 0 }
    }

    pub fn parent(self) -> Option<Self> {
        (self.level > 0).then(|| Self { face: self.face, level: self.level - 1, x: self.x / 2, y: self.y / 2 })
    }

    pub fn children(self) -> [Self; 4] {
        let (face, level, x, y) = (self.face, self.level + 1, self.x * 2, self.y * 2);
        [
            Self { face, level, x, y },
            Self { face, level, x: x + 1, y },
            Self { face, level, x, y: y + 1 },
            Self { face, level, x: x + 1, y: y + 1 },
        ]
    }

    /// Whether `self` is `ancestor` or lies inside it.
    #[cfg(test)]
    fn is_within(self, ancestor: Self) -> bool {
        let shift = self.level.wrapping_sub(ancestor.level);
        self.face == ancestor.face
            && self.level >= ancestor.level
            && self.x >> shift == ancestor.x
            && self.y >> shift == ancestor.y
    }
}

/// Bounding cone of a tile on the reference sphere.
#[derive(Clone, Copy, Debug)]
struct TileBounds {
    /// Unit direction of the tile centre.
    center_dir: DVec3,
    /// Largest angle between `center_dir` and any point of the tile, in radians.
    angular_radius: f64,
}

impl TileBounds {
    fn new(tile: TileId) -> Self {
        let (u0, u1, v0, v1) = tile_face_bounds(tile.level, tile.x, tile.y).expect("tile inside its quadtree");
        let (um, vm) = ((u0 + u1) * 0.5, (v0 + v1) * 0.5);
        let center_dir = cube_to_unit(tile.face, um, vm);
        // The tile is convex on the sphere, so its corners and edge midpoints bound it.
        let angular_radius = [(u0, v0), (u1, v0), (u0, v1), (u1, v1), (um, v0), (um, v1), (u0, vm), (u1, vm)]
            .into_iter()
            .map(|(u, v)| angle_between(center_dir, cube_to_unit(tile.face, u, v)))
            .fold(0.0, f64::max);
        Self { center_dir, angular_radius }
    }

    fn center(&self) -> DVec3 {
        self.center_dir * MOON_RADIUS
    }

    /// Radius of a sphere around `center()` that holds the tile's terrain at any height in the DEM range.
    fn radius(&self) -> f64 {
        2.0 * MOON_RADIUS * (self.angular_radius * 0.5).sin() + MAX_TERRAIN_HEIGHT
    }

    /// Largest gap between the reference sphere and the flat cells of the tile's grid.
    fn mesh_error(&self) -> f64 {
        let half_diagonal = self.angular_radius / TILE_SEGMENTS as f64 * CELL_SIZE_MARGIN;
        MOON_RADIUS * (1.0 - half_diagonal.cos())
    }
}

fn angle_between(a: DVec3, b: DVec3) -> f64 {
    a.cross(b).length().atan2(a.dot(b))
}

/// Camera quantities shared by every visibility and error test in one selection.
struct View {
    position: DVec3,
    right: DVec3,
    up: DVec3,
    forward: DVec3,
    tan_x: f64,
    tan_y: f64,
    /// Largest angle from the camera's nadir at which terrain can still rise above the horizon;
    /// `None` when the camera is inside the reference sphere.
    horizon_angle: Option<f64>,
    /// Converts `metres / distance` to pixels.
    pixels_per_radian: f64,
}

impl View {
    fn new(camera: &OrbitCamera, viewport_height: u32, aspect: f64) -> Self {
        let position = camera.position();
        let (right, up, forward) = camera.view_basis();
        let tan_y = (camera.fov_y * 0.5).tan();
        let r = position.length();
        // Sum of the camera's and the highest peak's horizon angles over the reference sphere.
        let horizon_angle = (r > MOON_RADIUS)
            .then(|| (MOON_RADIUS / r).acos() + (MOON_RADIUS / (MOON_RADIUS + MAX_TERRAIN_HEIGHT)).acos());
        Self {
            position,
            right,
            up,
            forward,
            tan_x: tan_y * aspect,
            tan_y,
            horizon_angle,
            pixels_per_radian: viewport_height.max(1) as f64 / (2.0 * tan_y),
        }
    }

    fn sees(&self, bounds: &TileBounds) -> bool {
        self.in_frustum(bounds.center() - self.position, bounds.radius()) && self.above_horizon(bounds)
    }

    /// Sphere test against the four side planes; the far plane is at infinity and the near plane is ignored.
    fn in_frustum(&self, rel: DVec3, radius: f64) -> bool {
        let depth = rel.dot(self.forward);
        depth + radius >= 0.0
            && rel.dot(self.right).abs() <= depth * self.tan_x + radius * (1.0 + self.tan_x * self.tan_x).sqrt()
            && rel.dot(self.up).abs() <= depth * self.tan_y + radius * (1.0 + self.tan_y * self.tan_y).sqrt()
    }

    fn above_horizon(&self, bounds: &TileBounds) -> bool {
        self.horizon_angle
            .map_or(true, |horizon| angle_between(bounds.center_dir, self.position) - bounds.angular_radius <= horizon)
    }

    /// Projected size in pixels of `geometric_error` at the tile's nearest possible point.
    fn screen_space_error(&self, bounds: &TileBounds, geometric_error: f64) -> f64 {
        let distance = (bounds.center().distance(self.position) - bounds.radius()).max(1.0);
        geometric_error * self.pixels_per_radian / distance
    }
}

#[derive(Clone, Debug)]
pub struct LodConfig {
    pub max_level: u8,
    pub screen_space_error_px: f64,
    pub max_visible_tiles: usize,
    /// Terrain geometric error in metres of each level's full-resolution grid against the finest data
    /// (`levels[].geometricError` of the manifest). Deeper levels halve the last entry per level; coarser ones
    /// double the first.
    pub terrain_error_m: Vec<f64>,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            max_level: 7,
            screen_space_error_px: 2.0,
            max_visible_tiles: MAX_VISIBLE_TILES,
            terrain_error_m: V2_TERRAIN_ERROR_M.to_vec(),
        }
    }
}

impl LodConfig {
    fn terrain_error(&self, level: i32) -> f64 {
        let (Some(&first), Some(&last)) = (self.terrain_error_m.first(), self.terrain_error_m.last()) else {
            return 0.0;
        };
        let levels = self.terrain_error_m.len() as i32;
        match level {
            l if l < 0 => first * 2f64.powi(-l),
            l if l < levels => self.terrain_error_m[l as usize],
            l => last * 0.5f64.powi(l + 1 - levels),
        }
    }

    /// Terrain error of a level-`level` tile as drawn: its grid has the vertex spacing of the full-resolution grid
    /// `GRID_LEVEL_SHIFT` levels up, so it has that level's error.
    pub fn drawn_error(&self, level: u8) -> f64 {
        self.terrain_error(i32::from(level) - GRID_LEVEL_SHIFT)
    }
}

/// Result of [`select_tiles`].
#[derive(Debug, Default)]
pub struct LodSelection {
    /// Non-overlapping tiles to draw, ordered by decreasing screen-space error (roughly near to far).
    pub tiles: Vec<TileId>,
    /// Tiles that must become ready before a selected tile can split, most urgent first.
    pub requests: Vec<TileId>,
}

/// Whether a tile's data can be drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileStatus {
    Ready,
    /// Not loaded yet, but it exists (or might).
    Pending,
    /// Not in the dataset, or it failed for good.
    Unavailable,
}

/// Split candidate ordered by screen-space error.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    error_px: f64,
    tile: TileId,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.error_px.total_cmp(&other.error_px).then_with(|| other.tile.cmp(&self.tile))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

struct Selector<'a> {
    view: View,
    config: &'a LodConfig,
    max_level: u8,
    threshold_px: f64,
    selected: HashMap<TileId, f64>,
    queue: BinaryHeap<Candidate>,
}

impl Selector<'_> {
    /// Selects `tile` and queues it for splitting if it is too coarse for the view.
    fn select(&mut self, tile: TileId, bounds: &TileBounds) {
        let error = self.config.drawn_error(tile.level).max(bounds.mesh_error());
        let error_px = self.view.screen_space_error(bounds, error);
        self.selected.insert(tile, error_px);
        if tile.level < self.max_level && error_px > self.threshold_px {
            self.queue.push(Candidate { error_px, tile });
        }
    }
}

/// Selects a non-overlapping set of visible, ready tiles covering the visible surface.
///
/// Starting from the six faces, it repeatedly splits the selected tile with the largest screen-space error,
/// replacing it by its visible children. A split waits until every visible child is ready, keeping the parent
/// meanwhile, so the surface never has holes; pending children are returned as requests. A tile with an
/// unavailable visible child never splits, which ends refinement where the dataset runs out. Refinement stops at
/// the first split that would exceed `max_visible_tiles`, so a tight budget goes to the most needed detail.
pub fn select_tiles(
    camera: &OrbitCamera,
    viewport_height: u32,
    aspect: f64,
    config: &LodConfig,
    status: impl Fn(TileId) -> TileStatus,
) -> LodSelection {
    let max_tiles = config.max_visible_tiles.max(Face::ALL.len());
    let mut selector = Selector {
        view: View::new(camera, viewport_height, aspect),
        config,
        max_level: config.max_level.min(MAX_LEVEL_LIMIT),
        threshold_px: config.screen_space_error_px.max(0.0),
        selected: HashMap::new(),
        queue: BinaryHeap::new(),
    };
    let mut requests = Vec::new();

    for face in Face::ALL {
        let root = TileId::root(face);
        let bounds = TileBounds::new(root);
        if !selector.view.sees(&bounds) {
            continue;
        }
        match status(root) {
            TileStatus::Ready => selector.select(root, &bounds),
            TileStatus::Pending => requests.push(Candidate { error_px: f64::INFINITY, tile: root }),
            TileStatus::Unavailable => {}
        }
    }

    while let Some(Candidate { error_px, tile }) = selector.queue.pop() {
        let children: Vec<_> = tile
            .children()
            .into_iter()
            .map(|child| (child, TileBounds::new(child)))
            .filter(|(_, bounds)| selector.view.sees(bounds))
            .collect();
        // A parent whose children are all culled is only visible through its looser bounds; keep it.
        if children.is_empty() {
            continue;
        }
        // Out of budget: stop, so detail is only ever added where the error is largest.
        if selector.selected.len() - 1 + children.len() > max_tiles {
            break;
        }
        let statuses: Vec<_> = children.iter().map(|&(child, _)| (child, status(child))).collect();
        if statuses.iter().any(|&(_, s)| s == TileStatus::Unavailable) {
            continue;
        }
        let pending: Vec<_> = statuses.into_iter().filter(|&(_, s)| s == TileStatus::Pending).collect();
        if !pending.is_empty() {
            requests.extend(pending.into_iter().map(|(tile, _)| Candidate { error_px, tile }));
            continue;
        }
        selector.selected.remove(&tile);
        for (child, bounds) in &children {
            selector.select(*child, bounds);
        }
    }

    let mut tiles: Vec<_> = selector.selected.into_iter().map(|(tile, error_px)| Candidate { error_px, tile }).collect();
    tiles.sort_unstable_by(|a, b| b.cmp(a));
    requests.sort_unstable_by(|a, b| b.cmp(a));
    LodSelection {
        tiles: tiles.into_iter().map(|c| c.tile).collect(),
        requests: requests.into_iter().map(|c| c.tile).collect(),
    }
}

/// The level-`level` tile of `face` containing face coordinates `(u, v)`.
pub fn tile_containing(face: Face, u: f64, v: f64, level: u8) -> TileId {
    let side = 1u32 << level;
    let index = |t: f64| (((t + 1.0) * 0.5 * side as f64).max(0.0) as u32).min(side - 1);
    TileId { face, level, x: index(u), y: index(v) }
}

/// Level of the coarsest tile in `selected` that shares an edge with `tile`, including across cube faces.
fn coarsest_neighbour_level(tile: TileId, selected: &HashSet<TileId>, max_level: u8) -> Option<u8> {
    let (u0, u1, v0, v1) = tile_face_bounds(tile.level, tile.x, tile.y)?;
    let (um, vm) = ((u0 + u1) * 0.5, (v0 + v1) * 0.5);
    let step = (u1 - u0) * 1e-3;
    // Edge midpoints, each with a step back into the tile; a coarser neighbour covers the whole edge.
    let edges = [((u0, vm), (step, 0.0)), ((u1, vm), (-step, 0.0)), ((um, v0), (0.0, step)), ((um, v1), (0.0, -step))];
    edges
        .into_iter()
        .filter_map(|((u, v), (du, dv))| {
            let edge = cube_to_unit(tile.face, u, v);
            let (face, nu, nv) = unit_to_cube(edge * 2.0 - cube_to_unit(tile.face, u + du, v + dv));
            (0..=max_level).map(|level| tile_containing(face, nu, nv, level)).find(|t| selected.contains(t))
        })
        .map(|t| t.level)
        .min()
}

/// Skirt depth in metres for each tile of a selection. Two tiles' edges differ by at most the sum of their drawn
/// errors, so a skirt reaching down by its own error plus its coarsest neighbour's closes any crack.
pub fn skirt_depths(tiles: &[TileId], config: &LodConfig) -> Vec<f64> {
    let selected: HashSet<_> = tiles.iter().copied().collect();
    let max_level = tiles.iter().map(|t| t.level).max().unwrap_or(0);
    tiles
        .iter()
        .map(|&tile| {
            let coarsest = coarsest_neighbour_level(tile, &selected, max_level).map_or(tile.level, |l| l.min(tile.level));
            config.drawn_error(tile.level) + config.drawn_error(coarsest)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{latlon_to_unit, tile_vertices};

    const ASPECT: f64 = 16.0 / 9.0;
    const HEIGHT_PX: u32 = 1080;
    const TYCHO: (f64, f64) = (crate::geometry::TYCHO_LAT_DEG, crate::geometry::TYCHO_LON_DEG);
    /// Camera distances to the target matching the HUD altitudes seen while zooming, from the start view down.
    const DISTANCES: [f64; 5] = [3.0 * MOON_RADIUS, 601_100.0, 69_300.0, 7_995.0, 1_000.0];
    const ELEVATIONS_DEG: [f64; 3] = [90.0, 30.0, 5.0];

    fn camera(distance: f64, elevation_deg: f64) -> OrbitCamera {
        let mut camera = OrbitCamera::new(TYCHO.0, TYCHO.1, distance);
        camera.elevation = elevation_deg.to_radians();
        camera
    }

    fn select(camera: &OrbitCamera, config: &LodConfig, status: impl Fn(TileId) -> TileStatus) -> LodSelection {
        select_tiles(camera, HEIGHT_PX, ASPECT, config, status)
    }

    fn all_ready(_: TileId) -> TileStatus {
        TileStatus::Ready
    }

    fn all_views() -> impl Iterator<Item = (f64, f64, OrbitCamera)> {
        DISTANCES.into_iter().flat_map(|d| ELEVATIONS_DEG.into_iter().map(move |e| (d, e, camera(d, e))))
    }

    fn tile_at(point: DVec3, level: u8) -> TileId {
        let (face, u, v) = unit_to_cube(point);
        tile_containing(face, u, v, level)
    }

    /// Surface points hit by rays through a 65×37 grid spanning the viewport.
    fn visible_surface_points(camera: &OrbitCamera) -> Vec<DVec3> {
        let (right, up, forward) = camera.view_basis();
        let origin = camera.position();
        let tan_y = (camera.fov_y * 0.5).tan();
        let mut points = Vec::new();
        for j in 0..=36 {
            for i in 0..=64 {
                let (sx, sy) = (-1.0 + i as f64 / 32.0, -1.0 + j as f64 / 18.0);
                let dir = (forward + right * (sx * tan_y * ASPECT) + up * (sy * tan_y)).normalize();
                let b = origin.dot(dir);
                let disc = b * b - (origin.length_squared() - MOON_RADIUS * MOON_RADIUS);
                if disc >= 0.0 && -b - disc.sqrt() > 0.0 {
                    points.push(origin + dir * (-b - disc.sqrt()));
                }
            }
        }
        points
    }

    fn assert_covers_view_without_overlap(camera: &OrbitCamera, tiles: &[TileId], label: &str) {
        for (i, a) in tiles.iter().enumerate() {
            for b in &tiles[i + 1..] {
                assert!(!a.is_within(*b) && !b.is_within(*a), "{label}: {a:?} overlaps {b:?}");
            }
        }
        let points = visible_surface_points(camera);
        assert!(!points.is_empty(), "{label}: camera sees no surface");
        for point in points {
            let (face, u, v) = unit_to_cube(point);
            let covered = tiles.iter().any(|t| {
                let (u0, u1, v0, v1) = tile_face_bounds(t.level, t.x, t.y).unwrap();
                t.face == face && (u0 - 1e-9..=u1 + 1e-9).contains(&u) && (v0 - 1e-9..=v1 + 1e-9).contains(&v)
            });
            assert!(covered, "{label}: hole at {face:?} u={u} v={v}");
        }
    }

    fn sample_tiles() -> impl Iterator<Item = TileId> {
        Face::ALL.into_iter().flat_map(|face| {
            (0..=7u8).flat_map(move |level| {
                let last = (1u32 << level) - 1;
                [(0, 0), (last / 2, last / 3), (last, last)].map(|(x, y)| TileId { face, level, x, y })
            })
        })
    }

    #[test]
    fn angular_radius_bounds_every_point_of_the_tile() {
        for tile in sample_tiles() {
            let bounds = TileBounds::new(tile);
            let (u0, u1, v0, v1) = tile_face_bounds(tile.level, tile.x, tile.y).unwrap();
            for j in 0..=32 {
                for i in 0..=32 {
                    let p = cube_to_unit(tile.face, u0 + (u1 - u0) * i as f64 / 32.0, v0 + (v1 - v0) * j as f64 / 32.0);
                    let angle = angle_between(bounds.center_dir, p);
                    assert!(angle <= bounds.angular_radius * (1.0 + 1e-9), "{tile:?}: {angle} > {}", bounds.angular_radius);
                }
            }
            // Tight enough to cull well: within 1% of the farthest grid vertex.
            let (origin, vertices) = tile_vertices(tile.face, tile.level, tile.x, tile.y, TILE_SEGMENTS).unwrap();
            let farthest = vertices
                .chunks(3)
                .map(|v| angle_between(bounds.center_dir, origin + DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)))
                .fold(0.0, f64::max);
            assert!(bounds.angular_radius <= farthest * 1.01, "{tile:?}: radius not tight");
        }
    }

    #[test]
    fn mesh_error_bounds_every_cell() {
        let side = TILE_SEGMENTS as usize + 1;
        for tile in sample_tiles() {
            let bounds = TileBounds::new(tile);
            let (origin, vertices) = tile_vertices(tile.face, tile.level, tile.x, tile.y, TILE_SEGMENTS).unwrap();
            let p: Vec<DVec3> =
                vertices.chunks(3).map(|v| origin + DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)).collect();
            let mut worst = 0.0f64;
            for j in 0..side - 1 {
                for i in 0..side - 1 {
                    let (a, b, c, d) = (p[j * side + i], p[j * side + i + 1], p[(j + 1) * side + i], p[(j + 1) * side + i + 1]);
                    // The diagonals' midpoints are the cell's deepest points below the sphere.
                    worst = worst.max(MOON_RADIUS - ((a + d) * 0.5).length()).max(MOON_RADIUS - ((b + c) * 0.5).length());
                }
            }
            // f32 vertex offsets add up to ~0.1 m of noise on level-0 tiles.
            assert!(bounds.mesh_error() + 0.1 >= worst, "{tile:?}: estimate {} < actual {worst}", bounds.mesh_error());
            assert!(bounds.mesh_error() <= worst * 2.0 + 0.1, "{tile:?}: estimate {} far above {worst}", bounds.mesh_error());
        }
    }

    #[test]
    fn selection_covers_every_view_without_holes_or_overlap() {
        let config = LodConfig::default();
        for (distance, elevation, camera) in all_views() {
            let selection = select(&camera, &config, all_ready);
            assert!(selection.requests.is_empty());
            assert_covers_view_without_overlap(&camera, &selection.tiles, &format!("{distance} m at {elevation}°"));
        }
    }

    #[test]
    fn partial_readiness_never_leaves_holes_and_requests_only_missing_tiles() {
        let config = LodConfig::default();
        let status = |t: TileId| match t.level <= 1 || (t.x * 7 + t.y * 13 + t.level as u32) % 4 != 0 {
            true => TileStatus::Ready,
            false => TileStatus::Pending,
        };
        for (distance, elevation, camera) in all_views() {
            let selection = select(&camera, &config, status);
            assert!(selection.tiles.iter().all(|&t| status(t) == TileStatus::Ready));
            assert!(selection.requests.iter().all(|&t| status(t) == TileStatus::Pending));
            assert_covers_view_without_overlap(&camera, &selection.tiles, &format!("{distance} m at {elevation}°"));
        }
    }

    #[test]
    fn parent_stays_until_all_visible_children_are_ready() {
        let camera = camera(1_000.0, 90.0);
        let missing = tile_at(camera.target(), 1);
        let selection = select(&camera, &LodConfig::default(), |t| match t == missing {
            true => TileStatus::Pending,
            false => TileStatus::Ready,
        });

        let root = TileId::root(missing.face);
        assert!(selection.tiles.contains(&root));
        assert!(selection.tiles.iter().all(|t| t.face != missing.face || t.level == 0));
        assert_eq!(selection.requests.first(), Some(&missing));
        assert_covers_view_without_overlap(&camera, &selection.tiles, "one missing child");
    }

    #[test]
    fn refinement_ends_where_the_dataset_does() {
        // Like v2: levels 0-3 everywhere, deeper levels only as complete sibling sets around Tycho.
        let tycho = latlon_to_unit(TYCHO.0.to_radians(), TYCHO.1.to_radians());
        let status = |t: TileId| match t.level <= 3 || t.parent() == Some(tile_at(tycho, t.level - 1)) {
            true => TileStatus::Ready,
            false => TileStatus::Unavailable,
        };
        let config = LodConfig::default();
        for elevation in ELEVATIONS_DEG {
            let away = {
                let mut camera = OrbitCamera::new(20.0, 60.0, 1_000.0);
                camera.elevation = elevation.to_radians();
                camera
            };
            let selection = select(&away, &config, status);
            assert!(selection.requests.is_empty());
            assert!(selection.tiles.iter().all(|t| t.level <= 3));
            assert_covers_view_without_overlap(&away, &selection.tiles, &format!("away at {elevation}°"));

            let over_tycho = camera(1_000.0, elevation);
            let selection = select(&over_tycho, &config, status);
            assert!(selection.requests.is_empty());
            assert!(selection.tiles.contains(&tile_at(tycho, 7)), "{elevation}°: Tycho not at level 7");
            assert_covers_view_without_overlap(&over_tycho, &selection.tiles, &format!("Tycho at {elevation}°"));
        }
    }

    #[test]
    fn typical_views_fit_well_inside_the_budget() {
        let config = LodConfig::default();
        let unbounded = LodConfig { max_visible_tiles: usize::MAX, ..LodConfig::default() };
        for (distance, elevation, camera) in all_views() {
            let tiles = select(&camera, &config, all_ready).tiles;
            assert!(tiles.len() <= MAX_VISIBLE_TILES / 2, "{distance} m at {elevation}°: {} tiles", tiles.len());
            assert_eq!(tiles, select(&camera, &unbounded, all_ready).tiles, "{distance} m at {elevation}°: budget-limited");
        }
    }

    #[test]
    fn tight_budget_is_respected_and_spent_on_the_nearest_terrain() {
        let camera = camera(601_100.0, 30.0);
        let unbounded = select(&camera, &LodConfig { max_visible_tiles: usize::MAX, ..LodConfig::default() }, all_ready);
        for budget in [6, 12, 24, 40] {
            let config = LodConfig { max_visible_tiles: budget, ..LodConfig::default() };
            let tiles = select(&camera, &config, all_ready).tiles;
            assert!(unbounded.tiles.len() > budget);
            assert!(tiles.len() <= budget, "budget {budget}: {} tiles", tiles.len());
            assert_covers_view_without_overlap(&camera, &tiles, &format!("budget {budget}"));

            let nearest = visible_surface_points(&camera)
                .into_iter()
                .min_by(|a, b| a.distance(camera.position()).total_cmp(&b.distance(camera.position())))
                .unwrap();
            let finest = tiles.iter().map(|t| t.level).max().unwrap();
            let near_tile = tiles.iter().find(|t| (0..=t.level).any(|l| tile_at(nearest, l) == **t)).unwrap();
            assert_eq!(near_tile.level, finest, "budget {budget}: nearest terrain is not the most detailed");
        }
    }

    #[test]
    fn zooming_in_refines_the_target_down_to_the_finest_level() {
        let config = LodConfig::default();
        let mut previous = 0;
        for distance in DISTANCES {
            let camera = camera(distance, 90.0);
            let tiles = select(&camera, &config, all_ready).tiles;
            let level = (0..=config.max_level)
                .rev()
                .find(|&l| tiles.contains(&tile_at(camera.target(), l)))
                .expect("target covered");
            assert!(level >= previous, "{distance} m: level {level} < {previous}");
            previous = level;
        }
        assert_eq!(previous, config.max_level);
    }

    #[test]
    fn distant_moon_is_drawn_from_the_face_roots() {
        let tiles = select(&camera(10.0 * MOON_RADIUS, 90.0), &LodConfig::default(), all_ready).tiles;
        assert!(!tiles.is_empty() && tiles.len() <= 6);
        assert!(tiles.iter().all(|t| t.level == 0));
    }

    #[test]
    fn frustum_culls_tiles_outside_the_view() {
        let mut camera = OrbitCamera::new(0.0, 0.0, MOON_RADIUS);
        camera.fov_y = 10f64.to_radians();
        let view = View::new(&camera, HEIGHT_PX, 1.0);
        assert!(view.in_frustum(DVec3::X * MOON_RADIUS - view.position, MOON_RADIUS * 0.01));
        assert!(!view.in_frustum(DVec3::Y * MOON_RADIUS - view.position, MOON_RADIUS * 0.01));
        assert!(!view.in_frustum(DVec3::X * 3.0 * MOON_RADIUS - view.position, MOON_RADIUS * 0.01));
    }

    #[test]
    fn horizon_culls_far_side_but_keeps_peaks_that_can_rise_above_it() {
        // Camera one radius above (0°, 0°): the reference sphere's horizon is 60° away.
        let view = View::new(&OrbitCamera::new(0.0, 0.0, MOON_RADIUS), HEIGHT_PX, ASPECT);
        assert!(!view.above_horizon(&TileBounds::new(TileId::root(Face::Nx))));
        assert!(view.above_horizon(&TileBounds::new(TileId::root(Face::Px))));

        let tile_at_angle = |degrees: f64| TileBounds {
            center_dir: DVec3::new(degrees.to_radians().cos(), degrees.to_radians().sin(), 0.0),
            angular_radius: 0.001,
        };
        // A 10.8 km peak is visible up to ~6.4° beyond the geometric horizon.
        assert!(view.above_horizon(&tile_at_angle(65.0)));
        assert!(!view.above_horizon(&tile_at_angle(68.0)));
    }

    #[test]
    fn children_and_parent_are_inverse() {
        let tile = TileId { face: Face::Ny, level: 3, x: 5, y: 2 };
        for child in tile.children() {
            assert_eq!(child.parent(), Some(tile));
            assert!(child.is_within(tile) && !tile.is_within(child));
        }
        assert_eq!(TileId::root(Face::Ny).parent(), None);
    }

    #[test]
    fn terrain_error_extends_past_the_manifest_levels() {
        let config = LodConfig::default();
        assert_eq!(config.terrain_error(0), V2_TERRAIN_ERROR_M[0]);
        assert_eq!(config.terrain_error(7), V2_TERRAIN_ERROR_M[7]);
        assert_eq!(config.terrain_error(8), V2_TERRAIN_ERROR_M[7] * 0.5);
        assert_eq!(config.terrain_error(9), V2_TERRAIN_ERROR_M[7] * 0.25);
        assert_eq!(config.terrain_error(-1), V2_TERRAIN_ERROR_M[0] * 2.0);
        assert_eq!(config.terrain_error(-2), V2_TERRAIN_ERROR_M[0] * 4.0);
    }

    #[test]
    fn drawn_error_is_that_of_the_level_with_the_same_vertex_spacing() {
        // 64 cells over 256 texels: every 4th texel, the spacing of the full grid two levels up.
        assert_eq!((TILE_SIZE - 1) % TILE_SEGMENTS, 0);
        assert_eq!(GRID_LEVEL_SHIFT, 2);
        let config = LodConfig::default();
        assert_eq!(config.drawn_error(7), V2_TERRAIN_ERROR_M[5]);
        assert_eq!(config.drawn_error(2), V2_TERRAIN_ERROR_M[0]);
        assert_eq!(config.drawn_error(0), V2_TERRAIN_ERROR_M[0] * 4.0);
    }

    #[test]
    fn skirts_reach_down_to_the_coarsest_neighbour() {
        let config = LodConfig::default();
        let px = |level, x, y| TileId { face: Face::Px, level, x, y };
        // px: three level-1 tiles, and the fourth (1, 1) split into level 2.
        let tiles = [px(1, 0, 0), px(1, 1, 0), px(1, 0, 1), px(2, 2, 2), px(2, 3, 2), px(2, 2, 3), px(2, 3, 3)];
        let depths = skirt_depths(&tiles, &config);
        let (d1, d2) = (config.drawn_error(1), config.drawn_error(2));
        assert_eq!(depths[..3], [2.0 * d1; 3]);
        // (2, 2) borders level-1 tiles on its left and bottom; (3, 3) only touches its siblings and other faces,
        // which are not selected here.
        assert_eq!(depths[3], d2 + d1);
        assert_eq!(depths[6], 2.0 * d2);

        // Across faces: nz split into level 1, every other face whole. Each nz child borders two face roots.
        let mut tiles: Vec<_> = [Face::Px, Face::Nx, Face::Py, Face::Ny, Face::Pz].map(TileId::root).to_vec();
        tiles.extend(TileId::root(Face::Nz).children());
        let depths = skirt_depths(&tiles, &config);
        let d0 = config.drawn_error(0);
        assert_eq!(depths[..5], [2.0 * d0; 5]);
        assert_eq!(depths[5..], [d1 + d0; 4]);
    }
}
