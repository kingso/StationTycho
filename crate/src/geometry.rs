use glam::DVec3;

/// IAU reference sphere radius of the Moon, in metres.
pub const MOON_RADIUS: f64 = 1_737_400.0;

/// Unit vector in the Moon ME body-fixed frame for `lat`/`lon` in radians.
pub fn latlon_to_unit(lat: f64, lon: f64) -> DVec3 {
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
}

/// Local east, north and up unit vectors at `lat`/`lon` in radians.
pub fn enu_frame(lat: f64, lon: f64) -> (DVec3, DVec3, DVec3) {
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let north = DVec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
    let up = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
    (east, north, up)
}

/// Cube face, named by its outward axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Face {
    Px,
    Nx,
    Py,
    Ny,
    Pz,
    Nz,
}

impl Face {
    pub const ALL: [Face; 6] = [Face::Px, Face::Nx, Face::Py, Face::Ny, Face::Pz, Face::Nz];

    /// Name used in R2 tile keys.
    pub fn key(self) -> &'static str {
        match self {
            Face::Px => "px",
            Face::Nx => "nx",
            Face::Py => "py",
            Face::Ny => "ny",
            Face::Pz => "pz",
            Face::Nz => "nz",
        }
    }

    /// Inverse of [`Face::key`].
    pub fn from_key(key: &str) -> Option<Self> {
        Face::ALL.into_iter().find(|face| face.key() == key)
    }

    /// Outward normal and the face's `u` and `v` axes, with `u × v = normal`.
    pub fn axes(self) -> (DVec3, DVec3, DVec3) {
        match self {
            Face::Px => (DVec3::X, DVec3::Y, DVec3::Z),
            Face::Nx => (DVec3::NEG_X, DVec3::NEG_Y, DVec3::Z),
            Face::Py => (DVec3::Y, DVec3::NEG_X, DVec3::Z),
            Face::Ny => (DVec3::NEG_Y, DVec3::X, DVec3::Z),
            Face::Pz => (DVec3::Z, DVec3::Y, DVec3::NEG_X),
            Face::Nz => (DVec3::NEG_Z, DVec3::Y, DVec3::X),
        }
    }
}

/// Maps a point on the surface of the cube `[-1, 1]^3` to the unit sphere.
pub fn cube_to_sphere(p: DVec3) -> DVec3 {
    let (x2, y2, z2) = (p.x * p.x, p.y * p.y, p.z * p.z);
    DVec3::new(
        p.x * (1.0 - y2 / 2.0 - z2 / 2.0 + y2 * z2 / 3.0).sqrt(),
        p.y * (1.0 - z2 / 2.0 - x2 / 2.0 + z2 * x2 / 3.0).sqrt(),
        p.z * (1.0 - x2 / 2.0 - y2 / 2.0 + x2 * y2 / 3.0).sqrt(),
    )
}

/// Unit-sphere direction for face coordinates `u`, `v` in `[-1, 1]`.
pub fn cube_to_unit(face: Face, u: f64, v: f64) -> DVec3 {
    let (normal, u_axis, v_axis) = face.axes();
    cube_to_sphere(normal + u_axis * u + v_axis * v)
}

/// Inverse of [`cube_to_unit`] for any non-zero direction.
#[allow(dead_code)] // Needed by tile streaming (M3/M4) and picking (M5).
pub fn unit_to_cube(dir: DVec3) -> (Face, f64, f64) {
    let s = dir.normalize();
    let a = s.abs();
    let face = if a.x >= a.y && a.x >= a.z {
        if s.x >= 0.0 { Face::Px } else { Face::Nx }
    } else if a.y >= a.z {
        if s.y >= 0.0 { Face::Py } else { Face::Ny }
    } else if s.z >= 0.0 {
        Face::Pz
    } else {
        Face::Nz
    };
    let (_, u_axis, v_axis) = face.axes();
    let (su, sv) = (s.dot(u_axis), s.dot(v_axis));
    // Closed-form inverse of su = u·sqrt(1/2 - v²/6), sv = v·sqrt(1/2 - u²/6), in a cancellation-free form.
    let (su2, sv2) = (su * su, sv * sv);
    let d = 2.0 * (su2 - sv2);
    let u2 = 12.0 * su2 / ((3.0 + d) + ((3.0 + d).powi(2) - 24.0 * su2).max(0.0).sqrt());
    let v2 = 12.0 * sv2 / ((3.0 - d) + ((3.0 - d).powi(2) - 24.0 * sv2).max(0.0).sqrt());
    (face, su.signum() * u2.min(1.0).sqrt(), sv.signum() * v2.min(1.0).sqrt())
}

/// Face coordinates for `lat`/`lon` in radians.
#[allow(dead_code)]
pub fn latlon_to_cube(lat: f64, lon: f64) -> (Face, f64, f64) {
    unit_to_cube(latlon_to_unit(lat, lon))
}

/// Latitude and longitude in radians for face coordinates `u`, `v`.
#[allow(dead_code)]
pub fn cube_to_latlon(face: Face, u: f64, v: f64) -> (f64, f64) {
    let s = cube_to_unit(face, u, v);
    (s.z.atan2(s.x.hypot(s.y)), s.y.atan2(s.x))
}

/// Face-coordinate bounds `(u0, u1, v0, v1)` of quadtree tile `(x, y)` at `level`.
pub fn tile_face_bounds(level: u8, x: u32, y: u32) -> Option<(f64, f64, f64, f64)> {
    let side = 1u32.checked_shl(u32::from(level))?;
    if x >= side || y >= side {
        return None;
    }
    let step = 2.0 / side as f64;
    let (u0, v0) = (-1.0 + x as f64 * step, -1.0 + y as f64 * step);
    Some((u0, u0 + step, v0, v0 + step))
}

/// Ground distance in metres per DEM texel along the tile's `u` and `v` directions, measured along its centre
/// lines on the reference sphere (a 257-texel tile spans 256 texel steps).
pub fn texel_spacing(face: Face, level: u8, x: u32, y: u32) -> Option<(f64, f64)> {
    let (u0, u1, v0, v1) = tile_face_bounds(level, x, y)?;
    let (um, vm) = ((u0 + u1) * 0.5, (v0 + v1) * 0.5);
    let arc = |from: (f64, f64), to: (f64, f64)| {
        const STEPS: usize = 16;
        let point = |k: usize| {
            let t = k as f64 / STEPS as f64;
            cube_to_unit(face, from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t)
        };
        (0..STEPS).map(|k| point(k).angle_between(point(k + 1))).sum::<f64>() * MOON_RADIUS
    };
    let last_texel = f64::from(crate::tiles::TILE_SIZE - 1);
    Some((arc((u0, vm), (u1, vm)) / last_texel, arc((um, v0), (um, v1)) / last_texel))
}

/// Number of vertices in a tile grid: `(segments + 1)²` grid vertices, then a skirt ring of `4 · segments`.
pub fn grid_vertex_count(segments: u32) -> u32 {
    (segments + 1) * (segments + 1) + 4 * segments
}

/// Grid `(i, j)` of each skirt-ring vertex: once around the tile edge, counter-clockwise seen from outside.
fn skirt_ring(segments: u32) -> impl Iterator<Item = (u32, u32)> {
    let s = segments;
    (0..s).map(|i| (i, 0)).chain((0..s).map(move |j| (s, j))).chain((0..s).map(move |i| (s - i, s))).chain((0..s).map(move |j| (0, s - j)))
}

/// Vertices of a `segments × segments` grid over a quadtree tile, as f32 offsets `[x, y, z]` from the tile's f64
/// centre on the reference sphere, followed by the skirt ring (duplicates of the edge vertices). Returns the
/// centre and the offsets; the shader adds heights along the sphere normal.
pub fn tile_vertices(face: Face, level: u8, x: u32, y: u32, segments: u32) -> Option<(DVec3, Vec<f32>)> {
    let (u0, u1, v0, v1) = tile_face_bounds(level, x, y)?;
    if segments == 0 {
        return None;
    }
    let side = segments + 1;
    let origin = cube_to_unit(face, (u0 + u1) * 0.5, (v0 + v1) * 0.5) * MOON_RADIUS;
    let offset = |i: u32, j: u32| {
        let (s_i, s_j) = (i as f64 / segments as f64, j as f64 / segments as f64);
        (cube_to_unit(face, u0 + (u1 - u0) * s_i, v0 + (v1 - v0) * s_j) * MOON_RADIUS - origin).as_vec3()
    };
    let mut vertices = Vec::with_capacity(grid_vertex_count(segments) as usize * 3);
    let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
    for (i, j) in grid.chain(skirt_ring(segments)) {
        vertices.extend_from_slice(&offset(i, j).to_array());
    }
    Some((origin, vertices))
}

/// Per-vertex `[u, v, skirt]` shared by every tile grid: `u`, `v` in `[0, 1]` across the tile (`(u, v) · 256` is
/// the vertex's DEM texel when `segments` divides 256), and `skirt` is 1 on the skirt ring.
pub fn grid_attributes(segments: u32) -> Vec<f32> {
    let side = segments + 1;
    let t = |n: u32| n as f32 / segments as f32;
    let grid = (0..side).flat_map(|j| (0..side).map(move |i| [t(i), t(j), 0.0]));
    let skirt = skirt_ring(segments).map(|(i, j)| [t(i), t(j), 1.0]);
    grid.chain(skirt).flatten().collect()
}

/// Triangle-list indices (grid, then skirt walls) and line-list indices (grid only) shared by every tile.
pub fn grid_indices(segments: u32) -> (Vec<u32>, Vec<u32>) {
    let side = segments + 1;
    // Counter-clockwise seen from outside, because u × v points outward.
    let mut triangles = Vec::with_capacity((6 * segments * segments + 24 * segments) as usize);
    for j in 0..segments {
        for i in 0..segments {
            let a = j * side + i;
            let (b, c, d) = (a + 1, a + side, a + side + 1);
            triangles.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    // Each wall hangs from edge vertices a -> b down to their skirt copies c -> d, facing away from the tile.
    let ring: Vec<_> = skirt_ring(segments).map(|(i, j)| j * side + i).collect();
    let first_skirt = side * side;
    for k in 0..ring.len() {
        let next = (k + 1) % ring.len();
        let (a, b) = (ring[k], ring[next]);
        let (c, d) = (first_skirt + k as u32, first_skirt + next as u32);
        triangles.extend_from_slice(&[a, c, b, b, c, d]);
    }
    let mut lines = Vec::with_capacity((4 * segments * side) as usize);
    for a in 0..side {
        for b in 0..segments {
            lines.extend_from_slice(&[a * side + b, a * side + b + 1]);
            lines.extend_from_slice(&[b * side + a, (b + 1) * side + a]);
        }
    }
    (triangles, lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_to_sphere_lands_on_unit_sphere() {
        for face in Face::ALL {
            for i in 0..=20 {
                for j in 0..=20 {
                    let (u, v) = (-1.0 + i as f64 * 0.1, -1.0 + j as f64 * 0.1);
                    let len = cube_to_unit(face, u, v).length();
                    assert!((len - 1.0).abs() < 1e-12, "{face:?} {u} {v}: {len}");
                }
            }
        }
    }

    #[test]
    fn latlon_round_trips_through_cube() {
        for lat_step in -179..=179 {
            for lon_step in -359..=360 {
                let (lat, lon) = ((lat_step as f64 * 0.5).to_radians(), (lon_step as f64 * 0.5).to_radians());
                let (face, u, v) = latlon_to_cube(lat, lon);
                assert!(u.abs() <= 1.0 && v.abs() <= 1.0);
                let (lat2, lon2) = cube_to_latlon(face, u, v);
                let dlon = (lon2 - lon + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
                assert!((lat2 - lat).abs() < 1e-12 && dlon.abs() < 1e-12, "lat {lat} lon {lon}");
            }
        }
    }

    #[test]
    fn cube_round_trips_through_latlon() {
        for face in Face::ALL {
            for i in 0..=40 {
                for j in 0..=40 {
                    let (u, v) = (-0.999 + i as f64 * 0.04995, -0.999 + j as f64 * 0.04995);
                    let (lat, lon) = cube_to_latlon(face, u, v);
                    let (face2, u2, v2) = latlon_to_cube(lat, lon);
                    assert_eq!(face, face2);
                    assert!((u2 - u).abs() < 1e-12 && (v2 - v).abs() < 1e-12, "{face:?} {u} {v}");
                }
            }
        }
    }

    #[test]
    fn plan_conventions_hold() {
        assert_eq!(latlon_to_cube(0.0, 0.0), (Face::Px, 0.0, 0.0));
        assert_eq!(latlon_to_cube(std::f64::consts::FRAC_PI_2, 0.0).0, Face::Pz);
    }

    fn vertex_positions(origin: DVec3, vertices: &[f32]) -> Vec<DVec3> {
        vertices.chunks(3).map(|v| origin + DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)).collect()
    }

    #[test]
    fn tile_offsets_reconstruct_sphere_points() {
        let (triangles, lines) = grid_indices(4);
        assert_eq!(grid_vertex_count(4), 25 + 16);
        assert_eq!(triangles.len(), 6 * 4 * 4 + 6 * 16);
        assert_eq!(lines.len(), 2 * 2 * 4 * 5);
        assert!(triangles.iter().chain(&lines).all(|&i| i < grid_vertex_count(4)));
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (origin, vertices) = tile_vertices(Face::Py, 1, x, y, 4).unwrap();
            assert_eq!(vertices.len(), 41 * 3);
            for p in vertex_positions(origin, &vertices) {
                assert!((p.length() - MOON_RADIUS).abs() < 0.1);
            }
        }
    }

    #[test]
    fn skirt_ring_duplicates_the_edge_and_is_flagged() {
        const S: u32 = 8;
        let side = S as usize + 1;
        let (origin, vertices) = tile_vertices(Face::Nz, 3, 2, 5, S).unwrap();
        let attributes = grid_attributes(S);
        assert_eq!(attributes.len(), grid_vertex_count(S) as usize * 3);
        let positions = vertex_positions(origin, &vertices);
        let attrs: Vec<_> = attributes.chunks(3).collect();
        for (k, (i, j)) in skirt_ring(S).enumerate() {
            let (skirt, edge) = (side * side + k, j as usize * side + i as usize);
            assert!(i == 0 || i == S || j == 0 || j == S);
            assert_eq!(positions[skirt], positions[edge]);
            assert_eq!((attrs[skirt][0], attrs[skirt][1], attrs[skirt][2]), (attrs[edge][0], attrs[edge][1], 1.0));
        }
        assert!(attrs[..side * side].iter().all(|a| a[2] == 0.0));
        // Grid vertices land on every 32nd texel of the 257-texel tile, the edges on texels 0 and 256.
        assert_eq!((attrs[0][0] * 256.0, attrs[side - 1][0] * 256.0, attrs[1][0] * 256.0), (0.0, 256.0, 32.0));
    }

    #[test]
    fn triangles_face_outward_and_skirts_face_away_from_the_tile() {
        const S: u32 = 2;
        let (triangles, _) = grid_indices(S);
        let grid_triangles = (6 * S * S) as usize;
        for face in Face::ALL {
            let (origin, vertices) = tile_vertices(face, 1, 1, 0, S).unwrap();
            let mut pos = vertex_positions(origin, &vertices);
            // Lower the skirt ring as the shader does.
            for p in pos.iter_mut().skip(((S + 1) * (S + 1)) as usize) {
                *p -= p.normalize() * 1000.0;
            }
            for (n, tri) in triangles.chunks(3).enumerate() {
                let (a, b, c) = (pos[tri[0] as usize], pos[tri[1] as usize], pos[tri[2] as usize]);
                let normal = (b - a).cross(c - a);
                let centroid = (a + b + c) / 3.0;
                if n * 3 < grid_triangles {
                    assert!(normal.dot(centroid) > 0.0, "{face:?} grid triangle {n}");
                } else {
                    let away = centroid - origin - origin.normalize() * (centroid - origin).dot(origin.normalize());
                    assert!(normal.dot(away) > 0.0, "{face:?} skirt triangle {n}");
                }
            }
        }
    }

    #[test]
    fn texel_spacing_matches_tile_size() {
        // A face spans a quarter of a great circle; its centre lines are exactly that long.
        let quarter = std::f64::consts::FRAC_PI_2 * MOON_RADIUS / 256.0;
        let (a, b) = texel_spacing(Face::Pz, 0, 0, 0).unwrap();
        assert!((a - quarter).abs() < 1.0 && (b - quarter).abs() < 1.0, "{a} {b}");
        // Level 7 tiles near Tycho: about 2729 km / 128 / 256 ≈ 83 m, within the mapping's ±25 % stretch.
        let (a, b) = texel_spacing(Face::Px, 7, 42, 0).unwrap();
        assert!((60.0..110.0).contains(&a) && (60.0..110.0).contains(&b), "{a} {b}");
        // Halving per level.
        let (parent, child) = (texel_spacing(Face::Ny, 3, 2, 5).unwrap(), texel_spacing(Face::Ny, 4, 5, 11).unwrap());
        assert!((child.0 / parent.0 - 0.5).abs() < 0.05 && (child.1 / parent.1 - 0.5).abs() < 0.05);
        assert!(texel_spacing(Face::Px, 1, 2, 0).is_none());
    }

    #[test]
    fn neighbouring_and_nested_tiles_share_vertices() {
        const SEGMENTS: u32 = 8;
        let side = SEGMENTS as usize + 1;
        let grid = |level, x, y| {
            let (origin, vertices) = tile_vertices(Face::Px, level, x, y, SEGMENTS).unwrap();
            vertex_positions(origin, &vertices)
        };
        // Right edge of (2, 5) against left edge of (3, 5) at level 3.
        let (left, right) = (grid(3, 2, 5), grid(3, 3, 5));
        for j in 0..side {
            assert!(left[j * side + SEGMENTS as usize].distance(right[j * side]) < 0.05);
        }
        // Child (4, 10) at level 4 starts at parent (2, 5)'s corner and matches its even vertices.
        let (parent, child) = (grid(3, 2, 5), grid(4, 4, 10));
        for j in (0..side).step_by(2) {
            for i in (0..side).step_by(2) {
                assert!(child[j * side + i].distance(parent[(j / 2) * side + i / 2]) < 0.05);
            }
        }
        assert!(tile_vertices(Face::Px, 2, 4, 0, 8).is_none());
        assert!(tile_vertices(Face::Px, 2, 0, 0, 0).is_none());
    }
}
