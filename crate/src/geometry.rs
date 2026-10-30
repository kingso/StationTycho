use glam::DVec3;

use crate::tiles::TILE_SIZE;

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

    /// Outward normal and the face's `u` and `v` axes, with `u × v = normal`.
    fn axes(self) -> (DVec3, DVec3, DVec3) {
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

/// Texture coordinate of face coordinate `t` in `[-1, 1]`, landing on texel centres of a shared-border tile.
pub fn face_to_tile_uv(t: f64) -> f64 {
    let last = (TILE_SIZE - 1) as f64;
    ((t + 1.0) * 0.5 * last + 0.5) / TILE_SIZE as f64
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

/// Vertices of a `segments × segments` grid over a quadtree tile, as f32 offsets from the tile's f64 centre.
/// Returns the centre and interleaved `[offset.xyz, normal.xyz, uv.xy]`; `uv` addresses the face's level-0 tile.
pub fn tile_vertices(face: Face, level: u8, x: u32, y: u32, segments: u32) -> Option<(DVec3, Vec<f32>)> {
    let (u0, u1, v0, v1) = tile_face_bounds(level, x, y)?;
    if segments == 0 {
        return None;
    }
    let side = segments + 1;
    let origin = cube_to_unit(face, (u0 + u1) * 0.5, (v0 + v1) * 0.5) * MOON_RADIUS;
    let mut vertices = Vec::with_capacity((side * side * 8) as usize);
    for j in 0..side {
        for i in 0..side {
            let u = u0 + (u1 - u0) * i as f64 / segments as f64;
            let v = v0 + (v1 - v0) * j as f64 / segments as f64;
            let s = cube_to_unit(face, u, v);
            let offset = (s * MOON_RADIUS - origin).as_vec3();
            let normal = s.as_vec3();
            vertices.extend_from_slice(&[
                offset.x,
                offset.y,
                offset.z,
                normal.x,
                normal.y,
                normal.z,
                face_to_tile_uv(u) as f32,
                face_to_tile_uv(v) as f32,
            ]);
        }
    }
    Some((origin, vertices))
}

/// Triangle-list and line-list indices shared by every `segments × segments` tile grid from [`tile_vertices`].
pub fn grid_indices(segments: u32) -> (Vec<u32>, Vec<u32>) {
    let side = segments + 1;
    // Counter-clockwise seen from outside, because u × v points outward.
    let mut triangles = Vec::with_capacity((6 * segments * segments) as usize);
    for j in 0..segments {
        for i in 0..segments {
            let a = j * side + i;
            let (b, c, d) = (a + 1, a + side, a + side + 1);
            triangles.extend_from_slice(&[a, b, d, a, d, c]);
        }
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
        vertices.chunks(8).map(|v| origin + DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)).collect()
    }

    #[test]
    fn tile_offsets_reconstruct_sphere_points() {
        let (triangles, lines) = grid_indices(4);
        assert_eq!(triangles.len(), 6 * 4 * 4);
        assert_eq!(lines.len(), 2 * 2 * 4 * 5);
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (origin, vertices) = tile_vertices(Face::Py, 1, x, y, 4).unwrap();
            assert_eq!(vertices.len(), 25 * 8);
            for p in vertex_positions(origin, &vertices) {
                assert!((p.length() - MOON_RADIUS).abs() < 0.1);
            }
        }
    }

    #[test]
    fn triangles_face_outward() {
        let (triangles, _) = grid_indices(2);
        for face in Face::ALL {
            let (origin, vertices) = tile_vertices(face, 0, 0, 0, 2).unwrap();
            let pos = vertex_positions(origin, &vertices);
            for tri in triangles.chunks(3) {
                let (a, b, c) = (pos[tri[0] as usize], pos[tri[1] as usize], pos[tri[2] as usize]);
                assert!((b - a).cross(c - a).dot((a + b + c) / 3.0) > 0.0, "{face:?}");
            }
        }
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

    #[test]
    fn tile_uv_hits_edge_texel_centres() {
        let half_texel = 0.5 / TILE_SIZE as f64;
        assert!((face_to_tile_uv(-1.0) - half_texel).abs() < 1e-12);
        assert!((face_to_tile_uv(1.0) - (1.0 - half_texel)).abs() < 1e-12);
    }
}
