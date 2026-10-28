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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Grid on part of the reference sphere, as f32 offsets from an f64 `origin`.
pub struct ChunkMesh {
    pub origin: DVec3,
    /// Interleaved `[offset.xyz, normal.xyz, uv.xy]` per vertex; `uv` addresses the face's level-0 tile.
    pub vertices: Vec<f32>,
    pub triangles: Vec<u32>,
    pub lines: Vec<u32>,
}

/// Texture coordinate of face coordinate `t` in `[-1, 1]`, landing on texel centres of a shared-border tile.
pub fn face_to_tile_uv(t: f64) -> f64 {
    let last = (TILE_SIZE - 1) as f64;
    ((t + 1.0) * 0.5 * last + 0.5) / TILE_SIZE as f64
}

/// Splits `face` into `chunks × chunks` chunks of `segments × segments` cells each.
pub fn face_chunks(face: Face, chunks: u32, segments: u32) -> Vec<ChunkMesh> {
    let side = segments + 1;
    let to_face = |c: u32, s: f64| -1.0 + 2.0 * (c as f64 + s) / chunks as f64;
    let mut out = Vec::with_capacity((chunks * chunks) as usize);
    for cj in 0..chunks {
        for ci in 0..chunks {
            let origin = cube_to_unit(face, to_face(ci, 0.5), to_face(cj, 0.5)) * MOON_RADIUS;
            let mut vertices = Vec::with_capacity((side * side * 8) as usize);
            for j in 0..side {
                for i in 0..side {
                    let u = to_face(ci, i as f64 / segments as f64);
                    let v = to_face(cj, j as f64 / segments as f64);
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
            let mut lines = Vec::with_capacity((4 * segments * side) as usize);
            for a in 0..side {
                for b in 0..segments {
                    lines.extend_from_slice(&[a * side + b, a * side + b + 1]);
                    lines.extend_from_slice(&[b * side + a, (b + 1) * side + a]);
                }
            }
            // Counter-clockwise seen from outside, because u × v points outward.
            let mut triangles = Vec::with_capacity((6 * segments * segments) as usize);
            for j in 0..segments {
                for i in 0..segments {
                    let a = j * side + i;
                    let (b, c, d) = (a + 1, a + side, a + side + 1);
                    triangles.extend_from_slice(&[a, b, d, a, d, c]);
                }
            }
            out.push(ChunkMesh { origin, vertices, triangles, lines });
        }
    }
    out
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

    #[test]
    fn chunk_offsets_reconstruct_sphere_points() {
        let chunks = face_chunks(Face::Py, 2, 4);
        assert_eq!(chunks.len(), 4);
        for chunk in &chunks {
            assert_eq!(chunk.vertices.len(), 25 * 8);
            assert_eq!(chunk.lines.len(), 2 * 2 * 4 * 5);
            assert_eq!(chunk.triangles.len(), 6 * 4 * 4);
            for v in chunk.vertices.chunks(8) {
                let p = chunk.origin + DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
                assert!((p.length() - MOON_RADIUS).abs() < 0.1);
            }
        }
    }

    #[test]
    fn triangles_face_outward() {
        for face in Face::ALL {
            let chunk = &face_chunks(face, 1, 2)[0];
            let pos = |i: u32| {
                let v = &chunk.vertices[i as usize * 8..];
                DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
            };
            for tri in chunk.triangles.chunks(3) {
                let (a, b, c) = (pos(tri[0]), pos(tri[1]), pos(tri[2]));
                let outward = chunk.origin + (a + b + c) / 3.0;
                assert!((b - a).cross(c - a).dot(outward) > 0.0, "{face:?}");
            }
        }
    }

    #[test]
    fn tile_uv_hits_edge_texel_centres() {
        let half_texel = 0.5 / TILE_SIZE as f64;
        assert!((face_to_tile_uv(-1.0) - half_texel).abs() < 1e-12);
        assert!((face_to_tile_uv(1.0) - (1.0 - half_texel)).abs() < 1e-12);
    }
}
