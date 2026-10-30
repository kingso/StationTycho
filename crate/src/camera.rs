use std::f64::consts::{FRAC_PI_2, PI, TAU};

use glam::{DMat4, DVec3, DVec4, Mat4};

use crate::geometry::{enu_frame, latlon_to_unit, MOON_RADIUS};

const MIN_DISTANCE: f64 = 0.5;
const MAX_DISTANCE: f64 = 10.0 * MOON_RADIUS;
const MIN_ELEVATION: f64 = 2.0 * PI / 180.0;
const MAX_LATITUDE: f64 = 89.5 * PI / 180.0;
const ROTATE_RADIANS_PER_PX: f64 = 0.005;

/// Orbit camera around a target point on the reference sphere, stored in f64.
pub struct OrbitCamera {
    /// Target latitude in radians.
    pub lat: f64,
    /// Target longitude in radians.
    pub lon: f64,
    /// Distance from the target in metres.
    pub distance: f64,
    /// Viewing heading in radians, 0 = looking north, increasing towards east.
    pub azimuth: f64,
    /// Angle of the camera above the target's local horizon, in radians.
    pub elevation: f64,
    /// Vertical field of view in radians.
    pub fov_y: f64,
}

impl OrbitCamera {
    pub fn new(lat_deg: f64, lon_deg: f64, distance: f64) -> Self {
        Self {
            lat: lat_deg.to_radians(),
            lon: lon_deg.to_radians(),
            distance: distance.clamp(MIN_DISTANCE, MAX_DISTANCE),
            azimuth: 0.0,
            elevation: FRAC_PI_2,
            fov_y: 60f64.to_radians(),
        }
    }

    /// Horizontal view direction (`heading`) and camera right vector, both tangent at the target.
    fn heading_and_right(&self) -> (DVec3, DVec3, DVec3) {
        let (east, north, up) = enu_frame(self.lat, self.lon);
        let (sin_az, cos_az) = self.azimuth.sin_cos();
        let heading = north * cos_az + east * sin_az;
        let right = east * cos_az - north * sin_az;
        (heading, right, up)
    }

    fn forward(&self) -> DVec3 {
        let (heading, _, up) = self.heading_and_right();
        let (sin_el, cos_el) = self.elevation.sin_cos();
        heading * cos_el - up * sin_el
    }

    pub fn target(&self) -> DVec3 {
        latlon_to_unit(self.lat, self.lon) * MOON_RADIUS
    }

    pub fn position(&self) -> DVec3 {
        self.target() - self.forward() * self.distance
    }

    /// Camera-space axes in body-fixed coordinates: right, up, forward.
    pub fn view_basis(&self) -> (DVec3, DVec3, DVec3) {
        let (_, right, _) = self.heading_and_right();
        let forward = self.forward();
        (right, right.cross(forward), forward)
    }

    pub fn altitude(&self) -> f64 {
        self.position().length() - MOON_RADIUS
    }

    /// View-projection with a rotation-only view; geometry must be supplied relative to `position()`.
    pub fn view_proj(&self, aspect: f64) -> Mat4 {
        self.view_proj_f64(aspect).as_mat4()
    }

    fn view_proj_f64(&self, aspect: f64) -> DMat4 {
        let (_, right, _) = self.heading_and_right();
        let f = self.forward();
        let u = right.cross(f);
        let view = DMat4::from_cols(
            DVec4::new(right.x, u.x, -f.x, 0.0),
            DVec4::new(right.y, u.y, -f.y, 0.0),
            DVec4::new(right.z, u.z, -f.z, 0.0),
            DVec4::W,
        );
        let near = (self.altitude() * 0.1).clamp(0.01, 1.0e5);
        let focal = 1.0 / (self.fov_y * 0.5).tan();
        // Right-handed, infinite far plane, reversed-Z: depth 1 at `near`, 0 at infinity.
        let proj = DMat4::from_cols(
            DVec4::new(focal / aspect, 0.0, 0.0, 0.0),
            DVec4::new(0.0, focal, 0.0, 0.0),
            DVec4::new(0.0, 0.0, 0.0, -1.0),
            DVec4::new(0.0, 0.0, near, 0.0),
        );
        proj * view
    }

    /// Moves the target so the ground follows a drag of `dx`/`dy` device pixels.
    pub fn pan(&mut self, dx: f64, dy: f64, viewport_height: f64) {
        let metres_per_px = 2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height.max(1.0);
        let (heading, right, _) = self.heading_and_right();
        let (east, north, _) = enu_frame(self.lat, self.lon);
        let step = right * (-dx * metres_per_px) + heading * (dy * metres_per_px);
        self.lat = (self.lat + step.dot(north) / MOON_RADIUS).clamp(-MAX_LATITUDE, MAX_LATITUDE);
        self.lon += step.dot(east) / (MOON_RADIUS * self.lat.cos());
        self.lon = (self.lon + PI).rem_euclid(TAU) - PI;
    }

    /// Changes heading and tilt from a drag of `dx`/`dy` device pixels.
    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.azimuth = (self.azimuth - dx * ROTATE_RADIANS_PER_PX).rem_euclid(TAU);
        self.elevation = (self.elevation + dy * ROTATE_RADIANS_PER_PX).clamp(MIN_ELEVATION, FRAC_PI_2);
    }

    /// Multiplies the distance to the target by `factor`.
    pub fn zoom(&mut self, factor: f64) {
        if factor.is_finite() && factor > 0.0 {
            self.distance = (self.distance * factor).clamp(MIN_DISTANCE, MAX_DISTANCE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ndc(clip: DVec4) -> DVec3 {
        clip.truncate() / clip.w
    }

    /// The f32 camera-relative path stays within 0.01 px of an all-f64 reference at 1 m altitude.
    #[test]
    fn camera_relative_rendering_is_stable_at_one_metre() {
        const HEIGHT_PX: f64 = 2160.0;
        let mut cam = OrbitCamera::new(-43.31, -11.36, 1.2);
        cam.elevation = 0.9;
        let origin = cam.target();
        let (east, north, up) = enu_frame(cam.lat, cam.lon);
        let point_offset = east * 0.3 - north * 0.2 + up * 0.25;

        for step in 0..200 {
            cam.azimuth = step as f64 * 0.0137;
            let camera_pos = cam.position();

            let reference = ndc(cam.view_proj_f64(16.0 / 9.0) * (origin + point_offset - camera_pos).extend(1.0));
            let rel = (origin - camera_pos).as_vec3() + point_offset.as_vec3();
            let gpu = ndc((cam.view_proj(16.0 / 9.0) * rel.extend(1.0)).as_dvec4());

            let error_px = (gpu.truncate() - reference.truncate()).length() * HEIGHT_PX * 0.5;
            assert!(error_px < 0.01, "step {step}: {error_px} px");
        }
    }

    #[test]
    fn camera_stays_above_surface() {
        let mut cam = OrbitCamera::new(10.0, 20.0, 1.0e6);
        for elevation_deg in [0.0, 5.0, 45.0, 90.0] {
            cam.elevation = f64::to_radians(elevation_deg).max(MIN_ELEVATION);
            assert!(cam.altitude() > 0.0);
        }
    }

    #[test]
    fn controls_respect_limits_and_reverse() {
        let mut cam = OrbitCamera::new(-43.31, -11.36, 5000.0);
        let (lat, lon) = (cam.lat, cam.lon);
        cam.pan(120.0, -80.0, 1080.0);
        assert!(cam.lat != lat && cam.lon != lon);
        cam.pan(-120.0, 80.0, 1080.0);
        assert!((cam.lat - lat).abs() < 1e-6 && (cam.lon - lon).abs() < 1e-6);

        cam.pan(0.0, 1.0e9, 1080.0);
        assert!(cam.lat.abs() <= MAX_LATITUDE);

        cam.rotate(0.0, -1.0e6);
        assert_eq!(cam.elevation, MIN_ELEVATION);
        cam.rotate(0.0, 1.0e6);
        assert_eq!(cam.elevation, FRAC_PI_2);

        cam.zoom(1.0e-12);
        assert_eq!(cam.distance, MIN_DISTANCE);
        cam.zoom(1.0e12);
        assert_eq!(cam.distance, MAX_DISTANCE);
    }
}
