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
