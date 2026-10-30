use glam::DVec3;

use crate::geometry::latlon_to_unit;

/// Default sub-solar longitude: the sun stands about 15° above Tycho's horizon in its western sky, low enough for
/// clear relief shading, with the evening terminator about 470 km east of Tycho.
pub const DEFAULT_SUBSOLAR_LON_DEG: f64 = -80.0;

/// Unit direction towards the sun in the body-fixed frame. The sub-solar latitude stays within ±1.6° on the Moon,
/// so it is taken as 0.
pub fn sun_direction(subsolar_lon_deg: f64) -> DVec3 {
    latlon_to_unit(0.0, subsolar_lon_deg.to_radians())
}

/// Wraps a longitude in degrees to `[-180, 180)`.
pub fn wrap_longitude_deg(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{MOON_RADIUS, TYCHO_LAT_DEG, TYCHO_LON_DEG};

    #[test]
    fn default_sun_is_low_over_tycho() {
        let tycho = latlon_to_unit(TYCHO_LAT_DEG.to_radians(), TYCHO_LON_DEG.to_radians());
        let elevation = sun_direction(DEFAULT_SUBSOLAR_LON_DEG).dot(tycho).asin().to_degrees();
        assert!((12.0..18.0).contains(&elevation), "{elevation}");
        // The terminator is where the sun is on the horizon: 90° of longitude from the sub-solar point.
        let terminator_lon = DEFAULT_SUBSOLAR_LON_DEG + 90.0;
        let km = (terminator_lon - TYCHO_LON_DEG) * TYCHO_LAT_DEG.to_radians().cos() * MOON_RADIUS.to_radians() / 1000.0;
        assert!((440.0..500.0).contains(&km), "{km}");
    }

    #[test]
    fn longitudes_wrap() {
        assert_eq!(wrap_longitude_deg(190.0), -170.0);
        assert_eq!(wrap_longitude_deg(-180.0), -180.0);
        assert_eq!(wrap_longitude_deg(180.0), -180.0);
        assert_eq!(wrap_longitude_deg(-80.0), -80.0);
        assert!((sun_direction(90.0) - DVec3::Y).length() < 1e-12);
    }
}
