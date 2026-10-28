// Cube-face geometry shared by the tile tools; must match crate/src/geometry.rs and crate/src/tiles.rs.
export const TILE = 257;
// v1 is the synthetic M3 test set; tiles are immutable, so real data gets its own version.
export const VERSION = "v2";
export const TEST_VERSION = "v1";
// Must match DEM_SCALE / DEM_OFFSET in crate/src/shaders/terrain.wgsl.
export const DEM_SCALE = 0.5;
export const DEM_OFFSET = -9200;
export const DEG = Math.PI / 180;

// Same axes as Face::axes: [normal, u, v], with u × v = normal.
export const FACES = {
  px: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  nx: [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
  py: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
  ny: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  pz: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
  nz: [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
};

export function cubePoint([n, a, b], u, v) {
  return [0, 1, 2].map((i) => n[i] + a[i] * u + b[i] * v);
}

/** Latitude and east longitude in radians of a point on the cube surface, via the cube-to-sphere mapping. */
export function cubeToLatLon(axes, u, v) {
  const p = cubePoint(axes, u, v);
  const [x2, y2, z2] = p.map((c) => c * c);
  const x = p[0] * Math.sqrt(1 - y2 / 2 - z2 / 2 + (y2 * z2) / 3);
  const y = p[1] * Math.sqrt(1 - z2 / 2 - x2 / 2 + (z2 * x2) / 3);
  const z = p[2] * Math.sqrt(1 - x2 / 2 - y2 / 2 + (x2 * y2) / 3);
  return [Math.atan2(z, Math.hypot(x, y)), Math.atan2(y, x)];
}

/** Face coordinate of pixel `p` (0..256) in tile column/row `t` at `level`. */
export function tileToFace(level, t, p) {
  return -1 + (2 * (t + p / (TILE - 1))) / 2 ** level;
}

export function tileKey(layer, face, level, x, y, version = VERSION) {
  return `${version}/${layer}/${face}/${level}/${x}/${y}.${layer === "dem" ? "bin" : "rgba"}`;
}

export function quantizeHeight(metres) {
  return Math.min(65535, Math.max(0, Math.round((metres - DEM_OFFSET) / DEM_SCALE)));
}

export function linearToSrgb8(c) {
  const v = Math.min(1, Math.max(0, c));
  return Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055));
}
