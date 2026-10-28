// Builds v1 quadtree tiles + manifest from the downloaded sources (PLAN.md §4.2).
// Usage: node tools/pipeline/build.mjs [--global-max-level N] [--detail-max-level N]
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { cubeToLatLon, DEG, DEM_OFFSET, DEM_SCALE, FACES, linearToSrgb8, quantizeHeight, TILE, tileKey, tileToFace, VERSION } from "../cube.mjs";
import { Raster, readAttachedLabel, readDetachedLabel } from "./pds.mjs";
import { LDEM, RAW_DIR, SLDEM_TYCHO, WAC_643 } from "./sources.mjs";

export const OUT_DIR = "data/tiles";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? Number(process.argv[i + 1]) : fallback;
};
const GLOBAL_MAX_LEVEL = arg("--global-max-level", 3);
const DETAIL_MAX_LEVEL = arg("--detail-max-level", 7);

// SLDEM2015 replaces LDEM_64 inside this window around Tycho, feathered over FEATHER_DEG at its edge.
const DETAIL = { latMin: -46.3, latMax: -40.3, lonMin: -15.5, lonMax: -7.2 };
const FEATHER_DEG = 0.5;
// Hapke-normalized reflectance that maps to white (maria ≈ 0.02–0.035, highlands ≈ 0.04–0.05, Tycho ≈ 0.065);
// polar gaps (|lat| > 70°) fill with POLAR_REFLECTANCE.
const ALBEDO_WHITE = 0.12;
const POLAR_REFLECTANCE = 0.045;

const raw = (f) => join(RAW_DIR, f.file);
console.log("Loading sources...");
const ldem = new Raster(raw(LDEM.image), readDetachedLabel(raw(LDEM.label)), { toValue: (dn) => dn * 0.5 });
const sldem = new Raster(raw(SLDEM_TYCHO.image), readDetachedLabel(raw(SLDEM_TYCHO.label)), {
  crop: { latMin: DETAIL.latMin - 0.1, latMax: DETAIL.latMax + 0.1, lonMin: DETAIL.lonMin + 360 - 0.1, lonMax: DETAIL.lonMax + 360 + 0.1 },
  toValue: (km) => km * 1000,
});
const wac = WAC_643.map((s) => new Raster(raw(s.image), readAttachedLabel(raw(s.image))));

const inDetail = (lat, lon) => lat >= DETAIL.latMin && lat <= DETAIL.latMax && lon >= DETAIL.lonMin && lon <= DETAIL.lonMax;

function heightAt(lat, lon) {
  const base = ldem.sample(lat, lon);
  if (!inDetail(lat, lon)) return base;
  const detail = sldem.sample(lat, lon);
  if (Number.isNaN(detail)) return base;
  const edge = Math.min(lat - DETAIL.latMin, DETAIL.latMax - lat, lon - DETAIL.lonMin, DETAIL.lonMax - lon);
  const w = Math.min(1, edge / FEATHER_DEG);
  return base + w * (detail - base);
}

function reflectanceAt(lat, lon) {
  const tile = wac.find((r) => r.covers(lat, lon));
  const value = tile ? tile.sample(lat, lon) : NaN;
  if (Number.isNaN(value)) return POLAR_REFLECTANCE;
  const w = Math.min(1, Math.max(0, Math.abs(lat) - 69));
  return value + w * (POLAR_REFLECTANCE - value);
}

/** Lat/lon in degrees (east longitude in [-180, 180)) for every pixel of a tile. */
function tileLatLon(axes, level, x, y) {
  const out = new Float64Array(TILE * TILE * 2);
  for (let py = 0; py < TILE; py++) {
    for (let px = 0; px < TILE; px++) {
      const [lat, lon] = cubeToLatLon(axes, tileToFace(level, x, px), tileToFace(level, y, py));
      const i = (py * TILE + px) * 2;
      out[i] = lat / DEG;
      out[i + 1] = lon / DEG;
    }
  }
  return out;
}

function intersectsDetail(axes, level, x, y) {
  const n = 32;
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const [lat, lon] = cubeToLatLon(axes, tileToFace(level, x, (i * (TILE - 1)) / n), tileToFace(level, y, (j * (TILE - 1)) / n));
      if (inDetail(lat / DEG, lon / DEG)) return true;
    }
  }
  return false;
}

/** Largest height error from representing this tile with every other sample, i.e. at the parent's resolution. */
function decimationError(heights) {
  const h = (px, py) => heights[py * TILE + px];
  let max = 0;
  for (let py = 0; py < TILE; py++) {
    for (let px = 0; px < TILE; px++) {
      if (px % 2 === 0 && py % 2 === 0) continue;
      const [x0, x1] = px % 2 ? [px - 1, px + 1] : [px, px];
      const [y0, y1] = py % 2 ? [py - 1, py + 1] : [py, py];
      const approx = (h(x0, y0) + h(x1, y0) + h(x0, y1) + h(x1, y1)) / 4;
      max = Math.max(max, Math.abs(h(px, py) - approx));
    }
  }
  return max;
}

function writeOut(key, bytes) {
  const path = join(OUT_DIR, key);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, bytes);
}

function buildTile(face, level, x, y) {
  const coords = tileLatLon(FACES[face], level, x, y);
  const albedo = Buffer.alloc(TILE * TILE * 4);
  const dem = Buffer.alloc(TILE * TILE * 2);
  const heights = new Float64Array(TILE * TILE);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < TILE * TILE; i++) {
    const lat = coords[i * 2];
    const lon = coords[i * 2 + 1];
    // Store the quantized height so min/max and errors describe exactly what the client receives.
    const q = quantizeHeight(heightAt(lat, lon));
    dem.writeUInt16LE(q, i * 2);
    const h = q * DEM_SCALE + DEM_OFFSET;
    heights[i] = h;
    min = Math.min(min, h);
    max = Math.max(max, h);
    const g = linearToSrgb8(reflectanceAt(lat, lon) / ALBEDO_WHITE);
    albedo[i * 4] = g;
    albedo[i * 4 + 1] = g;
    albedo[i * 4 + 2] = g;
    albedo[i * 4 + 3] = 255;
  }
  writeOut(tileKey("albedo", face, level, x, y), albedo);
  writeOut(tileKey("dem", face, level, x, y), dem);
  return { min, max, error: decimationError(heights) };
}

const levels = [];
const decimation = [];
const started = Date.now();
let pending = Object.keys(FACES).map((face) => ({ face, level: 0, x: 0, y: 0 }));
while (pending.length) {
  const level = pending[0].level;
  const entry = { level, geometricError: 0, tiles: [] };
  decimation[level] = 0;
  const next = [];
  for (const { face, x, y } of pending) {
    const { min, max, error } = buildTile(face, level, x, y);
    entry.tiles.push([face, x, y, min, max]);
    if (level > 0) decimation[level - 1] = Math.max(decimation[level - 1], error);
    const refine = level < GLOBAL_MAX_LEVEL || (level < DETAIL_MAX_LEVEL && intersectsDetail(FACES[face], level, x, y));
    if (refine) {
      for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) next.push({ face, level: level + 1, x: 2 * x + dx, y: 2 * y + dy });
    }
  }
  levels.push(entry);
  console.log(`level ${level}: ${entry.tiles.length} tiles (${((Date.now() - started) / 1000).toFixed(0)} s)`);
  pending = next;
}
// Upper bound on each level's error against the finest data: its own decimation error plus the next level's bound.
// The finest level has no children to measure against; assume errors keep halving.
const last = levels.length - 1;
levels[last].geometricError = decimation[last - 1] / 2;
for (let l = last - 1; l >= 0; l--) levels[l].geometricError = decimation[l] + levels[l + 1].geometricError;
for (const l of levels) l.geometricError = Math.round(l.geometricError * 10) / 10;

const manifest = {
  version: VERSION,
  tileSize: TILE,
  faces: Object.keys(FACES),
  dem: { format: "u16le", scale: DEM_SCALE, offset: DEM_OFFSET, unit: "m", reference: "sphere r=1737400 m" },
  albedo: { format: "rgba8-srgb", source: "LROC WAC Hapke-normalized 643 nm", white: ALBEDO_WHITE, polarFill: POLAR_REFLECTANCE },
  sources: [LDEM.image.url, SLDEM_TYCHO.image.url, ...WAC_643.map((s) => s.image.url)],
  detail: { ...DETAIL, featherDeg: FEATHER_DEG },
  levels,
};
writeOut(`${VERSION}/meta/manifest.json`, JSON.stringify(manifest));
console.log(`Wrote ${levels.reduce((n, l) => n + l.tiles.length, 0)} tiles per layer and manifest to ${OUT_DIR}/${VERSION}/`);
