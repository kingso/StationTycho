// Generates synthetic level-0 test tiles in the final R2 layout (PLAN.md §4.2–4.3) and optionally uploads them.
// Usage: node tools/make-test-tiles.mjs [--upload-local | --upload-remote]
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { cubeToLatLon, DEG, FACES, quantizeHeight, TEST_VERSION, TILE, tileKey } from "./cube.mjs";

const OUT_DIR = ".tiles";
const MOON_RADIUS = 1_737_400;
const TYCHO = { lat: -43.31, lon: -11.36, radiusM: 42_500 };

function angularDistance(lat1, lon1, lat2, lon2) {
  const h = Math.sin((lat2 - lat1) / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin((lon2 - lon1) / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(h)));
}

// Simple complex crater at Tycho: bowl, raised rim, central peak, plus a gentle global undulation.
function heightAt(lat, lon) {
  const r = angularDistance(lat, lon, TYCHO.lat * DEG, TYCHO.lon * DEG) * MOON_RADIUS;
  const R = TYCHO.radiusM;
  let h = r < R ? -3000 + 4000 * (r / R) ** 3 : 1000 * Math.exp(-(r - R) / 15_000);
  h += 1800 * Math.max(0, 1 - (r / 8000) ** 2);
  return h + 500 * Math.sin(2 * lat) * Math.cos(lon);
}

// Graticule every 15°, highlighted equator and prime meridian, and a ring around Tycho.
function albedoAt(lat, lon) {
  const latDeg = lat / DEG;
  const lonDeg = lon / DEG;
  const nearLine = (d) => Math.abs(d - 15 * Math.round(d / 15)) < 0.35;
  const tychoDeg = angularDistance(lat, lon, TYCHO.lat * DEG, TYCHO.lon * DEG) / DEG;
  if (tychoDeg > 1.2 && tychoDeg < 1.6) return [255, 255, 255];
  if (Math.abs(latDeg) < 0.35 || Math.abs(lonDeg) < 0.35) return [230, 160, 60];
  if (nearLine(latDeg) || nearLine(lonDeg)) return [190, 190, 200];
  const shade = 110 + 25 * Math.sin(3 * lon) * Math.cos(2 * lat);
  return [shade, shade, shade * 0.97].map(Math.round);
}

function writeTile(key, bytes) {
  const path = join(OUT_DIR, key);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, bytes);
  return path;
}

const files = [];
for (const [face, axes] of Object.entries(FACES)) {
  const albedo = Buffer.alloc(TILE * TILE * 4);
  const dem = Buffer.alloc(TILE * TILE * 2);
  for (let y = 0; y < TILE; y++) {
    for (let x = 0; x < TILE; x++) {
      const [lat, lon] = cubeToLatLon(axes, -1 + (2 * x) / (TILE - 1), -1 + (2 * y) / (TILE - 1));
      const i = y * TILE + x;
      const [r, g, b] = albedoAt(lat, lon);
      albedo.set([r, g, b, 255], i * 4);
      dem.writeUInt16LE(quantizeHeight(heightAt(lat, lon)), i * 2);
    }
  }
  for (const [key, bytes] of [
    [tileKey("albedo", face, 0, 0, 0, TEST_VERSION), albedo],
    [tileKey("dem", face, 0, 0, 0, TEST_VERSION), dem],
  ]) {
    files.push({ key, path: writeTile(key, bytes) });
  }
}
console.log(`Wrote ${files.length} tiles to ${OUT_DIR}/`);

const mode = process.argv[2];
if (mode === "--upload-local" || mode === "--upload-remote") {
  const [bucket, flag] = mode === "--upload-local" ? ["lunar-assets-local", "--local"] : ["lunar-assets", "--remote"];
  for (const { key, path } of files) {
    const args = ["wrangler", "r2", "object", "put", `${bucket}/${key}`, "--file", path, flag, "--content-type", "application/octet-stream"];
    const result = spawnSync("npx", args, { stdio: "inherit", shell: process.platform === "win32" });
    if (result.status !== 0) process.exit(result.status ?? 1);
  }
  console.log(`Uploaded ${files.length} tiles to ${bucket} (${flag}).`);
}
