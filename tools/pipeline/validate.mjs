// Checks data/tiles: manifest shape, tile files, complete quadtree children, and identical heights on shared edges.
// Usage: node tools/pipeline/validate.mjs [dir]
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { cubePoint, FACES, TILE, tileKey, tileToFace, VERSION } from "../cube.mjs";

const dir = process.argv[2] ?? "data/tiles";
const errors = [];
const fail = (msg) => {
  if (errors.length < 50) errors.push(msg);
};

const manifest = JSON.parse(readFileSync(join(dir, VERSION, "meta/manifest.json"), "utf8"));
if (manifest.version !== VERSION || manifest.tileSize !== TILE) fail("manifest version/tileSize mismatch");
if (!manifest.dem || !Number.isFinite(manifest.dem.scale) || !Number.isFinite(manifest.dem.offset)) fail("manifest dem quantization missing");
if (!Array.isArray(manifest.levels) || manifest.levels.length === 0) fail("manifest has no levels");

const present = new Set();
manifest.levels.forEach((l, i) => {
  if (l.level !== i) fail(`level ${i} listed as ${l.level}`);
  if (!(l.geometricError >= 0)) fail(`level ${i}: bad geometricError`);
  if (i > 0 && l.geometricError > manifest.levels[i - 1].geometricError) fail(`level ${i}: geometricError increases`);
  for (const [face, x, y, min, max] of l.tiles) {
    if (!(face in FACES) || x < 0 || y < 0 || x >= 2 ** i || y >= 2 ** i) fail(`level ${i}: bad tile ${face}/${x}/${y}`);
    if (!(min <= max)) fail(`${face}/${i}/${x}/${y}: min > max`);
    present.add(`${face}/${i}/${x}/${y}`);
  }
});

// Every tile except level 0 needs its parent and all three siblings, so the client can always fall back.
for (const key of present) {
  const [face, l, x, y] = key.split("/");
  const [level, tx, ty] = [Number(l), Number(x), Number(y)];
  if (level === 0) continue;
  if (!present.has(`${face}/${level - 1}/${tx >> 1}/${ty >> 1}`)) fail(`${key}: missing parent`);
  for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    const sib = `${face}/${level}/${(tx & ~1) + dx}/${(ty & ~1) + dy}`;
    if (!present.has(sib)) fail(`${key}: missing sibling ${sib}`);
  }
}

// Edge pixels at the same cube point (within or across faces) must hold identical heights at each level.
let edgeChecks = 0;
for (const { level, tiles } of manifest.levels) {
  const seen = new Map();
  for (const [face, x, y, min, max] of tiles) {
    for (const [layer, size] of [["dem", TILE * TILE * 2], ["albedo", TILE * TILE * 4]]) {
      const path = join(dir, tileKey(layer, face, level, x, y));
      if (!existsSync(path) || statSync(path).size !== size) fail(`${path}: missing or wrong size`);
    }
    const dem = readFileSync(join(dir, tileKey("dem", face, level, x, y)));
    let tileMin = Infinity;
    let tileMax = -Infinity;
    for (let i = 0; i < TILE * TILE; i++) {
      const h = dem.readUInt16LE(i * 2) * manifest.dem.scale + manifest.dem.offset;
      tileMin = Math.min(tileMin, h);
      tileMax = Math.max(tileMax, h);
    }
    if (tileMin !== min || tileMax !== max) fail(`${face}/${level}/${x}/${y}: manifest min/max ${min}/${max} != data ${tileMin}/${tileMax}`);
    for (let k = 0; k < TILE; k++) {
      for (const [px, py] of [[k, 0], [k, TILE - 1], [0, k], [TILE - 1, k]]) {
        const p = cubePoint(FACES[face], tileToFace(level, x, px), tileToFace(level, y, py));
        const key = p.map((c) => c.toFixed(9)).join(",");
        const q = dem.readUInt16LE((py * TILE + px) * 2);
        const prev = seen.get(key);
        if (prev === undefined) seen.set(key, q);
        else {
          edgeChecks++;
          if (prev !== q) fail(`level ${level} ${face}/${x}/${y} pixel ${px},${py}: ${q} != ${prev}`);
        }
      }
    }
  }
}

const tileCount = manifest.levels.reduce((n, l) => n + l.tiles.length, 0);
if (errors.length) {
  console.error(errors.join("\n"));
  console.error(`FAILED: ${errors.length}${errors.length === 50 ? "+" : ""} problems`);
  process.exit(1);
}
console.log(`OK: ${manifest.levels.length} levels, ${tileCount} tiles, ${edgeChecks} shared edge pixels identical.`);
