// Uploads built tiles and the manifest to R2.
// Usage: node tools/pipeline/upload.mjs --local|--remote [--max-level N]
import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { tileKey, VERSION } from "../cube.mjs";

const DIR = "data/tiles";
const mode = process.argv.includes("--remote") ? "--remote" : process.argv.includes("--local") ? "--local" : null;
if (!mode) {
  console.error("Specify --local or --remote.");
  process.exit(1);
}
const bucket = mode === "--local" ? "lunar-assets-local" : "lunar-assets";
const i = process.argv.indexOf("--max-level");
const maxLevel = i >= 0 ? Number(process.argv[i + 1]) : Infinity;

const manifestKey = `${VERSION}/meta/manifest.json`;
const manifest = JSON.parse(readFileSync(join(DIR, manifestKey), "utf8"));
const entries = manifest.levels
  .filter((l) => l.level <= maxLevel)
  .flatMap(({ level, tiles }) =>
    tiles.flatMap(([face, x, y]) =>
      ["albedo", "dem"].map((layer) => {
        const key = tileKey(layer, face, level, x, y);
        return { key, file: resolve(DIR, key) };
      }),
    ),
  );

const run = (args) => {
  const result = spawnSync("npx", ["wrangler", ...args], { stdio: "inherit", shell: process.platform === "win32" });
  if (result.status !== 0) process.exit(result.status ?? 1);
};

const listFile = join("data", "upload-tiles.json");
writeFileSync(listFile, JSON.stringify(entries));
run(["r2", "bulk", "put", bucket, "--filename", listFile, mode, "--content-type", "application/octet-stream"]);
// Manifest last, so clients never see tiles listed before they exist.
run(["r2", "object", "put", `${bucket}/${manifestKey}`, "--file", join(DIR, manifestKey), mode, "--content-type", "application/json"]);
console.log(`Uploaded ${entries.length} tiles and the manifest to ${bucket} (${mode}).`);
