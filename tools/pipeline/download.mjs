// Downloads the pipeline's source rasters into data/raw, resuming partial files.
// Usage: node tools/pipeline/download.mjs
import { createWriteStream, existsSync, mkdirSync, renameSync, statSync } from "node:fs";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { ALL_FILES, RAW_DIR } from "./sources.mjs";

async function remoteSize(url) {
  const res = await fetch(url, { method: "HEAD", redirect: "follow" });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return Number(res.headers.get("content-length"));
}

async function download({ url, file }) {
  const path = join(RAW_DIR, file);
  const part = `${path}.part`;
  const total = await remoteSize(url);
  if (existsSync(path) && statSync(path).size === total) {
    console.log(`ok       ${file}`);
    return;
  }
  const have = existsSync(part) ? statSync(part).size : 0;
  const res = await fetch(url, { redirect: "follow", headers: have ? { Range: `bytes=${have}-` } : {} });
  if (!(res.ok || res.status === 206)) throw new Error(`${url}: HTTP ${res.status}`);
  const append = have > 0 && res.status === 206;
  console.log(`${append ? "resume " : "fetch  "}  ${file} (${(total / 1e6).toFixed(1)} MB)`);

  let done = append ? have : 0;
  let lastLog = Date.now();
  const body = Readable.fromWeb(res.body);
  body.on("data", (chunk) => {
    done += chunk.length;
    if (Date.now() - lastLog > 10_000) {
      console.log(`         ${file}: ${((100 * done) / total).toFixed(1)}%`);
      lastLog = Date.now();
    }
  });
  await pipeline(body, createWriteStream(part, { flags: append ? "a" : "w" }));
  if (statSync(part).size !== total) throw new Error(`${file}: expected ${total} bytes, got ${statSync(part).size}`);
  renameSync(part, path);
}

mkdirSync(RAW_DIR, { recursive: true });
for (const f of ALL_FILES) await download(f);
console.log("All sources downloaded.");
