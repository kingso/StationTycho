// Minimal PDS3 simple-cylindrical raster reader with bilinear lat/lon sampling.
import { closeSync, openSync, readFileSync, readSync } from "node:fs";

/** Parses `KEY = value` pairs from a PDS3 label; the first occurrence of each key wins. */
export function parseLabel(text) {
  const out = {};
  for (const m of text.matchAll(/^\s*(\^?[A-Z_]+)\s*=\s*(.+?)\s*$/gm)) {
    if (!(m[1] in out)) out[m[1]] = m[2].replace(/<[^>]*>/g, "").replace(/"/g, "").trim();
  }
  return out;
}

function num(label, key) {
  const v = label[key];
  if (v === undefined) throw new Error(`Label missing ${key}`);
  if (v.startsWith("16#")) {
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, parseInt(v.slice(3, -1), 16));
    return view.getFloat32(0);
  }
  return Number(v);
}

/** Reads the attached label at the start of `path`. */
export function readAttachedLabel(path) {
  const fd = openSync(path, "r");
  const head = Buffer.alloc(65536);
  readSync(fd, head, 0, head.length, 0);
  closeSync(fd);
  const text = head.toString("latin1");
  const end = text.search(/\nEND\s*\n/);
  return parseLabel(end >= 0 ? text.slice(0, end) : text);
}

export function readDetachedLabel(path) {
  return parseLabel(readFileSync(path, "latin1"));
}

const mod = (a, n) => ((a % n) + n) % n;

export class Raster {
  /**
   * Loads `path` described by `label`, optionally cropped to `crop` = {latMin, latMax, lonMin, lonMax} in degrees
   * (longitudes in the label's own range). `toValue` converts a raw sample to the output unit.
   */
  constructor(path, label, { crop, toValue = (x) => x } = {}) {
    const lines = num(label, "LINES");
    const samples = num(label, "LINE_SAMPLES");
    const bits = num(label, "SAMPLE_BITS");
    const type = label.SAMPLE_TYPE;
    const Typed = type === "PC_REAL" && bits === 32 ? Float32Array : type === "LSB_INTEGER" && bits === 16 ? Int16Array : null;
    if (!Typed) throw new Error(`${path}: unsupported ${type}/${bits}`);
    const bytes = bits / 8;
    const pointer = label["^IMAGE"];
    const dataOffset = /^\d+$/.test(pointer) ? (Number(pointer) - 1) * num(label, "RECORD_BYTES") : 0;

    this.res = num(label, "MAP_RESOLUTION");
    this.centerLon = num(label, "CENTER_LONGITUDE");
    this.west = num(label, "WESTERNMOST_LONGITUDE");
    this.east = num(label, "EASTERNMOST_LONGITUDE");
    this.latMin = num(label, "MINIMUM_LATITUDE");
    this.latMax = num(label, "MAXIMUM_LATITUDE");
    this.global = this.east - this.west >= 360 && !crop;
    this.nullBelow = "CORE_NULL" in label ? -1e30 : -Infinity;
    this.toValue = toValue;

    let lineOffset = num(label, "LINE_PROJECTION_OFFSET");
    let sampleOffset = num(label, "SAMPLE_PROJECTION_OFFSET");
    let [r0, r1, c0, c1] = [0, lines - 1, 0, samples - 1];
    if (crop) {
      r0 = Math.max(0, Math.floor(lineOffset - crop.latMax * this.res));
      r1 = Math.min(lines - 1, Math.ceil(lineOffset - crop.latMin * this.res));
      c0 = Math.max(0, Math.floor(sampleOffset + (crop.lonMin - this.centerLon) * this.res));
      c1 = Math.min(samples - 1, Math.ceil(sampleOffset + (crop.lonMax - this.centerLon) * this.res));
      [this.latMin, this.latMax, this.west, this.east] = [crop.latMin, crop.latMax, crop.lonMin, crop.lonMax];
    }
    this.lines = r1 - r0 + 1;
    this.samples = c1 - c0 + 1;
    this.lineOffset = lineOffset - r0;
    this.sampleOffset = sampleOffset - c0;

    const buffer = new ArrayBuffer(this.lines * this.samples * bytes);
    const view = new Uint8Array(buffer);
    const fd = openSync(path, "r");
    const rowBytes = this.samples * bytes;
    if (c0 === 0 && this.samples === samples) {
      const CHUNK = 1 << 30;
      for (let done = 0; done < view.length; ) {
        const n = readSync(fd, view, done, Math.min(CHUNK, view.length - done), dataOffset + r0 * samples * bytes + done);
        if (n <= 0) throw new Error(`${path}: short read`);
        done += n;
      }
    } else {
      for (let r = 0; r < this.lines; r++) {
        readSync(fd, view, r * rowBytes, rowBytes, dataOffset + ((r0 + r) * samples + c0) * bytes);
      }
    }
    closeSync(fd);
    this.data = new Typed(buffer);
  }

  covers(lat, lon) {
    const l = this.west + mod(lon - this.west, 360);
    return lat >= this.latMin && lat <= this.latMax && l >= this.west && l <= this.east;
  }

  /** Bilinear sample at planetocentric `lat`, east `lon` in degrees; NaN outside coverage or on nulls. */
  sample(lat, lon) {
    if (!this.global && !this.covers(lat, lon)) return NaN;
    const l = this.west + mod(lon - this.west, 360);
    const i = Math.min(Math.max(this.lineOffset - lat * this.res, 0), this.lines - 1);
    let j = this.sampleOffset + (l - this.centerLon) * this.res;
    if (this.global) j = mod(j, this.samples);
    else j = Math.min(Math.max(j, 0), this.samples - 1);
    const i0 = Math.floor(i);
    const j0 = Math.floor(j);
    const i1 = Math.min(i0 + 1, this.lines - 1);
    const j1 = this.global ? (j0 + 1) % this.samples : Math.min(j0 + 1, this.samples - 1);
    const fi = i - i0;
    const fj = j - j0;
    const d = this.data;
    const s = this.samples;
    const a = d[i0 * s + j0];
    const b = d[i0 * s + j1];
    const c = d[i1 * s + j0];
    const e = d[i1 * s + j1];
    if (a <= this.nullBelow || b <= this.nullBelow || c <= this.nullBelow || e <= this.nullBelow) return NaN;
    return this.toValue((a * (1 - fj) + b * fj) * (1 - fi) + (c * (1 - fj) + e * fj) * fi);
  }
}
