export interface Env {
  MOON_BUCKET: R2Bucket;
  ASSETS: Fetcher;
}

// Tiles are versioned (/v{N}/...) and never overwritten, so they can be cached forever.
const CACHE_CONTROL = "public, max-age=31536000, immutable";

function objectHeaders(object: R2Object): Headers {
  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("ETag", object.httpEtag);
  headers.set("Cache-Control", CACHE_CONTROL);
  headers.set("Accept-Ranges", "bytes");
  return headers;
}

function contentRange(range: R2Range, size: number): string {
  // Local R2 includes `suffix: undefined` on offset ranges, so read values rather than narrowing on keys.
  const { offset: start, length: count, suffix } = range as { offset?: number; length?: number; suffix?: number };
  let offset: number;
  let length: number;
  if (suffix !== undefined) {
    length = Math.min(suffix, size);
    offset = size - length;
  } else {
    offset = start ?? 0;
    length = Math.min(count ?? size - offset, size - offset);
  }
  return `bytes ${offset}-${offset + length - 1}/${size}`;
}

async function fillCache(cacheKey: Request, key: string, env: Env): Promise<void> {
  const object = await env.MOON_BUCKET.get(key);
  if (object !== null) {
    await caches.default.put(cacheKey, new Response(object.body, { headers: objectHeaders(object) }));
  }
}

async function serveAsset(request: Request, env: Env, ctx: ExecutionContext, key: string): Promise<Response> {
  // cache.match evaluates Range (206) and If-None-Match (304) against the stored full response.
  const cached = await caches.default.match(request);
  if (cached) return cached;

  const cacheKey = new Request(request.url);
  const object = await env.MOON_BUCKET.get(key, { onlyIf: request.headers, range: request.headers });
  if (object === null) {
    return new Response(`File ${key} not found.`, { status: 404 });
  }

  const headers = objectHeaders(object);
  if (!("body" in object)) {
    const conditionalGet = request.headers.has("If-None-Match") || request.headers.has("If-Modified-Since");
    return new Response(null, { status: conditionalGet ? 304 : 412, headers });
  }

  if (request.headers.has("Range") && object.range) {
    headers.set("Content-Range", contentRange(object.range, object.size));
    ctx.waitUntil(fillCache(cacheKey, key, env));
    return new Response(object.body, { status: 206, headers });
  }

  const response = new Response(object.body, { headers });
  ctx.waitUntil(caches.default.put(cacheKey, response.clone()));
  return response;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (request.method !== "GET") {
      return new Response("Method Not Allowed", { status: 405, headers: { Allow: "GET" } });
    }

    // Route /assets/* requests to the R2 Bucket
    if (url.pathname.startsWith("/assets/")) {
      const key = url.pathname.slice("/assets/".length);
      if (!key) return new Response("Specify a lunar file path.", { status: 400 });

      try {
        return await serveAsset(request, env, ctx, key);
      } catch (e) {
        console.error(`Asset ${key} failed:`, e);
        return new Response("Internal Error", { status: 500 });
      }
    }

    // Fallback: Serve compiled WASM & HTML from the static assets binding
    return env.ASSETS.fetch(request);
  },
};
