export interface Env {
  MOON_BUCKET: R2Bucket;
  ASSETS: Fetcher;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method !== "GET") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    // Route /assets/* requests to the R2 Bucket
    if (url.pathname.startsWith("/assets/")) {
      const key = url.pathname.replace("/assets/", "");
      if (!key) return new Response("Specify a lunar file path.", { status: 400 });

      try {
        const object = await env.MOON_BUCKET.get(key);
        if (object === null) {
          return new Response(`File ${key} not found.`, { status: 404 });
        }

        const headers = new Headers();
        object.writeHttpMetadata(headers);
        headers.set("Access-Control-Allow-Origin", "*");
        headers.set("Cache-Control", "public, max-age=31536000, immutable");

        return new Response(object.body, { headers });
      } catch (e: any) {
        return new Response(`Internal Error: ${e.message}`, { status: 500 });
      }
    }

    // Fallback: Serve compiled WASM & HTML from the static assets binding
    return env.ASSETS.fetch(request);
  },
};
