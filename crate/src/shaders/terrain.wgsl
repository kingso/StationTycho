// dem_scale/dem_offset come from the manifest: height in metres = u16 value * dem_scale + dem_offset.
// sun.xyz: unit direction towards the sun in the body frame.
struct Frame { view_proj: mat4x4<f32>, debug_flags: u32, dem_scale: f32, dem_offset: f32, _pad: f32, sun: vec4<f32> };
// origin_rel.xyz: the tile's f64 centre minus the camera position, rounded to f32 on the CPU; origin_rel.w: skirt
// depth in metres. center.xyz: the tile centre in the body frame, only precise enough for directions.
// u_axis/v_axis.xyz: the cube face's u and v axes; .w: metres per DEM texel along them.
struct Object {
    origin_rel: vec4<f32>,
    color: vec4<f32>,
    center: vec4<f32>,
    u_axis: vec4<f32>,
    v_axis: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> object: Object;
@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var albedo_sampler: sampler;
@group(2) @binding(2) var dem_tex: texture_2d<u32>;

const DEBUG_HEIGHT: u32 = 2u;
const DEBUG_NO_SKIRTS: u32 = 8u;
const DEBUG_UNLIT: u32 = 16u;
// Index of the last texel of a 257-texel shared-border tile.
const LAST_TEXEL: f32 = 256.0;

struct VsIn {
    // Offset of the vertex on the reference sphere from the tile centre (per tile).
    @location(0) offset: vec3<f32>,
    // u, v in [0, 1] across the tile, and 1 on the skirt ring (shared by all tiles).
    @location(1) grid: vec3<f32>,
};

struct Displaced {
    rel: vec3<f32>,
    // Reference-sphere normal under the vertex.
    up: vec3<f32>,
    uv: vec2<f32>,
};

// Moves a grid vertex up by its DEM height along the sphere normal. The grid lands exactly on DEM texels, so the
// height needs no filtering, and tiles sharing an edge read the same border texels. Skirt vertices hang below.
fn displaced(v: VsIn) -> Displaced {
    let texel = vec2<u32>(round(v.grid.xy * LAST_TEXEL));
    var height = f32(textureLoad(dem_tex, texel, 0).r) * frame.dem_scale + frame.dem_offset;
    if ((frame.debug_flags & DEBUG_NO_SKIRTS) == 0u) {
        height -= v.grid.z * object.origin_rel.w;
    }
    let normal = normalize(object.center.xyz + v.offset);
    var out: Displaced;
    out.rel = object.origin_rel.xyz + v.offset + normal * height;
    out.up = normal;
    out.uv = (v.grid.xy * LAST_TEXEL + 0.5) / (LAST_TEXEL + 1.0);
    return out;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) up: vec3<f32>,
    // Position relative to the camera, which sits at the origin.
    @location(2) rel: vec3<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let d = displaced(v);
    var out: VsOut;
    out.pos = frame.view_proj * vec4(d.rel, 1.0);
    out.uv = d.uv;
    out.up = d.up;
    out.rel = d.rel;
    return out;
}

fn height_at(texel: vec2<i32>) -> f32 {
    return f32(textureLoad(dem_tex, vec2<u32>(texel), 0).r) * frame.dem_scale + frame.dem_offset;
}

// Height change per texel along u and v at `texel`: central differences, one-sided on the tile border.
fn slope_at(texel: vec2<i32>) -> vec2<f32> {
    let last = vec2<i32>(i32(LAST_TEXEL));
    let t = clamp(texel, vec2<i32>(0), last);
    let lo = max(t - 1, vec2<i32>(0));
    let hi = min(t + 1, last);
    let du = height_at(vec2(hi.x, t.y)) - height_at(vec2(lo.x, t.y));
    let dv = height_at(vec2(t.x, hi.y)) - height_at(vec2(t.x, lo.y));
    return vec2(du, dv) / vec2<f32>(hi - lo);
}

// Terrain normal from the full-resolution DEM: slopes bilinearly blended from the four nearest texels, turned into
// a tilt of the reference-sphere normal `up`.
fn terrain_normal(uv: vec2<f32>, up: vec3<f32>) -> vec3<f32> {
    let f = uv * (LAST_TEXEL + 1.0) - 0.5;
    let t0 = vec2<i32>(floor(f));
    let w = fract(f);
    let s = mix(
        mix(slope_at(t0), slope_at(t0 + vec2(1, 0)), w.x),
        mix(slope_at(t0 + vec2(0, 1)), slope_at(t0 + vec2(1, 1)), w.x),
        w.y,
    );
    let along_u = normalize(object.u_axis.xyz - dot(object.u_axis.xyz, up) * up);
    let along_v = normalize(object.v_axis.xyz - dot(object.v_axis.xyz, up) * up);
    return normalize(up - s.x / object.u_axis.w * along_u - s.y / object.v_axis.w * along_v);
}

// Lunar-Lambert reflectance (McEwen 1991): Lommel-Seeliger blended with Lambert by a phase-dependent weight.
// mu0 = cos(incidence), mu = cos(emission), phase in degrees.
fn lunar_lambert(mu0: f32, mu: f32, phase: f32) -> f32 {
    let l = clamp(1.0 + phase * (-0.019 + phase * (2.42e-4 - phase * 1.46e-6)), 0.0, 1.0);
    return 2.0 * l * mu0 / (mu0 + mu) + (1.0 - l) * mu0;
}

// Sun may be this far below the reference horizon and still reach peaks up to 10.8 km high (about 3.5°).
const SUN_BELOW_HORIZON_LIMIT: f32 = -0.061;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if ((frame.debug_flags & DEBUG_HEIGHT) != 0u) {
        let dims = textureDimensions(dem_tex);
        let texel = min(vec2<u32>(in.uv * vec2<f32>(dims)), dims - 1u);
        let height = f32(textureLoad(dem_tex, texel, 0).r) * frame.dem_scale + frame.dem_offset;
        return vec4(vec3(clamp((height + 5000.0) / 10000.0, 0.0, 1.0)), 1.0);
    }
    let albedo = textureSample(albedo_tex, albedo_sampler, in.uv);
    if ((frame.debug_flags & DEBUG_UNLIT) != 0u) {
        return albedo;
    }
    let up = normalize(in.up);
    let n = terrain_normal(in.uv, up);
    let sun = frame.sun.xyz;
    let view = normalize(-in.rel);
    let mu0 = dot(n, sun);
    // No atmosphere: no light at all without direct sun. Cast shadows come later (PLAN.md §5.3).
    if (mu0 <= 0.0 || dot(up, sun) < SUN_BELOW_HORIZON_LIMIT) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    let mu = max(dot(n, view), 0.0);
    let phase = degrees(acos(clamp(dot(sun, view), -1.0, 1.0)));
    // The albedo mosaic is normalised to incidence 30°, emission 0°, phase 30°; it shows as stored there.
    let reference = lunar_lambert(cos(radians(30.0)), 1.0, 30.0);
    return vec4(albedo.rgb * lunar_lambert(mu0, mu, phase) / reference, 1.0);
}

// Wireframe of the same displaced grid (line-list indices cover the grid, not the skirts).
@vertex
fn vs_wire(v: VsIn) -> @builtin(position) vec4<f32> {
    var pos = frame.view_proj * vec4(displaced(v).rel, 1.0);
    // Pull lines 0.1% towards the camera (reversed-Z) so they don't z-fight the triangles they outline.
    pos.z *= 1.001;
    return pos;
}

@fragment
fn fs_wire() -> @location(0) vec4<f32> {
    return object.color;
}
