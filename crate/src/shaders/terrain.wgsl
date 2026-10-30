// dem_scale/dem_offset come from the manifest: height in metres = u16 value * dem_scale + dem_offset.
struct Frame { view_proj: mat4x4<f32>, debug_flags: u32, dem_scale: f32, dem_offset: f32, _pad: f32 };
// origin_rel.xyz: the tile's f64 centre minus the camera position, rounded to f32 on the CPU; origin_rel.w: skirt
// depth in metres. center.xyz: the tile centre in the body frame, only precise enough for directions.
struct Object { origin_rel: vec4<f32>, color: vec4<f32>, center: vec4<f32> };

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> object: Object;
@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var albedo_sampler: sampler;
@group(2) @binding(2) var dem_tex: texture_2d<u32>;

const DEBUG_HEIGHT: u32 = 2u;
const DEBUG_NO_SKIRTS: u32 = 8u;
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
    out.uv = (v.grid.xy * LAST_TEXEL + 0.5) / (LAST_TEXEL + 1.0);
    return out;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let d = displaced(v);
    var out: VsOut;
    out.pos = frame.view_proj * vec4(d.rel, 1.0);
    out.uv = d.uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if ((frame.debug_flags & DEBUG_HEIGHT) != 0u) {
        let dims = textureDimensions(dem_tex);
        let texel = min(vec2<u32>(in.uv * vec2<f32>(dims)), dims - 1u);
        let height = f32(textureLoad(dem_tex, texel, 0).r) * frame.dem_scale + frame.dem_offset;
        return vec4(vec3(clamp((height + 5000.0) / 10000.0, 0.0, 1.0)), 1.0);
    }
    return textureSample(albedo_tex, albedo_sampler, in.uv);
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
