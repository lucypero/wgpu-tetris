struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) vertex_index: u32,
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4(in.position, 1.0);
    out.tex_coords = in.tex_coords;
    out.vertex_index = in.vertex_index;
    return out;
}

//fragment buffers: 2 for color, 1 for aspect ratio
@group(1) @binding(0)
var<storage, read> border_color_buf: array<vec4<f32>>;
@group(2) @binding(0)
var<storage, read> fill_color_buf: array<vec4<f32>>;
@group(3) @binding(0)
var<storage, read> aspect_ratio_buf: array<f32>;
@group(4) @binding(0)
var<storage, read> border_width_buf: array<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    //get number of bordererd rect using in.vertex_index
    var i = in.vertex_index / u32(6);

    var maxX = 1.0 - border_width_buf[i];
    var minX = border_width_buf[i];
    var maxY = 1.0 - (border_width_buf[i] * aspect_ratio_buf[i]);
    var minY = border_width_buf[i] * aspect_ratio_buf[i];

    if (in.tex_coords.x < maxX && in.tex_coords.x > minX &&
        in.tex_coords.y < maxY && in.tex_coords.y > minY) {
        return fill_color_buf[i];
    } else {
        return border_color_buf[i];
    }
}