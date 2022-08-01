struct ModelUniform {
    model: array< mat4x4<f32> >,
};
struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;
@group(1) @binding(0)
var<storage, read> model_mat: ModelUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @builtin(instance_index) inst_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) inst_index: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * model_mat.model[model.inst_index] * vec4<f32>(model.position, 1.0);
    out.tex_coords = model.tex_coords;
    out.inst_index = model.inst_index;
    return out;
}

struct ColorUniform {
    color: array< vec4<f32> >,
};
@group(2) @binding(0)
var<storage, read> color_buf: ColorUniform;

@group(3) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(3) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var t = textureSample(t_diffuse, s_diffuse, in.tex_coords).x;
    var the_tint = color_buf.color[in.inst_index].xyz;
//    var the_tint = vec3(1.0,1.0,1.0);

    var res = mix(the_tint, vec3(t), abs(t - 0.5) * 2.0);
    return vec4(res, color_buf.color[in.inst_index].w);
}
