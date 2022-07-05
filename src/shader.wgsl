struct ModelUniform {
    model: array<mat4x4<f32>>,
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
    @builtin(instance_index) inst_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * model_mat.model[model.inst_index] * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.2, 0.1, 1.0);
}
