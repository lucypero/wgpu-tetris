extern crate core;

use std::mem;
use std::mem::MaybeUninit;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use cgmath::prelude::*;
use cgmath::{Matrix4, Vector2, Vector3};
use rand::Rng;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};
use crate::{Game, game};

pub const WINDOW_INNER_WIDTH: u32 = 1000;
pub const WINDOW_INNER_HEIGHT: u32 = 600;
// Fixed number of block instances in the instance renderer
//  In the game, there will always be less than this.
const BLOCK_COUNT: usize = 1024;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vec4([f32; 4]);

impl From<[f32; 4]> for Vec4 {
    fn from(the_vec: [f32; 4]) -> Self {
        Vec4(the_vec)
    }
}

impl From<cgmath::Vector4<f32>> for Vec4 {
    fn from(the_vec: cgmath::Vector4<f32>) -> Self {
        Vec4(the_vec.into())
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Mat4([[f32; 4]; 4]);

impl From<[[f32; 4]; 4]> for Mat4 {
    fn from(the_mat: [[f32; 4]; 4]) -> Self {
        Mat4(the_mat)
    }
}

impl From<cgmath::Matrix4<f32>> for Mat4 {
    fn from(the_mat: cgmath::Matrix4<f32>) -> Self {
        Mat4(the_mat.into())
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

//Camera andy

struct Camera {
    initial_size: Vector2<f32>,
    position: Vector2<f32>,
    zoom_amount: f32,
    bind_group: BindGroupSetThing<CameraUniform>,
}

impl Camera {
    // do not use this
    fn change_zoom(&mut self, new_zoom: f32) {
        // let mut new_mat: Matrix4<f32> = Matrix4::from(self.camera.bind_group.the_data.view_proj.0);
        // new_mat = new_mat * Matrix4::from_scale(CAM_ZOOM_STEP);
        // self.camera.bind_group.the_data.view_proj = Mat4::from(new_mat);

        let view_proj: [[f32; 4]; 4] = {
            let proj = cgmath::ortho(0.0, self.initial_size.x * new_zoom, self.initial_size.y * new_zoom, 0.0, -1.0, 1.0);
            (OPENGL_TO_WGPU_MATRIX * proj).into()
        };

        self.bind_group.the_data.view_proj = view_proj.into();
        self.zoom_amount = new_zoom;
    }

    // do not use this
    fn pan(&mut self, vec: Vector3<f32>) {
        let mut new_mat: Matrix4<f32> = Matrix4::from(self.bind_group.the_data.view_proj.0);
        new_mat = new_mat * Matrix4::from_translation(vec);
        self.bind_group.the_data.view_proj = Mat4::from(new_mat);
    }

    fn update_buffer(&mut self, queue: &wgpu::Queue) {

        // write new matrix in the uniform

        // transformation
        let t_mat = Matrix4::from_translation(Vector3::new(self.position.x, self.position.y, 0.0));
        self.bind_group.the_data.view_proj = Mat4::from(t_mat);

        // zoom
        let view_proj = {
            let new_vec = self.initial_size * self.zoom_amount;
            let proj = cgmath::ortho(0.0, new_vec.x, new_vec.y, 0.0, -1.0, 1.0);
            (OPENGL_TO_WGPU_MATRIX * proj)
        };

        let new_mat: [[f32; 4]; 4] = (view_proj * t_mat).into();
        self.bind_group.the_data.view_proj = new_mat.into();

        // write new data to buffer
        queue.write_buffer(
            &self.bind_group.buffer,
            0,
            bytemuck::cast_slice(&[self.bind_group.the_data]),
        );
    }
}

// we are gonna use an ortho camera
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: Mat4,
}

// TODO: update camera uniform on resize
impl CameraUniform {
    fn new(size: Vector2<f32>) -> Self {
        let view_proj: [[f32; 4]; 4] = {
            let proj = cgmath::ortho(0.0, size.x, size.y, 0.0, -1.0, 1.0);
            (OPENGL_TO_WGPU_MATRIX * proj).into()
        };

        Self {
            view_proj: view_proj.into(),
        }
    }
}

struct ModelUniform {
    model: Vec<Mat4>,
}

impl ModelUniform {
    fn new(size: winit::dpi::PhysicalSize<u32>) -> Self {
        let mut i_x = 0;
        let mut y = 0;
        let mut model_vec: Vec<Mat4> = Vec::with_capacity(BLOCK_COUNT);

        for _ in 0..BLOCK_COUNT {
            let model: [[f32; 4]; 4] = cgmath::Matrix4::from_translation(
                Vector3 {
                    x: -game::BLOCK_SIZE,
                    y: -game::BLOCK_SIZE,
                    z: 0.0,
                }).into();
            model_vec.push(model.into());
        }

        Self {
            model: model_vec
        }
    }
}

struct ColorUniform {
    color: Vec<Vec4>,
}

impl ColorUniform {
    fn new() -> Self {
        let mut color_vec = Vec::with_capacity(BLOCK_COUNT);

        for i in 0..BLOCK_COUNT {
            let color: Vec4 = cgmath::Vector4 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
                w: 1.0,
            }.into();
            color_vec.push(color);
        }

        Self {
            color: color_vec
        }
    }
}

struct BindGroupSetThing<T> {
    the_data: T,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    /*
    (0)-----(1)
     |       |
     |       |
     |       |
    (2)-----(3)
     */
    Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 0.0] },
    Vertex { position: [game::BLOCK_SIZE, 0.0, 0.0], tex_coords: [1.0, 0.0] },
    Vertex { position: [0.0, game::BLOCK_SIZE, 0.0], tex_coords: [0.0, 1.0] },
    Vertex { position: [game::BLOCK_SIZE, game::BLOCK_SIZE, 0.0], tex_coords: [1.0, 1.0] },
];

const INDICES: &[u16] = &[
    1, 0, 2,
    2, 3, 1
];


pub struct Renderer {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    camera: Camera,
    model_uniform_set: BindGroupSetThing<ModelUniform>,
    color_uniform_set: BindGroupSetThing<ColorUniform>,
    diffuse_bind_group: wgpu::BindGroup,
}

impl Renderer {
    pub async fn new(window: &winit::window::Window, game: &Game) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        //println!("{:?}",instance.generate_report());

        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: if cfg!(target_arch = "wasm)") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None,
        ).await.unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let num_indices = INDICES.len() as u32;

        // -- bind descriptors --

        // we need an ortho camera
        let camera_uniform = CameraUniform::new(Vector2::new(size.width as f32, size.height as f32));
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_bind_group_layout"),
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });

        let camera_uniform_set = BindGroupSetThing {
            the_data: camera_uniform,
            buffer: camera_buffer,
            bind_group: camera_bind_group,
        };

        let camera = Camera {
            initial_size: Vector2::new(size.width as f32, size.height as f32),
            position: Vector2::new(0.0, 0.0),
            zoom_amount: 1.0,
            bind_group: camera_uniform_set,
        };

        // we need a model matrix and color

        // model matrix : we need to create the buffer and the bind group
        // this is the instance data for the tetris blocks

        let model_uniform = ModelUniform::new(size);

        let model_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("model matrix buffer"),
                contents: bytemuck::cast_slice(model_uniform.model.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        );

        let model_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("model_bind_group_layout"),
        });

        let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &model_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                }
            ],
            label: Some("model_bind_group"),
        });

        let model_uniform_set = BindGroupSetThing {
            the_data: model_uniform,
            buffer: model_buffer,
            bind_group: model_bind_group,
        };

        // color uniform
        // cute colors for all the bloccs
        let color_uniform = ColorUniform::new();

        let color_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("color buffer"),
                contents: bytemuck::cast_slice(color_uniform.color.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        );

        let color_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("color_bind_group_layout"),
        });

        let color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &color_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color_buffer.as_entire_binding(),
                }
            ],
            label: Some("model_bind_group"),
        });

        let color_uniform_set = BindGroupSetThing {
            the_data: color_uniform,
            buffer: color_buffer,
            bind_group: color_bind_group,
        };

        // the texture
        let diffuse_bytes = include_bytes!("block.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.to_rgba8();

        use image::GenericImageView;
        let dimensions = diffuse_image.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let diffuse_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("diffuse_texture"),
            }
        );

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &diffuse_rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1),
            },
            texture_size,
        );

        // We don't need to configure the texture view much, so let's
        // let wgpu define it.
        let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );

        // -- bind group end --

        // creating pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[ // THE ORDER HERE MATTERS !!!!
                    &camera_bind_group_layout,
                    &model_bind_group_layout,
                    &color_bind_group_layout,
                    &texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = Self::create_pipeline(&device, &config, &shader, &render_pipeline_layout);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            model_uniform_set,
            camera,
            color_uniform_set,
            diffuse_bind_group,
        }
    }

    fn create_pipeline(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, shader: &wgpu::ShaderModule, render_pipeline_layout: &wgpu::PipelineLayout) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn render(&mut self, game: &Game) -> Result<(), wgpu::SurfaceError> {

        // update all the buffers
        {
            //update 0..block len model uniforms
            for i in self.model_uniform_set.the_data.model.iter_mut().take(game.blocks.len()).enumerate() {
                let block = &game.blocks[i.0];

                let new_mat = Matrix4::from_translation(Vector3 {
                    x: game.grid_pos.x + game::BLOCK_SIZE * block.pos.x as f32,
                    y: game.grid_pos.y - game::BLOCK_SIZE * block.pos.y as f32,
                    z: 0.0,
                });
                i.1.0 = new_mat.into();
            }

            self.queue.write_buffer(&self.model_uniform_set.buffer,
                                    0,
                                    bytemuck::cast_slice(self.model_uniform_set.the_data.model.as_slice()));

            //update 0..block len color uniform

            for i in self.color_uniform_set.the_data.color.iter_mut().take(game.blocks.len()).enumerate() {
                let block = &game.blocks[i.0];
                i.1.0 = block.color.into();
            }

            self.queue.write_buffer(&self.color_uniform_set.buffer,
                                    0,
                                    bytemuck::cast_slice(self.color_uniform_set.the_data.color.as_slice()));
        }

        // do the render
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.0,
                                }
                            ),
                            store: true,
                        },
                    })
                ],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline); // 2.
            render_pass.set_bind_group(0, &self.camera.bind_group.bind_group, &[]);
            render_pass.set_bind_group(1, &self.model_uniform_set.bind_group, &[]);
            render_pass.set_bind_group(2, &self.color_uniform_set.bind_group, &[]);
            render_pass.set_bind_group(3, &self.diffuse_bind_group, &[]); // NEW!
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16); // 1.
            render_pass.draw_indexed(0..self.num_indices, 0, 0..game.blocks.len() as _); // 2.
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}