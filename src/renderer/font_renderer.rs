use super::*;
use libs::bytemuck;
use libs::cgmath::Vector2;
use libs::freetype_sys as ft;
use libs::wgpu;
use libs::wgpu::{util::DeviceExt, BindGroup, Device, Queue, RenderPass, Sampler};
use std::collections::HashMap;

pub const SPACE_WIDTH: f32 = 5.0;

pub struct TextRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub sampler: wgpu::Sampler,
    // @TODO(lucypero): do index buffer for text vertices later
    pub char_map: HashMap<char, Character>,
}

impl TextRenderer {
    pub fn get_string_extents(&self, the_str: &str) -> Vector2<f32> {
        //loop over the character and add up the pixel advance

        let mut extent_x:f32 = 0.0;
        let mut extent_y:f32 = 0.0;

        for c in the_str.chars() {
            if c == ' ' {
                extent_x += SPACE_WIDTH;
                continue;
            }

            let char_info = self.char_map.get(&c).unwrap();
            extent_x += (char_info.advance >> 6) as f32;

            if extent_y < char_info.size.y as f32 {
                extent_y = char_info.size.y as f32;
            }
        }

        Vector2::new(extent_x, extent_y)
    }
}

pub struct StringOnScreen {
    pub text: String,
    pub pos: Vector2<f32>,
}

#[derive(Debug)]
pub struct Character {
    bind_group: BindGroup,
    size: Vector2<u32>,
    bearing: Vector2<i32>,
    advance: u32,
}

pub fn init_text_renderer(
    device: &Device,
    queue: &Queue,
    surface_config: &wgpu::SurfaceConfiguration,
) -> TextRenderer {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let char_map =
        unsafe { render_font(&device, &queue, &sampler, "assets/Roboto.ttf".into(), 22) };

    // make the text vertex buffer
    let zeroed_verts = vec![0_u8; 6 * std::mem::size_of::<Vertex>() * MAX_CHARACTERS_ON_SCREEN];

    //vertex buffer for text
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("glyph vertex buffer"),
        contents: &zeroed_verts,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    // text pipeline
    let text_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/text.wgsl"));

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
            &get_camera_texture_bind_group_layout(&device),
            &get_normal_texture_bind_group_layout(&device),
        ],
        push_constant_ranges: &[],
    });

    let pipeline = create_pipeline(
        &device,
        &surface_config,
        &text_shader,
        &pipeline_layout,
        "text pipeline",
    );

    TextRenderer {
        pipeline,
        vertex_buffer,
        sampler,
        char_map,
    }
}

impl TextRenderer {
    pub fn render<'a>(&'a self, render_pass: &mut RenderPass<'a>, camera_bg: &'a BindGroup, the_strings: &Vec<StringOnScreen>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bg, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        let mut i = 0;

        for label in the_strings.iter() {
            for c in label.text.chars() {
                if c == ' ' {
                    continue;
                }

                let char_info = self.char_map.get(&c).unwrap();
                render_pass.set_bind_group(1, &char_info.bind_group, &[]);
                render_pass.draw(i as u32 * 6..i as u32 * 6 + 6, 0..1);

                i += 1;
            }
        }
    }

    /// generates all the vertices for all the rendered text and re-writes the vertex buffer
    pub fn update_vertices(&self, queue: &Queue, the_strings: &Vec<StringOnScreen>) {
        //updating text vertex buffer

        // calculating how many glyphs to draw
        let characters_to_draw_len = the_strings.iter()
            .map(|s| s.text.chars())
            .flatten()
            .filter(|c| *c != ' ')
            .count();

        let mut vertices: Vec<Vertex> = Vec::with_capacity(6 * characters_to_draw_len);

        for label in the_strings.iter() {
            let string_start_x: f32 = label.pos.x;
            let string_start_y: f32 = label.pos.y;

            let mut string_x = string_start_x; // the start x pos
            let string_y = string_start_y; // the start y pos

            for c in label.text.chars() {
                // inserting line breaks
                if c == ' ' {
                    string_x += 10.0;
                    continue;
                }

                // @TODO(lucypero): disabling soft wrap for now
                /*
                        // soft wrap
                        if string_x > WINDOW_INNER_HEIGHT as f32 {
                            string_x = string_start_x;
                            //adding line_height
                            string_y += 22.0;
                        }
                */

                let char_info = self.char_map.get(&c).unwrap();

                let xpos: f32 = string_x + char_info.bearing.x as f32;
                let ypos: f32 = string_y + char_info.size.y as f32 - char_info.bearing.y as f32;
                let w: f32 = char_info.size.x as f32;
                let h: f32 = char_info.size.y as f32;

                vertices.push(Vertex {
                    position: [xpos, ypos - h, 0.0],
                    tex_coords: [0.0, 0.0],
                });
                vertices.push(Vertex {
                    position: [xpos, ypos, 0.0],
                    tex_coords: [0.0, 1.0],
                });
                vertices.push(Vertex {
                    position: [xpos + w, ypos, 0.0],
                    tex_coords: [1.0, 1.0],
                });
                vertices.push(Vertex {
                    position: [xpos, ypos - h, 0.0],
                    tex_coords: [0.0, 0.0],
                });
                vertices.push(Vertex {
                    position: [xpos + w, ypos, 0.0],
                    tex_coords: [1.0, 1.0],
                });
                vertices.push(Vertex {
                    position: [xpos + w, ypos - h, 0.0],
                    tex_coords: [1.0, 0.0],
                });
                string_x += (char_info.advance >> 6) as f32;
            }
        }

        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(vertices.as_slice()),
        );
    }
}

/// Renders all ASCII characters of `font_name` font into textures into a hashmap with a size of `size`
unsafe fn render_font(
    device: &Device,
    queue: &Queue,
    sampler: &Sampler,
    font_name: String,
    font_size: u32,
) -> HashMap<char, Character> {
    let mut ft: ft::FT_Library = std::mem::zeroed();
    let res = ft::FT_Init_FreeType(&mut ft);
    if res != 0 {
        panic!("freetype could not init");
    }

    let mut face: ft::FT_Face = std::mem::zeroed();

    // add null terminator to font name to make it a cstr
    let mut font_name = font_name.clone();
    font_name.push('\0');

    let res = ft::FT_New_Face(ft, font_name.as_ptr() as *const i8, 0, &mut face);
    if res != 0 {
        panic!("could not load font");
    }

    ft::FT_Set_Pixel_Sizes(face, 0, font_size);

    let char_count = (0x20..0x7Fu8).len();

    // render all ascii
    // 128 code points (0x20..0x7F).
    let mut char_map = HashMap::with_capacity(char_count);
    for c in 0x20u8..0x7Fu8 {
        let c = c as char;
        if c == ' ' {
            continue;
        }

        let res = ft::FT_Load_Char(face, c as u32, ft::FT_LOAD_RENDER);
        if res != 0 {
            panic!("could not load {}", c);
        }

        let g = &(*(*face).glyph);
        let char_w = g.bitmap.width as u32;
        let char_h = g.bitmap.rows as u32;
        let bearing = Vector2::new(g.bitmap_left, g.bitmap_top);
        let advance = g.advance.x as u32;

        let char_buffer =
            std::slice::from_raw_parts_mut(g.bitmap.buffer, (char_w * char_h) as usize);

        // creating and rendering texture
        let texture_size = wgpu::Extent3d {
            width: char_w,
            height: char_h,
            depth_or_array_layers: 1,
        };
        let character_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: None,
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &character_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &char_buffer,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(char_w),
                rows_per_image: std::num::NonZeroU32::new(char_h),
            },
            texture_size,
        );

        // create the view
        let character_texture_view =
            character_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let character_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &get_normal_texture_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&character_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        char_map.insert(
            c,
            Character {
                bind_group: character_texture_bind_group,
                size: Vector2::new(char_w, char_h),
                bearing,
                advance,
            },
        );
    }

    ft::FT_Done_Face(face);
    ft::FT_Done_FreeType(ft);

    char_map
}
