use cgmath::{
    Vector3,
    Vector2,
    Vector4,
};

use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};
use crate::input::{
    Input,
    Keys
};

// (0,0) is bottom left
type Pos = Vector2<i32>;
type Vec2 = Vector2<f32>;
type Vec3 = Vector3<f32>;
type Vec4 = Vector4<f32>;

pub const BLOCK_SIZE: f32 = 20.0;
pub const BLOCK_GAP: f32 = 0.0;

// cute block colors
mod color {
    use crate::game::Vec4;

    pub const RED: Vec4 = Vec4::new(1.0, 0.0, 0.0, 1.0);
    pub const GREEN: Vec4 = Vec4::new(0.0, 1.0, 0.0, 1.0);
    pub const PINK: Vec4 = Vec4::new(1., 0.541, 0.909, 1.0);
    pub const YELLOW: Vec4 = Vec4::new(1., 0.968, 0.541, 1.0);
    pub const COLORS: [Vec4; 4] = [RED, GREEN, PINK, YELLOW];
}

pub struct Block {
    pub pos: Pos,
    pub color: Vec4,
}

pub struct Game {
    pub blocks: Vec<Block>,
    pub grid_w: usize,
    pub grid_h: usize,
    // the bottom left position of the grid.
    pub grid_pos: Vec2,
}

impl Game {
    pub fn new() -> Self {
        let blocks = vec![
            Block {
                pos: Pos::new(0, 0),
                color: color::RED,
            },
            Block {
                pos: Pos::new(1, 0),
                color: color::PINK,
            },
            Block {
                pos: Pos::new(2, 0),
                color: color::YELLOW,
            },
            Block {
                pos: Pos::new(3, 0),
                color: color::GREEN,
            },
        ];

        Self {
            blocks,
            grid_w: 20,
            grid_h: 50,
            grid_pos: Vec2::new(10., 500.),
        }
    }

    pub fn update(&mut self, input: &Input) {
        for i in self.blocks.iter_mut() {
            if input.get_key_down(Keys::W) {
                i.pos.y += 1;
            }
            if input.get_key_down(Keys::S) {
                i.pos.y -= 1;
            }
            if input.get_key_down(Keys::A) {
                i.pos.x -= 1;
            }
            if input.get_key_down(Keys::D) {
                i.pos.x += 1;
            }

        }
    }
}
