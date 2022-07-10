use cgmath::{Matrix, Matrix2, Matrix3, Matrix4, SquareMatrix, Vector2, Vector3, Vector4};
use std::ops::Add;
use std::time::Duration;

use crate::input::{Input, Keys};

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
    pub const COLOR_1: Vec4 = Vec4::new(0.466, 0.882, 0.996, 1.0);
    pub const COLOR_2: Vec4 = Vec4::new(0.815, 0.121, 1., 1.0);
    pub const COLOR_3: Vec4 = Vec4::new(0.039, 1., 0.647, 1.0);
    pub const COLORS: [Vec4; 7] = [RED, GREEN, PINK, YELLOW, COLOR_1, COLOR_2, COLOR_3];
}

pub struct Block {
    pub pos: Pos,
    pub color: Vec4,
}

pub struct BlockSet {
    pub positions: Vec<bool>,
    pub pos_w: usize,
    pub pos: Pos,
    //pos of bottom left block
    pub color: Vec4,
}

impl BlockSet {
    // sample block set
    fn new_t() -> Self {
        let positions = vec![false, false, false, true, true, true, false, true, false];

        Self {
            positions,
            pos: Pos::new(5, 20),
            pos_w: 3,
            color: color::COLORS[0],
        }
    }

    fn new_square() -> Self {
        let positions = vec![true, true, true, true];

        Self {
            positions,
            pos: Pos::new(5, 20),
            pos_w: 2,
            color: color::COLORS[1],
        }
    }

    fn new_line() -> Self {
        let positions = vec![
            false, false, false, false, false, false, false, false, true, true, true, true, false,
            false, false, false,
        ];

        Self {
            positions,
            pos: Pos::new(5, 20),
            pos_w: 4,
            color: color::COLORS[2],
        }
    }

    fn new_l() -> Self {
        let positions = vec![false, false, false, true, true, true, false, false, true];

        Self {
            positions,
            pos: Pos::new(5, 20),
            pos_w: 3,
            color: color::COLORS[3],
        }
    }

    fn new_j() -> Self {
        let positions = vec![false, false, false, true, true, true, true, false, false];

        Self {
            positions,
            pos: Pos::new(5, 20),
            pos_w: 3,
            color: color::COLORS[4],
        }
    }

    fn new_s() -> Self {
        let positions = vec![false, false, false, false, true, true, true, true, false];

        Self {
            positions,
            pos: Pos::new(5, 20),
            pos_w: 3,
            color: color::COLORS[5],
        }
    }

    fn new_z() -> Self {
        let positions = vec![false, false, false, true, true, false, false, true, true];

        Self {
            positions,
            pos: Pos::new(5, 20),
            pos_w: 3,
            color: color::COLORS[6],
        }
    }

    fn rotate<const CW: bool>(&mut self) {
        match self.pos_w {
            2 => {
                // we don't actually do anything here hahahha
                //  because the square fills all slots anyway.
            }
            3 => {
                assert_eq!(self.positions.len(), 3 * 3);
                let new_mat = rot_mat3::<CW>(Matrix3::new(
                    self.positions[0],
                    self.positions[1],
                    self.positions[2],
                    self.positions[3],
                    self.positions[4],
                    self.positions[5],
                    self.positions[6],
                    self.positions[7],
                    self.positions[8],
                ));

                self.positions[0] = new_mat.x.x;
                self.positions[1] = new_mat.x.y;
                self.positions[2] = new_mat.x.z;
                self.positions[3] = new_mat.y.x;
                self.positions[4] = new_mat.y.y;
                self.positions[5] = new_mat.y.z;
                self.positions[6] = new_mat.z.x;
                self.positions[7] = new_mat.z.y;
                self.positions[8] = new_mat.z.z;
            }
            4 => {
                assert_eq!(self.positions.len(), 4 * 4);
                let new_mat = rot_mat4::<CW>(Matrix4::new(
                    self.positions[0],
                    self.positions[1],
                    self.positions[2],
                    self.positions[3],
                    self.positions[4],
                    self.positions[5],
                    self.positions[6],
                    self.positions[7],
                    self.positions[8],
                    self.positions[9],
                    self.positions[10],
                    self.positions[11],
                    self.positions[12],
                    self.positions[13],
                    self.positions[14],
                    self.positions[15],
                ));

                self.positions[0] = new_mat.x.x;
                self.positions[1] = new_mat.x.y;
                self.positions[2] = new_mat.x.z;
                self.positions[3] = new_mat.x.w;
                self.positions[4] = new_mat.y.x;
                self.positions[5] = new_mat.y.y;
                self.positions[6] = new_mat.y.z;
                self.positions[7] = new_mat.y.w;
                self.positions[8] = new_mat.z.x;
                self.positions[9] = new_mat.z.y;
                self.positions[10] = new_mat.z.z;
                self.positions[11] = new_mat.z.w;
                self.positions[12] = new_mat.w.x;
                self.positions[13] = new_mat.w.y;
                self.positions[14] = new_mat.w.z;
                self.positions[15] = new_mat.w.w;
            }
            _ => panic!("what block set is this?!"),
        };
    }
}

fn rot_mat3<const RIGHT: bool>(mat: Matrix3<bool>) -> Matrix3<bool> {
    if RIGHT {
        Matrix3::new(
            mat[0][2], mat[1][2], mat[2][2], mat[0][1], mat[1][1], mat[2][1], mat[0][0], mat[1][0],
            mat[2][0],
        )
    } else {
        Matrix3::new(
            mat[2][0], mat[1][0], mat[0][0], mat[2][1], mat[1][1], mat[0][1], mat[2][2], mat[1][2],
            mat[0][2],
        )
    }
}

fn rot_mat4<const RIGHT: bool>(mat: Matrix4<bool>) -> Matrix4<bool> {
    if RIGHT {
        Matrix4::new(
            mat[0][3], mat[1][3], mat[2][3], mat[3][3], mat[0][2], mat[1][2], mat[2][2], mat[3][2],
            mat[0][1], mat[1][1], mat[2][1], mat[3][1], mat[0][0], mat[1][0], mat[2][0], mat[3][0],
        )
    } else {
        Matrix4::new(
            mat[3][0], mat[2][0], mat[1][0], mat[0][0], mat[3][1], mat[2][1], mat[1][1], mat[0][1],
            mat[3][2], mat[2][2], mat[1][2], mat[0][2], mat[3][3], mat[2][3], mat[1][3], mat[0][3],
        )
    }
}

pub struct Game {
    pub fixed_blocks: Vec<Block>,
    pub active_block_set: Option<BlockSet>,
    pub grid_w: usize,
    pub grid_h: usize,
    // the bottom left position of the grid.
    pub grid_pos: Vec2,
    tick_timer: Duration,
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
            fixed_blocks: blocks,
            grid_w: 20,
            grid_h: 50,
            grid_pos: Vec2::new(10., 500.),
            active_block_set: Some(BlockSet::new_line()),
            tick_timer: Duration::from_secs(0),
        }
    }

    fn down_tick(&mut self) {
        if let Some(ref mut act_block) = self.active_block_set {
            act_block.pos.y -= 1;
        }
    }

    pub fn update(&mut self, input: &Input, dt: Duration) {
        self.tick_timer += dt;

        if self.tick_timer.as_secs() >= 1 {
            //perform tick
            self.down_tick();

            self.tick_timer = Duration::from_secs(0);
        }

        static mut BLOCK_TYPE_SPAWN: usize = 0;

        if input.get_key_down(Keys::Left) {
            //swap block
            unsafe {
                BLOCK_TYPE_SPAWN += 1;
                if BLOCK_TYPE_SPAWN > 6 {
                    BLOCK_TYPE_SPAWN = 0;
                }

                match BLOCK_TYPE_SPAWN {
                    0 => {
                        self.active_block_set = Some(BlockSet::new_line());
                    }
                    1 => {
                        self.active_block_set = Some(BlockSet::new_l());
                    }
                    2 => {
                        self.active_block_set = Some(BlockSet::new_j());
                    }
                    3 => {
                        self.active_block_set = Some(BlockSet::new_t());
                    }
                    4 => {
                        self.active_block_set = Some(BlockSet::new_square());
                    }
                    5 => {
                        self.active_block_set = Some(BlockSet::new_s());
                    }
                    6 => {
                        self.active_block_set = Some(BlockSet::new_z());
                    }
                    _ => {
                        panic!("never gets here");
                    }
                }
            }
        }

        if let Some(ref mut act_block) = self.active_block_set {
            if input.get_key_down(Keys::A) {
                act_block.pos.x -= 1;
            }
            if input.get_key_down(Keys::D) {
                act_block.pos.x += 1;
            }
            if input.get_key_down(Keys::W) {
                act_block.pos.y += 1;
            }
            if input.get_key_down(Keys::S) {
                act_block.pos.y -= 1;
            }
            if input.get_key_down(Keys::H) {
                act_block.rotate::<false>();
            }
            if input.get_key_down(Keys::J) {
                act_block.rotate::<true>();
            }
        }
    }
}
