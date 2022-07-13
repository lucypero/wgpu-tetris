use cgmath::{Vector2, Vector3, Vector4};
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

pub struct Camera {
    pub initial_size: Vector2<f32>,
    pub position: Vector2<f32>,
    pub zoom_amount: f32,
}

impl Camera {
    fn do_move_controls(&mut self, input: &Input) {
        const CAM_SPEED: f32 = 20.0;
        const CAM_ZOOM_STEP: f32 = 0.03;
        const CAM_ZOOM_MIN: f32 = 0.13;
        const CAM_ZOOM_MAX: f32 = 4.0;

        //move camera
        if input.get_key(Keys::W) {
            self.position += Vector2::new(0.0, CAM_SPEED);
        }
        if input.get_key(Keys::A) {
            self.position += Vector2::new(CAM_SPEED, 0.0);
        }
        if input.get_key(Keys::S) {
            self.position += Vector2::new(0.0, -CAM_SPEED);
        }
        if input.get_key(Keys::D) {
            self.position += Vector2::new(-CAM_SPEED, 0.0);
        }
        if input.get_key(Keys::NumpadAdd) {
            self.zoom_amount -= CAM_ZOOM_STEP;
            if self.zoom_amount <= CAM_ZOOM_MIN {
                self.zoom_amount = CAM_ZOOM_MIN;
            }
        }
        if input.get_key(Keys::NumpadSubtract) {
            self.zoom_amount += CAM_ZOOM_STEP;
            if self.zoom_amount >= CAM_ZOOM_MAX {
                self.zoom_amount = CAM_ZOOM_MAX;
            }
        }
    }
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
                let old_pos = self.positions.clone();
                if CW {
                    self.positions[0] = old_pos[3 * 0 + 2];
                    self.positions[1] = old_pos[3 * 1 + 2];
                    self.positions[2] = old_pos[3 * 2 + 2];
                    self.positions[3] = old_pos[3 * 0 + 1];
                    self.positions[4] = old_pos[3 * 1 + 1];
                    self.positions[5] = old_pos[3 * 2 + 1];
                    self.positions[6] = old_pos[3 * 0 + 0];
                    self.positions[7] = old_pos[3 * 1 + 0];
                    self.positions[8] = old_pos[3 * 2 + 0];
                } else {
                    self.positions[0] = old_pos[3 * 2 + 0];
                    self.positions[1] = old_pos[3 * 1 + 0];
                    self.positions[2] = old_pos[3 * 0 + 0];
                    self.positions[3] = old_pos[3 * 2 + 1];
                    self.positions[4] = old_pos[3 * 1 + 1];
                    self.positions[5] = old_pos[3 * 0 + 1];
                    self.positions[6] = old_pos[3 * 2 + 2];
                    self.positions[7] = old_pos[3 * 1 + 2];
                    self.positions[8] = old_pos[3 * 0 + 2];
                }
            }
            4 => {
                assert_eq!(self.positions.len(), 4 * 4);
                let old_pos = self.positions.clone();
                if CW {
                    self.positions[0] = old_pos[4 * 0 + 3];
                    self.positions[1] = old_pos[4 * 1 + 3];
                    self.positions[2] = old_pos[4 * 2 + 3];
                    self.positions[3] = old_pos[4 * 3 + 3];
                    self.positions[4] = old_pos[4 * 0 + 2];
                    self.positions[5] = old_pos[4 * 1 + 2];
                    self.positions[6] = old_pos[4 * 2 + 2];
                    self.positions[7] = old_pos[4 * 3 + 2];
                    self.positions[8] = old_pos[4 * 0 + 1];
                    self.positions[9] = old_pos[4 * 1 + 1];
                    self.positions[10] = old_pos[4 * 2 + 1];
                    self.positions[11] = old_pos[4 * 3 + 1];
                    self.positions[12] = old_pos[4 * 0 + 0];
                    self.positions[13] = old_pos[4 * 1 + 0];
                    self.positions[14] = old_pos[4 * 2 + 0];
                    self.positions[15] = old_pos[4 * 3 + 0];
                } else {
                    self.positions[0] = old_pos[4 * 3 + 0];
                    self.positions[1] = old_pos[4 * 2 + 0];
                    self.positions[2] = old_pos[4 * 1 + 0];
                    self.positions[3] = old_pos[4 * 0 + 0];
                    self.positions[4] = old_pos[4 * 3 + 1];
                    self.positions[5] = old_pos[4 * 2 + 1];
                    self.positions[6] = old_pos[4 * 1 + 1];
                    self.positions[7] = old_pos[4 * 0 + 1];
                    self.positions[8] = old_pos[4 * 3 + 2];
                    self.positions[9] = old_pos[4 * 2 + 2];
                    self.positions[10] = old_pos[4 * 1 + 2];
                    self.positions[11] = old_pos[4 * 0 + 2];
                    self.positions[12] = old_pos[4 * 3 + 3];
                    self.positions[13] = old_pos[4 * 2 + 3];
                    self.positions[14] = old_pos[4 * 1 + 3];
                    self.positions[15] = old_pos[4 * 0 + 3];
                }
            }
            _ => panic!("what block set is this?!"),
        };
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
    pub camera: Camera,
}

impl Game {
    pub fn new(cam_initial_size: Vector2<u32>) -> Self {
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
            camera: Camera {
                initial_size: Vector2::new(cam_initial_size.x as f32, cam_initial_size.y as f32),
                position: Vector2::new(0., 0.),
                zoom_amount: 1.0,
            },
        }
    }

    fn do_block_controls(&mut self, input: &Input) {
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
            // rotate counter clockwise
            if input.get_key_down(Keys::H) {
                act_block.rotate::<false>();
            }
            // rotate clockwise
            if input.get_key_down(Keys::J) {
                act_block.rotate::<true>();
            }
            // do a 180
            if input.get_key_down(Keys::K) {
                // yeah i just rotate twice, deal with it
                act_block.rotate::<true>();
                act_block.rotate::<true>();
            }
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

        // (for testing) swappin active block type
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

        static mut CONTROL_CAMERA: bool = false;

        unsafe {
            if input.get_key_down(Keys::Down) {
                CONTROL_CAMERA = !CONTROL_CAMERA;
            }

            if CONTROL_CAMERA {
                self.camera.do_move_controls(&input);
            } else {
                self.do_block_controls(&input);
            }
        }
    }
}
