use crate::gui::*;
use crate::input::{Input, Keys};
use libs::cgmath::{Vector2, Vector3, Vector4};
use libs::rand;
use libs::rand::seq::SliceRandom;
use libs::thunderdome::{Arena, Index};
use std::collections::{HashMap, VecDeque};
use std::ops::{Add, Mul};
use std::time::Duration;

pub type v2 = Vector2<f32>;
pub type v3 = Vector3<f32>;
pub type v4 = Vector4<f32>;
pub type v2i = Vector2<i32>;
pub type v3i = Vector3<i32>;

// (0,0) is bottom left (of grid)
pub const BLOCK_SIZE: f32 = 32.0;
pub const GRID_WIDTH: usize = 10;
pub const GRID_HEIGHT: usize = 20;

const ACTIVE_BLOCK_START_POS: v2i = v2i::new(3, 20);
const HOLD_BLOCK_POS: v2 = v2::new(30.0, BLOCK_SIZE * 10.0);
const NEXT_BLOCKS_COUNT: usize = 5;
const FIRST_BLOCK_POS_Y: f32 = 250.0;
const BLOCKS_TWEEN_DURATION: u64 = 150;
const BLOCK_FALL_DURATION:u64 = 1000;

#[derive(Copy, Clone, Debug)]
enum BlockSetType {
    T,
    Square,
    Line,
    L,
    J,
    S,
    Z,
}
const BLOCK_COUNT: usize = 8;

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum EaseFunction {
    Linear,
    CircOut,
    EaseOutBounce,
    EaseOutElastic,
    COUNT,
}

/// T should only be a float or a vec of floats
pub struct Tween<T> {
    pub start: T,
    pub end: T,
    pub duration: Duration,
    pub ease_func: EaseFunction,
    /// 0 to 1
    pub t: f32,
}

// mixing functions
pub fn tween_mix_v4(v1: v4, v2: v4, t: f32) -> v4 {
    v4::new(
        v1.x * (1.0 - t) + v2.x * t,
        v1.y * (1.0 - t) + v2.y * t,
        v1.z * (1.0 - t) + v2.z * t,
        v1.w * (1.0 - t) + v2.w * t,
    )
}

pub fn tween_get_value<T>(tween: &Tween<T>) -> T
where
    T: Mul<f32, Output = T> + Add<T, Output = T> + Copy + Clone,
{
    let t = match tween.ease_func {
        EaseFunction::Linear => tween.t,
        EaseFunction::CircOut => f32::sqrt(1.0 - f32::powf(tween.t - 1.0, 2.0)),
        EaseFunction::EaseOutBounce => {
            const N1: f32 = 7.5625;
            const D1: f32 = 2.75;

            if tween.t < 1.0 / D1 {
                N1 * tween.t * tween.t
            } else if tween.t < 2.0 / D1 {
                let temp = tween.t - (1.5 / D1);
                N1 * temp * temp + 0.75
            } else if tween.t < 2.5 / D1 {
                let temp = tween.t - (2.25 / D1);
                N1 * temp * temp + 0.9375
            } else {
                let temp = tween.t - (2.625 / D1);
                N1 * temp * temp + 0.984375
            }
        }
        EaseFunction::EaseOutElastic => {
            const C4: f32 = (2.0 * std::f32::consts::PI) / 5.0;

            if tween.t < 0.0 {
                0.0
            } else if tween.t > 1.0 {
                1.0
            } else {
                f32::powf(2.0, -10.0 * tween.t) * f32::sin((tween.t * 10.0 - 0.75) * C4) + 1.0
            }
        }
        _ => tween.t,
    };

    tween.start * (1.0 - t) + tween.end * t
}

pub fn tween_new_pos<T>(tween: &Tween<T>, new_start: T, new_end: T) -> Tween<T>
where
    T: Copy + Clone,
{
    Tween {
        start: new_start,
        end: new_end,
        ease_func: tween.ease_func,
        duration: tween.duration,
        t: 0.,
    }
}

#[derive(PartialEq)]
enum GameState {
    Menu,
    Playing,
    Pause,
}

pub struct Game {
    pub blocks: Arena<Block>,
    active_block_set: Option<BlockSet>,
    hold_block_preview: Option<StaticBlockSet>,
    pub grid: Grid,
    tick_timer: Duration,
    pub camera: Camera,
    hold_enabled: bool,
    next_blocks: VecDeque<StaticBlockSet>,
    next_block_types: Vec<BlockSetType>,
    next_block_index: usize,
    rng: rand::rngs::ThreadRng,
    pub gui: Gui,
    current_slide: i32,
    game_state: GameState,
}

struct StaticBlockSet {
    positions: Vec<bool>,
    set_type: BlockSetType,
    blocks: Vec<Index>,
    pos_w: usize,
    pos: v2,
}

struct BlockSet {
    static_block_set: StaticBlockSet,
    grid_pos: v2i,
    blocks_ghost: Vec<Index>,
    ghost_offset: i32, //y offset from grid origin
}

pub struct Block {
    pub pos: v2,
    pub color: v4,
    pub tween: Tween<v2>,
}

pub struct Camera {
    pub initial_size: Vector2<f32>,
    pub position: Vector2<f32>,
    pub zoom_amount: f32,
    pub tween: Tween<v2>,
}

pub struct Grid {
    pub block_positions: HashMap<v2i, Index>,
    pub pos: v2,
    // the bottom left position of the grid.
    pub width: usize,
    pub height: usize,
}

// cute block colors
pub mod color {
    use libs::rand::Rng;

    use crate::game::v4;

    pub const RED: v4 = v4::new(1.0, 0.0, 0.0, 1.0);
    pub const GREEN: v4 = v4::new(0.0, 1.0, 0.0, 1.0);
    pub const PINK: v4 = v4::new(1., 0.541, 0.909, 1.0);
    pub const YELLOW: v4 = v4::new(1., 0.968, 0.541, 1.0);
    pub const COLOR_1: v4 = v4::new(0.466, 0.882, 0.996, 1.0);
    pub const COLOR_2: v4 = v4::new(0.815, 0.121, 1., 1.0);
    pub const COLOR_3: v4 = v4::new(0.039, 1., 0.647, 1.0);
    pub const COLORS: [v4; 7] = [RED, GREEN, PINK, YELLOW, COLOR_1, COLOR_2, COLOR_3];

    pub const GHOST: v4 = v4::new(1., 1., 1., 0.3);
    pub const GRID_BG: v4 = v4::new(1., 1., 1., 0.1);
    pub const HUD_BG: v4 = v4::new(0., 0., 0., 1.);

    pub const NORMAL_MENU_BORDER_COLOR: v4 = v4::new(1., 1., 1., 0.5);
    pub const HOT_MENU_BORDER_COLOR: v4 = PINK;

    pub fn get_random(rng: &mut libs::rand::rngs::ThreadRng) -> v4 {
        COLORS[rng.gen_range(0..COLORS.len())]
    }
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

impl BlockSet {
    fn try_to_fit(&mut self, new_positions: Vec<bool>, grid: &Grid) -> (bool, i32) {
        const MAX_ITERATIONS: i32 = 5;

        let mut new_testing_pos;

        /*
        0
        1 0
        1 1
        0 1
        -1 1
        -1 0
        -1 -1
        0 -1
        1 -1

        2 0
        2 1 <- first loop end
        2 2
        1 2
        0 2
        -1 2 <- second loop end
        -2 2
        -2 1
        -2 0
        -2 -1 <- third loop end
        -2 -2
        -1 -2
        0 -2
        1 -2 <- fourth loop end
        2 -2
        2 -1 <- fifth loop end

         */

        if self.does_fit(self.grid_pos, &new_positions, grid) {
            self.static_block_set.positions = new_positions;
            return (true, 0);
        }

        for iter_number in 1..=MAX_ITERATIONS {
            for y_2 in 0..iter_number {
                new_testing_pos = v2i::new(self.grid_pos.x + iter_number, self.grid_pos.y + y_2);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.grid_pos = new_testing_pos;
                    self.static_block_set.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for x_2 in (-iter_number + 1..=iter_number).rev() {
                new_testing_pos = v2i::new(self.grid_pos.x + x_2, self.grid_pos.y + iter_number);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.grid_pos = new_testing_pos;
                    self.static_block_set.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for y_2 in (-iter_number + 1..=iter_number).rev() {
                new_testing_pos = v2i::new(self.grid_pos.x - iter_number, self.grid_pos.y + y_2);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.grid_pos = new_testing_pos;
                    self.static_block_set.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for x_2 in -iter_number..iter_number {
                new_testing_pos = v2i::new(self.grid_pos.x + x_2, self.grid_pos.y - iter_number);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.grid_pos = new_testing_pos;
                    self.static_block_set.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for y_2 in (-1..=iter_number).rev() {
                new_testing_pos = v2i::new(self.grid_pos.x + iter_number, self.grid_pos.y + y_2);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.grid_pos = new_testing_pos;
                    self.static_block_set.positions = new_positions;
                    return (true, iter_number);
                }
            }
        }

        (false, 0)
    }

    fn remove_blocks(&mut self, blocks: &mut Arena<Block>) {
        let mut blocks_i = 0;
        for is_occupied in self.static_block_set.positions.iter() {
            if !is_occupied {
                continue;
            }

            blocks.remove(self.blocks_ghost[blocks_i]);
            blocks.remove(self.static_block_set.blocks[blocks_i]);
            blocks_i += 1;
        }
    }

    fn remove_ghost_blocks(&mut self, blocks: &mut Arena<Block>) {
        let mut blocks_i = 0;
        for is_occupied in self.static_block_set.positions.iter() {
            if !is_occupied {
                continue;
            }

            blocks.remove(self.blocks_ghost[blocks_i]);
            blocks_i += 1;
        }
    }

    // does it fit in an alternate global position or alternate position values?
    fn does_fit(&self, new_pos: v2i, new_positions: &Vec<bool>, grid: &Grid) -> bool {
        for (index, is_occupied) in new_positions.iter().enumerate() {
            if !is_occupied {
                continue;
            }

            let x = index % self.static_block_set.pos_w;
            let y = index / self.static_block_set.pos_w;
            let pos_to_test = v2i::new(new_pos.x + x as i32, new_pos.y + y as i32);

            if grid.block_positions.contains_key(&pos_to_test) || !grid.is_inside(pos_to_test) {
                return false;
            }
        }
        true
    }

    fn from_pos(
        grid: &Grid,
        pos: v2i,
        pos_w: usize,
        positions: Vec<bool>,
        arena: &mut Arena<Block>,
        set_type: BlockSetType,
        color: v4,
    ) -> Self {
        let mut blocks = Vec::new();
        let mut blocks_ghost = Vec::new();

        // creating the new blocks that we need
        for a in positions.iter() {
            if !a {
                continue;
            }

            blocks.push(arena.insert(Block::new(v2::new(0.0, 0.0), color)));
            blocks_ghost.push(arena.insert(Block::new(v2::new(0.0, 0.0), color::GHOST)));
        }

        let static_block_set = StaticBlockSet {
            positions,
            set_type,
            blocks,
            pos_w,
            pos: grid.get_real_position(pos),
        };

        let mut act_block = Self {
            static_block_set,
            grid_pos: pos,
            blocks_ghost,
            ghost_offset: 0,
        };

        act_block.update_pos::<false>(grid, pos, arena);

        act_block
    }

    fn new_t(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, true, false, true, false];
        let pos_w = 3;
        let pos = ACTIVE_BLOCK_START_POS;
        let color = color::COLORS[0];

        Self::from_pos(grid, pos, pos_w, positions, arena, BlockSetType::T, color)
    }

    fn new_square(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![true, true, true, true];
        let pos_w = 2;
        let pos = ACTIVE_BLOCK_START_POS;
        let color = color::COLORS[1];

        Self::from_pos(
            grid,
            pos,
            pos_w,
            positions,
            arena,
            BlockSetType::Square,
            color,
        )
    }

    fn new_line(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![
            false, false, false, false, false, false, false, false, true, true, true, true, false,
            false, false, false,
        ];

        let pos_w = 4;
        let pos = ACTIVE_BLOCK_START_POS;
        let color = color::COLORS[2];

        Self::from_pos(
            grid,
            pos,
            pos_w,
            positions,
            arena,
            BlockSetType::Line,
            color,
        )
    }

    fn new_l(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, true, false, false, true];
        let pos_w = 3;
        let pos = ACTIVE_BLOCK_START_POS;
        let color = color::COLORS[3];

        Self::from_pos(grid, pos, pos_w, positions, arena, BlockSetType::L, color)
    }

    fn new_j(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, true, true, false, false];
        let pos_w = 3;
        let pos = ACTIVE_BLOCK_START_POS;
        let color = color::COLORS[4];

        Self::from_pos(grid, pos, pos_w, positions, arena, BlockSetType::J, color)
    }

    fn new_s(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, false, true, true, true, true, false];
        let pos_w = 3;
        let pos = ACTIVE_BLOCK_START_POS;
        let color = color::COLORS[5];

        Self::from_pos(grid, pos, pos_w, positions, arena, BlockSetType::S, color)
    }

    fn new_z(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, false, false, true, true];
        let pos_w = 3;
        let pos = ACTIVE_BLOCK_START_POS;
        let color = color::COLORS[6];

        Self::from_pos(grid, pos, pos_w, positions, arena, BlockSetType::Z, color)
    }

    fn get_rotated_pos<const CW: bool>(positions: &Vec<bool>, width: usize) -> Vec<bool> {
        match width {
            2 => {
                // we don't actually do anything here hahahha
                //  because the square fills all slots anyway.
                positions.clone()
            }
            3 => {
                assert_eq!(positions.len(), 3 * 3);
                if CW {
                    vec![
                        positions[3 * 0 + 2],
                        positions[3 * 1 + 2],
                        positions[3 * 2 + 2],
                        positions[3 * 0 + 1],
                        positions[3 * 1 + 1],
                        positions[3 * 2 + 1],
                        positions[3 * 0 + 0],
                        positions[3 * 1 + 0],
                        positions[3 * 2 + 0],
                    ]
                } else {
                    vec![
                        positions[3 * 2 + 0],
                        positions[3 * 1 + 0],
                        positions[3 * 0 + 0],
                        positions[3 * 2 + 1],
                        positions[3 * 1 + 1],
                        positions[3 * 0 + 1],
                        positions[3 * 2 + 2],
                        positions[3 * 1 + 2],
                        positions[3 * 0 + 2],
                    ]
                }
            }
            4 => {
                assert_eq!(positions.len(), 4 * 4);
                if CW {
                    vec![
                        positions[4 * 0 + 3],
                        positions[4 * 1 + 3],
                        positions[4 * 2 + 3],
                        positions[4 * 3 + 3],
                        positions[4 * 0 + 2],
                        positions[4 * 1 + 2],
                        positions[4 * 2 + 2],
                        positions[4 * 3 + 2],
                        positions[4 * 0 + 1],
                        positions[4 * 1 + 1],
                        positions[4 * 2 + 1],
                        positions[4 * 3 + 1],
                        positions[4 * 0 + 0],
                        positions[4 * 1 + 0],
                        positions[4 * 2 + 0],
                        positions[4 * 3 + 0],
                    ]
                } else {
                    vec![
                        positions[4 * 3 + 0],
                        positions[4 * 2 + 0],
                        positions[4 * 1 + 0],
                        positions[4 * 0 + 0],
                        positions[4 * 3 + 1],
                        positions[4 * 2 + 1],
                        positions[4 * 1 + 1],
                        positions[4 * 0 + 1],
                        positions[4 * 3 + 2],
                        positions[4 * 2 + 2],
                        positions[4 * 1 + 2],
                        positions[4 * 0 + 2],
                        positions[4 * 3 + 3],
                        positions[4 * 2 + 3],
                        positions[4 * 1 + 3],
                        positions[4 * 0 + 3],
                    ]
                }
            }
            _ => panic!("what block set is this?!"),
        }
    }

    fn update_pos<const INTERPOLATE: bool>(
        &mut self,
        grid: &Grid,
        new_pos: v2i,
        blocks: &mut Arena<Block>,
    ) {
        let pos: v2 = grid.get_real_position(new_pos);
        self.grid_pos = new_pos;

        let mut block_index = 0;

        // getting ghost block position
        let mut ghost_pos = v2i::new(new_pos.x, new_pos.y);

        while self.does_fit(ghost_pos, &self.static_block_set.positions, grid) {
            ghost_pos.y -= 1;
        }

        ghost_pos.y += 1;
        let ghost_pos: v2 = grid.get_real_position(ghost_pos);

        for (index, a) in self.static_block_set.positions.iter().enumerate() {
            if !a {
                // u gotta put this away i think
                continue;
            }

            let pos_x = index % self.static_block_set.pos_w;
            let pos_y = index / self.static_block_set.pos_w;

            let pos_f32 = v2::new(
                pos.x + BLOCK_SIZE * pos_x as f32,
                pos.y - BLOCK_SIZE * pos_y as f32,
            );

            let ghost_pos_f32 = v2::new(
                pos.x + BLOCK_SIZE * pos_x as f32,
                ghost_pos.y - BLOCK_SIZE * pos_y as f32,
            );

            blocks[self.static_block_set.blocks[block_index]]
                .update_target_pos::<INTERPOLATE>(pos_f32);
            blocks[self.blocks_ghost[block_index]].update_target_pos::<INTERPOLATE>(ghost_pos_f32);
            block_index += 1;
        }
    }
}

impl StaticBlockSet {
    fn from_pos(
        positions: Vec<bool>,
        pos_w: usize,
        arena: &mut Arena<Block>,
        set_type: BlockSetType,
        color: v4,
    ) -> Self {
        let mut blocks = Vec::new();

        // creating the new blocks that we need
        for a in positions.iter() {
            if !a {
                continue;
            }

            blocks.push(arena.insert(Block::new(v2::new(0.0, 0.0), color)));
        }

        Self {
            positions,
            set_type,
            blocks,
            pos_w,
            pos: v2::new(0.0, 0.0),
        }
    }

    fn new_t(arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, true, false, true, false];
        Self::from_pos(positions, 3, arena, BlockSetType::T, color::COLORS[0])
    }

    fn new_square(arena: &mut Arena<Block>) -> Self {
        let positions = vec![true, true, true, true];
        Self::from_pos(positions, 2, arena, BlockSetType::Square, color::COLORS[1])
    }

    fn new_line(arena: &mut Arena<Block>) -> Self {
        let positions = vec![
            false, false, false, false, false, false, false, false, true, true, true, true, false,
            false, false, false,
        ];
        Self::from_pos(positions, 4, arena, BlockSetType::Line, color::COLORS[2])
    }

    fn new_l(arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, true, false, false, true];
        Self::from_pos(positions, 3, arena, BlockSetType::L, color::COLORS[3])
    }

    fn new_j(arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, true, true, false, false];
        Self::from_pos(positions, 3, arena, BlockSetType::J, color::COLORS[4])
    }

    fn new_s(arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, false, true, true, true, true, false];
        Self::from_pos(positions, 3, arena, BlockSetType::S, color::COLORS[5])
    }

    fn new_z(arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, false, false, true, true];
        Self::from_pos(positions, 3, arena, BlockSetType::Z, color::COLORS[6])
    }

    // given a new pos, it sets the new pos then moves all the owned blocks
    fn update_pos<const INTERPOLATE: bool>(&mut self, new_pos: v2, blocks: &mut Arena<Block>) {
        self.pos = new_pos;

        let mut block_index = 0;

        for (index, a) in self.positions.iter().enumerate() {
            if !a {
                // u gotta put this away i think
                continue;
            }

            let pos_x = index % self.pos_w;
            let pos_y = index / self.pos_w;

            let pos_f32 = v2::new(
                self.pos.x + BLOCK_SIZE * pos_x as f32,
                self.pos.y - BLOCK_SIZE * pos_y as f32,
            );

            blocks[self.blocks[block_index]].update_target_pos::<INTERPOLATE>(pos_f32);
            block_index += 1;
        }
    }
}

impl Grid {
    // we do not check if above
    fn is_inside(&self, pos: v2i) -> bool {
        pos.x < self.width as i32 && pos.x >= 0 && pos.y >= 0
    }

    fn get_real_position(&self, pos: v2i) -> v2 {
        v2::new(
            self.pos.x + pos.x as f32 * BLOCK_SIZE,
            self.pos.y - pos.y as f32 * BLOCK_SIZE,
        )
    }
}

fn put_down_act_block(grid: &mut Grid, act_block: &BlockSet, blocks: &mut Arena<Block>) {
    {
        let mut blocks_i = 0;
        for (index, is_occupied) in act_block.static_block_set.positions.iter().enumerate() {
            if !is_occupied {
                continue;
            }

            let x = index % act_block.static_block_set.pos_w;
            let y = index / act_block.static_block_set.pos_w;
            let the_pos = v2i::new(
                act_block.grid_pos.x + x as i32,
                act_block.grid_pos.y + y as i32,
            );

            let block_index = act_block.static_block_set.blocks[blocks_i];

            grid.block_positions.insert(the_pos, block_index);
            //removing ghost blocks
            blocks.remove(act_block.blocks_ghost[blocks_i]);
            blocks[block_index].update_target_pos::<true>(grid.get_real_position(the_pos));
            blocks_i += 1;
        }
    }

    // checking if u cleared lines
    // check act_block.pos.y to act_block.pos.y + act_block.pos_w

    let mut cleared_lines: Vec<i32> = Vec::with_capacity(act_block.static_block_set.pos_w);

    for y in act_block.grid_pos.y..act_block.grid_pos.y + act_block.static_block_set.pos_w as i32 {
        let mut line_cleared = true;

        'inner: for x in 0..grid.width {
            let pos_to_check = v2i::new(x as i32, y);
            if !grid.block_positions.contains_key(&pos_to_check) {
                line_cleared = false;
                break 'inner;
            }
        }

        if line_cleared {
            cleared_lines.push(y);
        }
    }

    // clearing all full lines and shifting all blocks above by 1 row per cleared line
    for (index, cleared_row) in cleared_lines.iter().enumerate() {
        //clearing line
        let cleared_row = cleared_row - index as i32;

        for x in 0..grid.width {
            let pos_to_remove = v2i::new(x as i32, cleared_row);
            let b_index = grid.block_positions.remove(&pos_to_remove);
            if let Some(b_index) = b_index {
                // removing the actual block
                // u gotta remove the actual block on the arena or disable it or something
                blocks.remove(b_index);
                // let nowhere_pos = Vector2::new(-500., 0.);
                // blocks[b_index].update_target_pos::<false>(nowhere_pos);
            }
        }

        // shifting all the blocks above 1 less
        for y_2 in cleared_row + 1..grid.height as i32 {
            for x in 0..grid.width {
                let mut pos_to_shift = v2i::new(x as i32, y_2);
                if let Some(block_index) = grid.block_positions.remove(&pos_to_shift) {
                    pos_to_shift.y -= 1;
                    grid.block_positions.insert(pos_to_shift, block_index);
                    //update the block position
                    blocks[block_index]
                        .update_target_pos::<true>(grid.get_real_position(pos_to_shift))
                }
            }
        }
    }

    // checking if game over
    // getting higher
    let mut game_over = false;
    for i in grid.block_positions.iter() {
        if i.0.y >= grid.height as i32 {
            game_over = true;
            break;
        }
    }

    if game_over {
        //clearing all grid blocks from arena
        for i in grid.block_positions.iter() {
            blocks.remove(*i.1);
        }

        grid.block_positions.clear();
        println!("game over");
    }
}

impl Block {
    fn new(pos: v2, color: v4) -> Self {
        let tween = Tween {
            start: pos,
            end: pos,
            duration: Duration::from_millis(BLOCKS_TWEEN_DURATION),
            ease_func: EaseFunction::CircOut,
            t: 0.,
        };

        Self { pos, color, tween }
    }

    fn update_target_pos<const INTERPOLATE: bool>(&mut self, new_pos: v2) {
        let (start, end) = if INTERPOLATE {
            (self.pos, new_pos)
        } else {
            (new_pos, new_pos)
        };

        if !INTERPOLATE {
            self.pos = new_pos;
        }

        self.tween = tween_new_pos(&self.tween, start, end);
    }
}

impl Game {
    pub fn new(cam_initial_size: Vector2<u32>) -> Self {
        let mut arena = Arena::new();

        let grid_pos = Vector2::new(
            BLOCK_SIZE * 6.0 + 10.,
            cam_initial_size.y as f32 - BLOCK_SIZE * 2. - 10.,
        );
        let grid_positions = HashMap::new();

        let grid = Grid {
            block_positions: grid_positions,
            pos: grid_pos,
            width: GRID_WIDTH,
            height: GRID_HEIGHT,
        };

        //making background
        for x in -1i32..=grid.width as i32 {
            for y in -1i32..grid.height as i32 {
                let pos = v2i::new(x, y);
                let real_pos = grid.get_real_position(pos);

                let color = if x >= 0 && x < grid.width as i32 && y >= 0 && y < grid.height as i32 {
                    color::GRID_BG
                } else {
                    color::HUD_BG
                };

                arena.insert(Block::new(real_pos, color));
            }
        }

        //do the random next list

        // use strum::IntoEnumIterator;
        // self.swap_active_block_set(BlockSetType::iter().get(rand_index).unwrap());

        let mut rng = rand::thread_rng();

        let mut next_block_types: Vec<BlockSetType> = Vec::with_capacity(BLOCK_COUNT * 2);

        for _ in 0..2 {
            next_block_types.push(BlockSetType::T);
            next_block_types.push(BlockSetType::Square);
            next_block_types.push(BlockSetType::Line);
            next_block_types.push(BlockSetType::L);
            next_block_types.push(BlockSetType::J);
            next_block_types.push(BlockSetType::S);
            next_block_types.push(BlockSetType::Z);
        }

        next_block_types.as_mut_slice().shuffle(&mut rng);

        let mut next_blocks: VecDeque<StaticBlockSet> =
            VecDeque::with_capacity(NEXT_BLOCKS_COUNT + 1);

        for i in 0..NEXT_BLOCKS_COUNT + 1 {
            next_blocks.push_back(match next_block_types[i] {
                BlockSetType::T => StaticBlockSet::new_t(&mut arena),
                BlockSetType::Square => StaticBlockSet::new_square(&mut arena),
                BlockSetType::Line => StaticBlockSet::new_line(&mut arena),
                BlockSetType::L => StaticBlockSet::new_l(&mut arena),
                BlockSetType::J => StaticBlockSet::new_j(&mut arena),
                BlockSetType::S => StaticBlockSet::new_s(&mut arena),
                BlockSetType::Z => StaticBlockSet::new_z(&mut arena),
            });
        }

        // positioning the next blocks

        for (i, block_set) in next_blocks.iter_mut().enumerate() {
            let mut block_pos = v2::new(
                grid.pos.x + (GRID_WIDTH + 2) as f32 * BLOCK_SIZE + 20.0,
                FIRST_BLOCK_POS_Y,
            );

            if i >= NEXT_BLOCKS_COUNT {
                block_pos.x += 1000.0; //has to be past window size
                block_pos.y += BLOCK_SIZE * 4.0 * (NEXT_BLOCKS_COUNT - 1) as f32;
            } else {
                block_pos.y += BLOCK_SIZE * 4.0 * i as f32;
            }

            block_set.update_pos::<false>(block_pos, &mut arena);
        }

        let current_slide: i32 = 1;
        let cam_initial_pos: v2 = v2::new(
            (crate::WINDOW_INNER_WIDTH as f32) * current_slide as f32,
            0.,
        );

        fn pos_slide(slide: i32, pos: v2) -> v2 {
            let mut res = pos;
            res.x += slide as f32 * crate::WINDOW_INNER_WIDTH as f32 * -1.0;
            res
        }

        let end_y = 6;

        let logo_pos_x = crate::WINDOW_INNER_WIDTH as f32 / 2.0 - (BLOCK_SIZE * 23.0) / 2.0;

        // tetris logo test
        let mut draw_blocs_hor = |y, bloc_arr: &[i32]| {
            for i in 0..bloc_arr.len() {
                if bloc_arr[i] == 0 {
                    continue;
                }

                let bloc_pos = pos_slide(
                    current_slide,
                    v2::new(logo_pos_x + BLOCK_SIZE * i as f32, BLOCK_SIZE * y as f32),
                );

                let color = if i <= 3 {
                    color::COLORS[2]
                } else if i <= 7 {
                    color::COLORS[4]
                } else if i <= 11 {
                    color::COLORS[2]
                } else if i <= 15 {
                    color::COLORS[4]
                } else if i <= 19 {
                    color::COLORS[2]
                } else {
                    color::COLORS[4]
                };

                let mut bloc = Block::new(bloc_pos, color);
                bloc.tween = Tween {
                    start: bloc.pos - v2::new(0., 230.0),
                    end: bloc.pos,
                    duration: Duration::from_millis(700),
                    ease_func: EaseFunction::EaseOutElastic,
                    t: 0.0 - ((i as f32) * 0.06) - ((end_y - y) as f32 * 0.06),
                };

                arena.insert(bloc);
            }
        };

        draw_blocs_hor(
            2,
            &[
                1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1,
            ],
        );
        draw_blocs_hor(
            3,
            &[
                0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
            ],
        );
        draw_blocs_hor(
            4,
            &[
                0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            ],
        );
        draw_blocs_hor(
            5,
            &[
                0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1,
            ],
        );
        draw_blocs_hor(
            6,
            &[
                0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
            ],
        );

        let mut game = Self {
            blocks: arena,
            active_block_set: None,
            grid,
            tick_timer: Duration::from_secs(0),
            camera: Camera {
                initial_size: Vector2::new(cam_initial_size.x as f32, cam_initial_size.y as f32),
                position: cam_initial_pos,
                zoom_amount: 1.0,
                tween: Tween {
                    start: cam_initial_pos,
                    end: cam_initial_pos,
                    duration: Duration::from_millis(2000),
                    ease_func: EaseFunction::EaseOutBounce,
                    t: 0.0,
                },
            },
            hold_block_preview: None,
            hold_enabled: true,
            next_blocks,
            next_block_types,
            next_block_index: 0,
            rng,
            current_slide,
            gui: init_gui(),
            game_state: GameState::Menu,
        };

        game.swap_active_block_set_from_next_blocks();

        game
    }

    fn do_block_controls(&mut self, input: &Input) {
        let mut spawn_new_random_block = false;
        let mut new_hold: Option<BlockSetType> = None;

        if let Some(ref mut act_block) = self.active_block_set {
            if input.get_key_down(Keys::A) {
                let new_pos = v2i::new(act_block.grid_pos.x - 1, act_block.grid_pos.y);

                if act_block.does_fit(new_pos, &act_block.static_block_set.positions, &self.grid) {
                    act_block.update_pos::<true>(&self.grid, new_pos, &mut self.blocks);
                }
            } else if input.get_key_down(Keys::D) {
                let new_pos = v2i::new(act_block.grid_pos.x + 1, act_block.grid_pos.y);

                if act_block.does_fit(new_pos, &act_block.static_block_set.positions, &self.grid) {
                    act_block.update_pos::<true>(&self.grid, new_pos, &mut self.blocks);
                }
            }

            //hard drop
            if input.get_key_down(Keys::Space) {
                let mut new_pos = v2i::new(act_block.grid_pos.x, act_block.grid_pos.y);

                while act_block.does_fit(new_pos, &act_block.static_block_set.positions, &self.grid)
                {
                    new_pos.y -= 1;
                }

                new_pos.y += 1;

                act_block.update_pos::<true>(&self.grid, new_pos, &mut self.blocks);

                //changing tween of blocks
                for i in act_block.static_block_set.blocks.iter() {
                    let b = &mut self.blocks[*i];
                    b.tween = Tween {
                        start: b.tween.start,
                        end: b.tween.end,
                        duration: Duration::from_millis(BLOCK_FALL_DURATION),
                        ease_func: EaseFunction::EaseOutBounce,
                        t: 0.
                    };
                }

                put_down_act_block(&mut self.grid, act_block, &mut self.blocks);

                self.hold_enabled = true;
                //spawn new block
                spawn_new_random_block = true;
            }

            //this will be Some if a rotation is tried
            let mut new_positions = None;

            // rotation
            if input.get_key_down(Keys::H) {
                // clockwise
                new_positions = Some(BlockSet::get_rotated_pos::<false>(
                    &act_block.static_block_set.positions,
                    act_block.static_block_set.pos_w,
                ));
            } else if input.get_key_down(Keys::J) {
                // counter-clockwise
                new_positions = Some(BlockSet::get_rotated_pos::<true>(
                    &act_block.static_block_set.positions,
                    act_block.static_block_set.pos_w,
                ));
            } else if input.get_key_down(Keys::K) {
                // 180 rotation
                // yeah i just rotate twice, deal with it
                let mut new_positions_inner = BlockSet::get_rotated_pos::<true>(
                    &act_block.static_block_set.positions,
                    act_block.static_block_set.pos_w,
                );
                new_positions_inner = BlockSet::get_rotated_pos::<true>(
                    &new_positions_inner,
                    act_block.static_block_set.pos_w,
                );

                new_positions = Some(new_positions_inner);
            }

            //this will be Some if a rotation has been attempted
            if let Some(new_rot_positions) = new_positions {
                // resetting down timer for testing
                self.tick_timer = Duration::from_secs(0);

                let (_did_it_fit, _iter_number) =
                    act_block.try_to_fit(new_rot_positions, &self.grid);
                act_block.update_pos::<true>(&self.grid, act_block.grid_pos, &mut self.blocks);
                // println!("Did it fit? {did_it_fit}, iter number: {iter_number}");
            }

            if input.get_key_down(Keys::U) {
                if self.hold_enabled {
                    new_hold = Some(act_block.static_block_set.set_type);
                    // transferring blocks form act block to hold block preview
                    // act_block.static_block_set.transfer_blocks(self.hold_block_type)
                    act_block.remove_ghost_blocks(&mut self.blocks);
                    self.hold_enabled = false;
                }
            }
        }

        if let Some(_new_hold) = new_hold {
            let mut hold_preview = self.active_block_set.take().unwrap().static_block_set;

            // moving hold block to active block set
            if let Some(hold_block) = self.hold_block_preview.take() {
                self.consume_block_set_into_active_set(hold_block);
            } else {
                self.swap_active_block_set_from_next_blocks();
            }

            hold_preview.update_pos::<true>(HOLD_BLOCK_POS, &mut self.blocks);

            //TODO what happens to the old hold?
            self.hold_block_preview = Some(hold_preview);
        } else if spawn_new_random_block {
            self.swap_active_block_set_from_next_blocks();
        }

        if input.get_key_down(Keys::S) {
            self.down_tick();
            self.tick_timer = Duration::from_secs(0);
        }
    }

    fn swap_active_block_set(&mut self, blockset_type: BlockSetType) {
        self.active_block_set = Some(match blockset_type {
            BlockSetType::T => BlockSet::new_t(&self.grid, &mut self.blocks),
            BlockSetType::Square => BlockSet::new_square(&self.grid, &mut self.blocks),
            BlockSetType::Line => BlockSet::new_line(&self.grid, &mut self.blocks),
            BlockSetType::L => BlockSet::new_l(&self.grid, &mut self.blocks),
            BlockSetType::J => BlockSet::new_j(&self.grid, &mut self.blocks),
            BlockSetType::S => BlockSet::new_s(&self.grid, &mut self.blocks),
            BlockSetType::Z => BlockSet::new_z(&self.grid, &mut self.blocks),
        });
    }

    fn consume_block_set_into_active_set(&mut self, block_set: StaticBlockSet) {
        // moving block from the next list to active block set
        let mut blocks_ghost = Vec::new();

        for a in block_set.positions.iter() {
            if !a {
                continue;
            }
            blocks_ghost.push(self.blocks.insert(Block::new(block_set.pos, color::GHOST)));
        }

        self.active_block_set = Some(BlockSet {
            static_block_set: block_set,
            grid_pos: ACTIVE_BLOCK_START_POS,
            blocks_ghost,
            ghost_offset: 0,
        });

        self.active_block_set.as_mut().unwrap().update_pos::<true>(
            &self.grid,
            ACTIVE_BLOCK_START_POS,
            &mut self.blocks,
        );
    }

    fn swap_active_block_set_from_next_blocks(&mut self) {
        let next_block = self.next_blocks.pop_front().unwrap();
        self.consume_block_set_into_active_set(next_block);

        // Creating a new block at the end of the queue
        self.next_blocks
            .push_back(match self.next_block_types[self.next_block_index] {
                BlockSetType::T => StaticBlockSet::new_t(&mut self.blocks),
                BlockSetType::Square => StaticBlockSet::new_square(&mut self.blocks),
                BlockSetType::Line => StaticBlockSet::new_line(&mut self.blocks),
                BlockSetType::L => StaticBlockSet::new_l(&mut self.blocks),
                BlockSetType::J => StaticBlockSet::new_j(&mut self.blocks),
                BlockSetType::S => StaticBlockSet::new_s(&mut self.blocks),
                BlockSetType::Z => StaticBlockSet::new_z(&mut self.blocks),
            });

        self.next_block_index += 1;

        if self.next_block_index >= self.next_blocks.len() {
            self.next_block_index = 0;
            self.next_block_types.as_mut_slice().shuffle(&mut self.rng);
        }

        // Updating position of the queue blocks
        for (i, block_set) in self.next_blocks.iter_mut().enumerate() {
            let mut block_pos = v2::new(
                self.grid.pos.x + (GRID_WIDTH + 2) as f32 * BLOCK_SIZE + 20.0,
                FIRST_BLOCK_POS_Y,
            );

            if i >= NEXT_BLOCKS_COUNT {
                block_pos.x += 1000.0; //has to be past window size
                block_pos.y += BLOCK_SIZE * 4.0 * (NEXT_BLOCKS_COUNT - 1) as f32;
                block_set.update_pos::<false>(block_pos, &mut self.blocks);
            } else {
                block_pos.y += BLOCK_SIZE * 4.0 * i as f32;
                block_set.update_pos::<true>(block_pos, &mut self.blocks);
            }
        }
    }

    fn down_tick(&mut self) {
        let mut spawn_new_block_set = false;

        if let Some(ref mut act_block) = self.active_block_set {
            let new_pos = v2i::new(act_block.grid_pos.x, act_block.grid_pos.y - 1);

            if act_block.does_fit(new_pos, &act_block.static_block_set.positions, &self.grid) {
                //updating block positions too
                act_block.update_pos::<true>(&self.grid, new_pos, &mut self.blocks);
            } else {
                put_down_act_block(&mut self.grid, act_block, &mut self.blocks);
                self.hold_enabled = true;
                spawn_new_block_set = true;
            }
        }

        if spawn_new_block_set {
            self.swap_active_block_set_from_next_blocks();
        }
    }
}

fn draw_gui(game: &mut Game, _input: &Input) {
    gui_bind_layout(
        &mut game.gui,
        Layout::VerticalCentered(
            v2::new(200.0, 80.0),
            v2::new(-(crate::WINDOW_INNER_WIDTH as f32), 0.0),
        ),
    );

    if do_button(&mut game.gui, "Play".into()) {
        println!("pressed play");
        game_restart(game);
        go_to_slide(game, 0);
    }
    if do_button(&mut game.gui, "Controls".into()) {
        println!("pressed controls");
    }
    if do_button(&mut game.gui, "Quit".into()) {
        println!("pressed quit");
    }
}

fn go_to_slide(game: &mut Game, slide: i32) {
    game.camera.tween = tween_new_pos(
        &game.camera.tween,
        game.camera.position,
        v2::new((crate::WINDOW_INNER_WIDTH as f32) * (slide) as f32, 0.),
    );
    game.current_slide = slide;
}

fn game_restart(game: &mut Game) {
    // @TODO(lucypero): reset score i guess, when u have it
    game.tick_timer = Duration::from_secs(0);
    game.game_state = GameState::Playing;

    //clearing all grid blocks
    for i in game.grid.block_positions.iter() {
        game.blocks.remove(*i.1);
    }
    game.grid.block_positions.clear();

    //clearing all blocks from active set
    if let Some(ab) = &game.active_block_set {
        for i in ab.blocks_ghost.iter() {
            game.blocks.remove(*i);
        }

        for i in ab.static_block_set.blocks.iter() {
            game.blocks.remove(*i);
        }
    };

    game.swap_active_block_set_from_next_blocks();

    if let Some(hold) = &mut game.hold_block_preview {
        for i in hold.blocks.iter() {
            game.blocks.remove(*i);
        }
    }

    game.hold_block_preview = None;
    game.hold_enabled = true;
}

pub fn game_update(game: &mut Game, input: &Input, dt: Duration) {
    //animating camera
    {
        game.camera.tween.t +=
            dt.as_micros() as f32 / game.camera.tween.duration.as_micros() as f32;
        if game.camera.tween.t > 1.0 {
            game.camera.tween.t = 1.0;
        }

        game.camera.position = tween_get_value(&game.camera.tween);
    }


    // updating all block positions
    for (_, b) in game.blocks.iter_mut() {
        b.tween.t += dt.as_micros() as f32 / b.tween.duration.as_micros() as f32;

        if b.tween.t > 1.0 {
            b.tween.t = 1.0;
        }

        b.pos = tween_get_value(&b.tween);
    }

    // moving between slides
    {
        if input.get_key_down(Keys::Up) {
            game.camera.tween = tween_new_pos(
                &game.camera.tween,
                game.camera.position,
                v2::new(
                    (crate::WINDOW_INNER_WIDTH as f32) * (game.current_slide - 1) as f32,
                    0.,
                ),
            );

            game.current_slide -= 1;
        }

        if input.get_key_down(Keys::Down) {
            game.camera.tween = tween_new_pos(
                &game.camera.tween,
                game.camera.position,
                v2::new(
                    (crate::WINDOW_INNER_WIDTH as f32) * (game.current_slide + 1) as f32,
                    0.,
                ),
            );

            game.current_slide += 1;
        }
    }

    //game state-specific stuff/controls
    match game.game_state {
        GameState::Menu => {}
        GameState::Playing => {

            // tick timer (block drop)
            game.tick_timer += dt;

            if game.tick_timer.as_millis() >= 400 {
                //perform tick
                game.down_tick();
                game.tick_timer -= Duration::from_millis(400);
            }


            game.do_block_controls(&input);
            // do controls
        }
        GameState::Pause => {}
    }

    //debug stuff

    //pause game
    if input.get_key_down(Keys::NumpadAdd) {
        if game.game_state == GameState::Playing {
            game.game_state = GameState::Pause;
        } else if game.game_state == GameState::Pause {
            game.game_state = GameState::Playing;
        }
    }

    //reset game
    if input.get_key_down(Keys::NumpadSubtract) {
        game_restart(game);
    }



    // gui stuff

    gui_frame_start(
        &mut game.gui,
        input.mouse_x,
        input.mouse_y,
        game.camera.position,
        input.get_key(Keys::Mouse1),
        dt,
    );

    draw_text_pos(
        &mut game.gui,
        format!("ft: {}us", dt.as_micros()).into(),
        v2::new(2.0, 20.0),
    );

    draw_gui(game, input);

    gui_frame_end(&mut game.gui);
}
