use std::collections::HashMap;
use cgmath::{Vector2, Vector4};
use std::time::Duration;
use rand::{Rng};
use thunderdome::{Arena, Index};
use crate::input::{Input, Keys};
use tween::{CircIn, CircInOut, CircOut, SineInOut, SineOut, Tweener};

// (0,0) is bottom left (of grid)
type Pos = Vector2<i32>;
type Vec2 = Vector2<f32>;
type Vec4 = Vector4<f32>;
type TweenUsed = CircOut<f32, i64>;
type MyTween = Tweener<TweenUsed>;

pub const BLOCK_SIZE: f32 = 32.0;
pub const GRID_WIDTH: usize = 10;
pub const GRID_HEIGHT: usize = 20;

pub const INITIAL_ACTIVE_BLOCK_POS: Vec2 = Vec2::new(30.0, 40.0);

pub struct Game {
    pub blocks: Arena<Block>,
    pub active_block_set: Option<BlockSet>,
    pub grid: Grid,
    tick_timer: Duration,
    pub camera: Camera,
}

pub struct BlockSet {
    pub positions: Vec<bool>,
    pub blocks: Vec<Index>,
    pub pos_w: usize,
    pub pos: Pos,
    //pos of bottom left block
    // ghost
    pub blocks_ghost: Vec<Index>,
    pub ghost_offset: i32, //y offset from grid origin
}

pub struct Block {
    pub pos: Vec2,
    pub color: Vec4,
    pub tweener_x: MyTween,
    pub tweener_y: MyTween,
}

pub struct Camera {
    pub initial_size: Vector2<f32>,
    pub position: Vector2<f32>,
    pub zoom_amount: f32,
}

pub struct Grid {
    pub block_positions: HashMap<Pos, Index>,
    pub pos: Vec2,
    // the bottom left position of the grid.
    pub width: usize,
    pub height: usize,
}

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

    pub const GHOST: Vec4 = Vec4::new(1., 1., 1., 0.3);
    pub const GRID_BG: Vec4 = Vec4::new(1., 1., 1., 0.1);
    pub const HUD_BG: Vec4 = Vec4::new(0., 0., 0., 1.);
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
    fn try_to_fit(&mut self, new_positions: Vec<bool>, grid: &Grid, blocks: &mut Arena<Block>) -> (bool, i32) {
        const MAX_ITERATIONS: i32 = 5;

        let mut new_testing_pos = self.pos;

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

        if self.does_fit(self.pos, &new_positions, grid) {
            self.positions = new_positions;
            return (true, 0);
        }

        for iter_number in 1..=MAX_ITERATIONS {
            for y_2 in 0..iter_number {
                new_testing_pos = Pos::new(self.pos.x + iter_number, self.pos.y + y_2);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.pos = new_testing_pos;
                    self.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for x_2 in (-iter_number + 1..=iter_number).rev() {
                new_testing_pos = Pos::new(self.pos.x + x_2, self.pos.y + iter_number);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.pos = new_testing_pos;
                    self.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for y_2 in (-iter_number + 1..=iter_number).rev() {
                new_testing_pos = Pos::new(self.pos.x - iter_number, self.pos.y + y_2);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.pos = new_testing_pos;
                    self.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for x_2 in (-iter_number..iter_number) {
                new_testing_pos = Pos::new(self.pos.x + x_2, self.pos.y - iter_number);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.pos = new_testing_pos;
                    self.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for y_2 in (-1..=iter_number).rev() {
                new_testing_pos = Pos::new(self.pos.x + iter_number, self.pos.y + y_2);
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.pos = new_testing_pos;
                    self.positions = new_positions;
                    return (true, iter_number);
                }
            }
        }


        (false, 0)
    }


    // does it fit in an alternate global position or alternate position values?
    fn does_fit(&self, new_pos: Pos, new_positions: &Vec<bool>, grid: &Grid) -> bool {
        for (index, is_occupied) in new_positions.iter().enumerate() {
            if !is_occupied {
                continue;
            }

            let x = index % self.pos_w;
            let y = index / self.pos_w;
            let pos_to_test = Pos::new(new_pos.x + x as i32, new_pos.y + y as i32);

            if grid.block_positions.contains_key(&pos_to_test) ||
                !grid.is_inside(pos_to_test) {
                return false;
            }
        }
        true
    }

    fn from_pos(grid: &Grid, pos: Pos, pos_w: usize, positions: Vec<bool>,
                arena: &mut Arena<Block>, color: Vec4) -> Self {
        let mut blocks = Vec::new();
        let mut blocks_ghost = Vec::new();

        // creating the new blocks that we need
        for (index, a) in positions.iter().enumerate() {
            if !a {
                continue;
            }

            blocks.push(arena.insert(Block::new(Vec2::new(0.0, 0.0), color)));
            blocks_ghost.push(arena.insert(Block::new(Vec2::new(0.0, 0.0), color::GHOST)));
        }

        let mut act_block = Self {
            positions,
            blocks,
            pos,
            pos_w,
            blocks_ghost,
            ghost_offset: 0,
        };


        act_block.update_pos::<false>(grid, pos, arena);

        act_block
    }

    fn new_t(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, true, false, true, false];
        let pos_w = 3;
        let pos = Pos::new(5, 20);
        let color = color::COLORS[0];

        Self::from_pos(grid, pos, pos_w, positions, arena, color)
    }

    fn new_square(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![true, true, true, true];
        let pos_w = 2;
        let pos = Pos::new(5, 20);
        let color = color::COLORS[1];

        Self::from_pos(grid, pos, pos_w, positions, arena, color)
    }

    fn new_line(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![
            false, false, false, false,
            false, false, false, false,
            true, true, true, true,
            false, false, false, false,
        ];

        let pos_w = 4;
        let pos = Pos::new(5, 20);
        let color = color::COLORS[2];

        Self::from_pos(grid, pos, pos_w, positions, arena, color)
    }

    fn new_l(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, true, false, false, true];
        let pos_w = 3;
        let pos = Pos::new(5, 20);
        let color = color::COLORS[3];

        Self::from_pos(grid, pos, pos_w, positions, arena, color)
    }

    fn new_j(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, true, true, false, false];
        let pos_w = 3;
        let pos = Pos::new(5, 20);
        let color = color::COLORS[4];

        Self::from_pos(grid, pos, pos_w, positions, arena, color)
    }

    fn new_s(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, false, true, true, true, true, false];
        let pos_w = 3;
        let pos = Pos::new(5, 20);
        let color = color::COLORS[5];

        Self::from_pos(grid, pos, pos_w, positions, arena, color)
    }

    fn new_z(grid: &Grid, arena: &mut Arena<Block>) -> Self {
        let positions = vec![false, false, false, true, true, false, false, true, true];
        let pos_w = 3;
        let pos = Pos::new(5, 20);
        let color = color::COLORS[6];

        Self::from_pos(grid, pos, pos_w, positions, arena, color)
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

    fn update_pos<const interpolate: bool>(&mut self, grid: &Grid, new_pos: Pos, blocks: &mut Arena<Block>) {
        let mut pos: Vec2 = grid.get_real_position(new_pos);
        self.pos = new_pos;

        let mut block_index = 0;

        // getting ghost block position
        let mut ghost_pos = Pos::new(new_pos.x, new_pos.y);

        while self.does_fit(ghost_pos, &self.positions, grid) {
            ghost_pos.y -= 1;
        }

        ghost_pos.y += 1;
        let ghost_pos: Vec2 = grid.get_real_position(ghost_pos);

        for (index, a) in self.positions.iter().enumerate() {
            if !a {
                // u gotta put this away i think
                continue;
            }

            let mut pos_x = index % self.pos_w;
            let mut pos_y = index / self.pos_w;

            let pos_f32 = Vec2::new(pos.x + BLOCK_SIZE * pos_x as f32,
                                    pos.y - BLOCK_SIZE * pos_y as f32);

            let ghost_pos_f32 = Vec2::new(pos.x + BLOCK_SIZE * pos_x as f32,
                                          ghost_pos.y - BLOCK_SIZE * pos_y as f32);

            blocks[self.blocks[block_index]].update_target_pos::<interpolate>(pos_f32);
            blocks[self.blocks_ghost[block_index]].update_target_pos::<interpolate>(ghost_pos_f32);
            block_index += 1;
        }
    }
}


impl Grid {
    fn new(pos: Vec2) -> Self {
        Self {
            block_positions: HashMap::new(),
            pos,
            width: GRID_WIDTH,
            height: GRID_HEIGHT,
        }
    }

    // we do not check if above
    fn is_inside(&self, pos: Pos) -> bool {
        pos.x < self.width as i32 &&
            pos.x >= 0 &&
            pos.y >= 0
    }

    fn get_real_position(&self, pos: Pos) -> Vec2 {
        Vec2::new(self.pos.x + pos.x as f32 * BLOCK_SIZE, self.pos.y - pos.y as f32 * BLOCK_SIZE)
    }
}

fn put_down_act_block(grid: &mut Grid, act_block: &BlockSet, blocks: &mut Arena<Block>) {
    {
        let mut blocks_i = 0;
        for (index, is_occupied) in act_block.positions.iter().enumerate() {
            if !is_occupied {
                continue;
            }

            let x = index % act_block.pos_w;
            let y = index / act_block.pos_w;
            let the_pos = Pos::new(act_block.pos.x + x as i32, act_block.pos.y + y as i32);

            let block_index = act_block.blocks[blocks_i];

            grid.block_positions.insert(the_pos, block_index);
            //removing ghost blocks
            blocks.remove(act_block.blocks_ghost[blocks_i]);
            blocks[block_index].update_target_pos::<true>(grid.get_real_position(the_pos));
            blocks_i += 1;
        }
    }


    // checking if u cleared lines
    // check act_block.pos.y to act_block.pos.y + act_block.pos_w

    let mut cleared_lines: Vec<i32> = Vec::with_capacity(act_block.pos_w);

    for y in act_block.pos.y..act_block.pos.y + act_block.pos_w as i32 {
        let mut line_cleared = true;

        'inner: for x in 0..grid.width {
            let pos_to_check = Pos::new(x as i32, y);
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
            let pos_to_remove = Pos::new(x as i32, cleared_row);
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
                let mut pos_to_shift = Pos::new(x as i32, y_2);
                if let Some(block_index) = grid.block_positions.remove(&pos_to_shift) {
                    pos_to_shift.y -= 1;
                    grid.block_positions.insert(pos_to_shift, block_index);
                    //update the block position
                    blocks[block_index].update_target_pos::<true>(grid.get_real_position(pos_to_shift))
                }
            }
        }
    }
}

impl Block {
    fn new(pos: Vec2, color: Vec4) -> Self {
        let range = pos.x..=pos.x;
        let duration = 1000;

        let tween = TweenUsed::new(range, duration);
        let tweener_x = Tweener::new(tween);

        let range = pos.y..=pos.y;
        let duration = 1000;

        let tween = TweenUsed::new(range, duration);
        let tweener_y = Tweener::new(tween);

        Self {
            pos,
            color,
            tweener_x,
            tweener_y,
        }
    }

    fn update_target_pos<const interpolate: bool>(&mut self, new_pos: Vec2) {

        let range = if interpolate {
            self.pos.x..=new_pos.x
        } else {
            new_pos.x..=new_pos.x
        };

        let duration = 300;

        let tween = TweenUsed::new(range, duration);
        self.tweener_x = Tweener::new(tween);

        let range = if interpolate {
            self.pos.y..=new_pos.y
        } else {
            new_pos.y..=new_pos.y
        };

        let tween = TweenUsed::new(range, duration);
        self.tweener_y = Tweener::new(tween);
    }
}

impl Game {
    pub fn new(cam_initial_size: Vector2<u32>) -> Self {
        let mut test_grid: Vec<Pos> = Vec::new();

        let mut arena = Arena::new();
        let grid_pos = Vector2::new(BLOCK_SIZE + 10., cam_initial_size.y as f32 - BLOCK_SIZE * 2. - 10.);
        let mut grid_positions = HashMap::new();

        let grid = Grid {
            block_positions: grid_positions,
            pos: grid_pos,
            width: GRID_WIDTH,
            height: GRID_HEIGHT,
        };

        let active_block_set = Some(BlockSet::new_line(&grid, &mut arena));

        //making background
        for x in -1i32..=grid.width as i32 {
            for y in -1i32..=grid.height as i32 {
                let pos = Pos::new(x,y);
                let real_pos = grid.get_real_position(pos);

                let color = if x >= 0 && x < grid.width as i32 &&
                            y >= 0 && y < grid.height as i32 {
                    color::GRID_BG
                } else {
                    color::HUD_BG
                };

                arena.insert(Block::new(real_pos, color));
            }
        }

        Self {
            blocks: arena,
            active_block_set,
            grid,
            tick_timer: Duration::from_secs(0),
            camera: Camera {
                initial_size: Vector2::new(cam_initial_size.x as f32, cam_initial_size.y as f32),
                position: Vector2::new(0., 0.),
                zoom_amount: 1.0,
            },
        }
    }

    fn do_block_controls(&mut self, input: &Input) {
        let mut spawn_new_block_set = false;

        if let Some(ref mut act_block) = self.active_block_set {
            if input.get_key_down(Keys::A) {
                let new_pos = Pos::new(act_block.pos.x - 1, act_block.pos.y);

                if act_block.does_fit(new_pos, &act_block.positions, &self.grid) {
                    act_block.update_pos::<true>(&self.grid, new_pos, &mut self.blocks);
                }
            } else if input.get_key_down(Keys::D) {
                let new_pos = Pos::new(act_block.pos.x + 1, act_block.pos.y);

                if act_block.does_fit(new_pos, &act_block.positions, &self.grid) {
                    act_block.update_pos::<true>(&self.grid, new_pos, &mut self.blocks);
                }
            }

            //hard drop
            if input.get_key_down(Keys::Space) {
                let mut new_pos = Pos::new(act_block.pos.x, act_block.pos.y);

                while act_block.does_fit(new_pos, &act_block.positions, &self.grid) {
                    new_pos.y -= 1;
                }

                new_pos.y += 1;

                act_block.update_pos::<true>(&self.grid, new_pos, &mut self.blocks);

                put_down_act_block(&mut self.grid, act_block, &mut self.blocks);
                //spawn new block
                spawn_new_block_set = true;
            }

            //this will be Some if a rotation is tried
            let mut new_positions = None;

            // rotation
            if input.get_key_down(Keys::H) { // clockwise
                new_positions = Some(BlockSet::get_rotated_pos::<false>(&act_block.positions, act_block.pos_w));
            } else if input.get_key_down(Keys::J) { // counter-clockwise
                new_positions = Some(BlockSet::get_rotated_pos::<true>(&act_block.positions, act_block.pos_w));
            } else if input.get_key_down(Keys::K) { // 180 rotation
                // yeah i just rotate twice, deal with it
                let mut new_positions_inner = BlockSet::get_rotated_pos::<true>(&act_block.positions, act_block.pos_w);
                new_positions_inner = BlockSet::get_rotated_pos::<true>(&new_positions_inner, act_block.pos_w);

                new_positions = Some(new_positions_inner);
            }

            //this will be Some if a rotation has been attempted
            if let Some(new_rot_positions) = new_positions {

                // resetting down timer for testing
                self.tick_timer = Duration::from_secs(0);

                let (did_it_fit, iter_number) =
                    act_block.try_to_fit(new_rot_positions, &self.grid, &mut self.blocks);
                act_block.update_pos::<true>(&self.grid, act_block.pos, &mut self.blocks);
                // println!("Did it fit? {did_it_fit}, iter number: {iter_number}");
            }
        }

        if spawn_new_block_set {
            self.swap_active_block_set();
        }

        if input.get_key_down(Keys::S) {
            self.down_tick();
            self.tick_timer = Duration::from_secs(0);
        }
    }

    fn swap_active_block_set(&mut self) {
        let mut r = rand::thread_rng();
        let block_type: usize = r.gen_range(0..6);
        match block_type {
            0 => self.active_block_set = Some(BlockSet::new_t(&self.grid, &mut self.blocks)),
            1 => self.active_block_set = Some(BlockSet::new_square(&self.grid, &mut self.blocks)),
            2 => self.active_block_set = Some(BlockSet::new_line(&self.grid, &mut self.blocks)),
            3 => self.active_block_set = Some(BlockSet::new_l(&self.grid, &mut self.blocks)),
            4 => self.active_block_set = Some(BlockSet::new_j(&self.grid, &mut self.blocks)),
            5 => self.active_block_set = Some(BlockSet::new_s(&self.grid, &mut self.blocks)),
            _ => self.active_block_set = Some(BlockSet::new_z(&self.grid, &mut self.blocks)),
        }
    }

    fn down_tick(&mut self) {
        let mut spawn_new_block_set = false;

        if let Some(ref mut act_block) = self.active_block_set {
            let new_pos = Pos::new(act_block.pos.x, act_block.pos.y - 1);

            if act_block.does_fit(new_pos, &act_block.positions, &self.grid) {
                //updating block positions too
                act_block.update_pos::<true>(&self.grid, new_pos, &mut self.blocks);
            } else {
                put_down_act_block(&mut self.grid, act_block, &mut self.blocks);
                spawn_new_block_set = true;
            }
        }

        if spawn_new_block_set {
            self.swap_active_block_set();
        }
    }

    pub fn update(&mut self, input: &Input, dt: Duration) {
        self.tick_timer += dt;

        if self.tick_timer.as_millis() >= 400 {
            //perform tick
            self.down_tick();
            self.tick_timer -= Duration::from_millis(400);
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

        // updating all block positions
        // TODO: do a lerp here
        for (_, b) in self.blocks.iter_mut() {
            if let Some(val) = b.tweener_x.update(dt.as_millis() as i64) {
                b.pos.x = val;
            }
            if let Some(val) = b.tweener_y.update(dt.as_millis() as i64) {
                b.pos.y = val;
            }
        }
    }
}
