use std::collections::HashMap;
use cgmath::{Vector2, Vector4};
use std::time::Duration;
use rand::{Rng};

use crate::input::{Input, Keys};

// (0,0) is bottom left
type Pos = Vector2<i32>;
type Vec2 = Vector2<f32>;
type Vec4 = Vector4<f32>;

pub const BLOCK_SIZE: f32 = 30.0;
pub const GRID_WIDTH: usize = 10;
pub const GRID_HEIGHT: usize = 20;

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
    pub color: Vec4,
}

impl Block {
    fn new() -> Self {
        Self {
            color: color::PINK,
        }
    }
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
    fn try_to_fit(&mut self, new_positions: Vec<bool>, grid: &Grid) -> (bool, i32) {
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
                println!("testing pos is {new_testing_pos:?}");
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.pos = new_testing_pos;
                    self.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for x_2 in (-iter_number + 1..=iter_number).rev() {
                new_testing_pos = Pos::new(self.pos.x + x_2, self.pos.y + iter_number);
                println!("testing pos is {new_testing_pos:?}");
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.pos = new_testing_pos;
                    self.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for y_2 in (-iter_number + 1..=iter_number).rev() {
                new_testing_pos = Pos::new(self.pos.x - iter_number, self.pos.y + y_2);
                println!("testing pos is {new_testing_pos:?}");
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.pos = new_testing_pos;
                    self.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for x_2 in (-iter_number..iter_number) {
                new_testing_pos = Pos::new(self.pos.x + x_2, self.pos.y - iter_number);
                println!("testing pos is {new_testing_pos:?}");
                if self.does_fit(new_testing_pos, &new_positions, grid) {
                    self.pos = new_testing_pos;
                    self.positions = new_positions;
                    return (true, iter_number);
                }
            }
            for y_2 in (-1..=iter_number).rev() {
                new_testing_pos = Pos::new(self.pos.x + iter_number, self.pos.y + y_2);
                println!("testing pos is {new_testing_pos:?}");
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

            if grid.blocks.contains_key(&pos_to_test) ||
                !grid.is_inside(pos_to_test) {
                return false;
            }
        }
        true
    }

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
            false, false, false, false,
            false, false, false, false,
            true, true, true, true,
            false, false, false, false,
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
}

pub struct Grid {
    pub blocks: HashMap<Pos, Block>,
    pub pos: Vec2,
    // the bottom left position of the grid.
    pub width: usize,
    pub height: usize,
}

impl Grid {
    fn from_test_grid(pos: Vec2, positions: Vec<Pos>) -> Self {
        let mut blocks = HashMap::new();

        for i in positions {
            blocks.insert(i, Block {
                color: color::PINK,
            });
        }

        Self {
            width: GRID_WIDTH,
            height: GRID_HEIGHT,
            pos,
            blocks,
        }
    }

    fn new(pos: Vec2) -> Self {
        Self {
            width: GRID_WIDTH,
            height: GRID_HEIGHT,
            pos,
            blocks: HashMap::new(),
        }
    }

    // we do not check if above
    fn is_inside(&self, pos: Pos) -> bool {
        pos.x < self.width as i32 &&
            pos.x >= 0 &&
            pos.y >= 0
    }
}

pub struct Game {
    pub active_block_set: Option<BlockSet>,
    pub grid: Grid,
    tick_timer: Duration,
    pub camera: Camera,
}

fn put_down_act_block(grid: &mut Grid, act_block: &BlockSet) {
    for (index, is_occupied) in act_block.positions.iter().enumerate() {
        if !is_occupied {
            continue;
        }

        let x = index % act_block.pos_w;
        let y = index / act_block.pos_w;
        let the_pos = Pos::new(act_block.pos.x + x as i32, act_block.pos.y + y as i32);

        grid.blocks.insert(the_pos, Block {
            color: act_block.color
        });
    }

    // checking if u cleared lines
    // check act_block.pos.y to act_block.pos.y + act_block.pos_w

    let mut cleared_lines: Vec<i32> = Vec::with_capacity(act_block.pos_w);

    for y in act_block.pos.y..act_block.pos.y + act_block.pos_w as i32 {
        let mut line_cleared = true;

        'inner: for x in 0..grid.width {
            let pos_to_check = Pos::new(x as i32, y);
            if !grid.blocks.contains_key(&pos_to_check) {
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
            grid.blocks.remove(&pos_to_remove);
        }

        // shifting all the blocks above 1 less
        for y_2 in cleared_row + 1..grid.height as i32 {
            for x in 0..grid.width {
                let mut pos_to_shift = Pos::new(x as i32, y_2);
                if let Some(block) = grid.blocks.remove(&pos_to_shift) {
                    pos_to_shift.y -= 1;
                    grid.blocks.insert(pos_to_shift, block);
                }
            }
        }
    }
}

impl Game {
    pub fn new(cam_initial_size: Vector2<u32>) -> Self {
        let mut test_grid: Vec<Pos> = Vec::new();

        assert!(GRID_WIDTH > 8);
        assert!(GRID_HEIGHT > 15);
        for y in 0..GRID_HEIGHT - 5 {
            for x in 0..GRID_WIDTH {
                let pos = Pos::new(x as i32, y as i32);

                if pos.x != 3 &&
                    pos.x != 4 &&
                    pos != Pos::new(5, 0) &&
                    pos != Pos::new(5, 1) &&
                    pos != Pos::new(5, 2) &&
                    pos != Pos::new(6, 0) &&
                    pos != Pos::new(6, 1) &&
                    pos != Pos::new(6, 2) &&
                    pos != Pos::new(7, 0) &&
                    pos != Pos::new(7, 1) &&
                    pos != Pos::new(7, 2)
                {
                    test_grid.push(pos);
                }
            }
        }

        Self {
            grid: Grid::from_test_grid(
                Vector2::new(10., cam_initial_size.y as f32 - BLOCK_SIZE - 10.),
                test_grid,
            ),
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
        let mut spawn_new_block_set = false;

        if let Some(ref mut act_block) = self.active_block_set {
            if input.get_key_down(Keys::A) {
                let new_pos = Pos::new(act_block.pos.x - 1, act_block.pos.y);

                if act_block.does_fit(new_pos, &act_block.positions, &self.grid) {
                    act_block.pos = new_pos;
                }
            } else if input.get_key_down(Keys::D) {
                let new_pos = Pos::new(act_block.pos.x + 1, act_block.pos.y);

                if act_block.does_fit(new_pos, &act_block.positions, &self.grid) {
                    act_block.pos = new_pos;
                }
            }


            //hard drop
            if input.get_key_down(Keys::Space) {
                let mut new_pos = Pos::new(act_block.pos.x, act_block.pos.y);

                while act_block.does_fit(new_pos, &act_block.positions, &self.grid) {
                    new_pos.y -= 1;
                }

                new_pos.y += 1;

                act_block.pos = new_pos;

                put_down_act_block(&mut self.grid, act_block);
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
                //old code
                // if act_block.does_fit(act_block.pos, &new_rot_positions, &self.grid) {
                //     act_block.positions = new_rot_positions;
                // }
                let (did_it_fit, iter_number) = act_block.try_to_fit(new_rot_positions, &self.grid);
                println!("Did it fit? {did_it_fit}, iter number: {iter_number}");
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

    fn down_tick(&mut self) {
        let mut spawn_new_block_set = false;

        if let Some(ref mut act_block) = self.active_block_set {
            let new_pos = Pos::new(act_block.pos.x, act_block.pos.y - 1);

            if act_block.does_fit(new_pos, &act_block.positions, &self.grid) {
                act_block.pos = new_pos;
            } else {
                put_down_act_block(&mut self.grid, act_block);
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
    }
}
