use std::{collections::HashMap, time::Duration};

use libs::cgmath::Zero;

use crate::game::*;

const WIDGETS_MAX: usize = 50;
const WIDGETS_MARGIN: f32 = 20.0;
const LINE_HEIGHT: f32 = 20.0;

type Key = String;

// immediate mode gui
pub struct Gui {
    mouse_x: f32,
    mouse_y: f32,
    mouse_down: bool,
    widgets_prev: HashMap<Key, Widget>,
    pub widgets: Vec<Widget>,

    bound_layout: Layout,
    cur_layout_widgets: Vec<usize>,

    // active widget is the last widget that the mouse clicked on
    //   (mouse could be anywhere else while still holding the mouse button)
    active: Option<Key>,

    // hot widget is the one the mouse is on
    hot: Option<Key>,

    was_hot: bool,

    pub hot_tween: Tween
}

pub enum Layout {
    NoLayout,
    VerticalCentered(v2, v2), //arg: size of buttons, offset
    ///arg: pos of first widget and margin betw widgets.
    /// v2 is pos of first widget, next are positioned below
    Vertical(v2, f32), 
}

#[derive(Clone)]
pub enum WidgetType {
    Button,
    Label,
}

#[derive(Clone)]
pub struct Widget {
    pub widget_type: WidgetType,
    pub label: String,
    pub computed: bool,
    pub computed_position: v2,
    pub computed_size: v2,
}

// Check whether current mouse position is within a rectangle
fn regionhit(gui: &Gui, x: f32, y: f32, w: f32, h: f32) -> bool {
    if gui.mouse_x < x || gui.mouse_y < y || gui.mouse_x >= x + w || gui.mouse_y >= y + h {
        return false;
    }

    return true;
}

pub fn gui_frame_start(gui: &mut Gui, mouse_x: f32, mouse_y: f32, mouse_down: bool, dt: Duration) {

    //advance t

    //updating tweens

    gui.hot_tween.t += dt.as_micros() as f32 / gui.hot_tween.duration.as_micros() as f32;
    if gui.hot_tween.t > 1.0 {
        gui.hot_tween.t = 1.0;
    }

    gui.mouse_x = mouse_x;
    gui.mouse_y = mouse_y;
    gui.mouse_down = mouse_down;

    if !gui.was_hot && gui.hot.is_some() {
        println!("hot started");
        gui.hot_tween.t = 0.0;
        gui.was_hot = true;
    }

    if gui.was_hot && gui.hot.is_none() {
        println!("hot stopped");
        gui.was_hot = false;
    }

    gui.hot = None;

    gui.widgets_prev.clear();

    // copying all widgets to the cache
    for w in gui.widgets.iter() {
        gui.widgets_prev.insert(w.label.clone(), w.clone());
    }

    gui.widgets.clear();
}

pub fn gui_frame_end(gui: &mut Gui) {
    //processing last bound layout
    process_layout(gui);

    if !gui.mouse_down {
        gui.active = None;
    }

}

pub fn is_widget_hot(gui: &Gui, w: &Widget) -> bool {
    //get id and compare to hot id
    let id = &w.label;
    gui.hot == Some(id.clone())
}

pub fn init_gui() -> Gui {
    Gui {
        mouse_x: 0.0,
        mouse_y: 0.0,
        mouse_down: false,
        widgets_prev: HashMap::with_capacity(WIDGETS_MAX),
        widgets: Vec::with_capacity(WIDGETS_MAX),
        active: None,
        hot: None,
        bound_layout: Layout::NoLayout,
        cur_layout_widgets: vec![],
        was_hot : false,
        hot_tween : Tween {
            start: 0.0,
            end: 1.0,
            duration: Duration::from_millis(500),
            ease_func: EaseFunction::CircOut,
            t: 0.0
        }
    }
}

// we finish computing all widgets bound to last layout
fn process_layout(gui: &mut Gui) {
    match gui.bound_layout {
        Layout::NoLayout => {}
        Layout::VerticalCentered(size, offset) => {
            let count = gui.cur_layout_widgets.len();

            // to center horizontally on game window
            let pos_x = crate::WINDOW_INNER_WIDTH as f32 / 2.0 - size.x / 2.0;
            let size_total_y = ((size.y + WIDGETS_MARGIN) * count as f32) - WIDGETS_MARGIN;

            let pos_y = crate::WINDOW_INNER_HEIGHT as f32 / 2.0 - size_total_y / 2.0;
            // adjusting for bottom left corner as origin

            for (i_enum, index) in gui.cur_layout_widgets.iter().enumerate() {
                let w = &mut gui.widgets[*index];
                w.computed_position =
                    v2::new(pos_x, pos_y + ((size.y + WIDGETS_MARGIN) * i_enum as f32)) + offset;
                w.computed_size = size;
                w.computed = true;
            }
        }
        Layout::Vertical(pos, margin) => {

            let mut pos_y = pos.y;
            for index in gui.cur_layout_widgets.iter() {
                let w = &mut gui.widgets[*index];
                // w.computed_position = v2::new(pos.x, pos.y + (w.computed_size.y + margin) * i_enum as f32);
                w.computed_position = v2::new(pos.x, pos_y);
                w.computed = true;

                pos_y += w.computed_size.y + margin;
            }
        }
    }

    gui.cur_layout_widgets.clear();
    gui.bound_layout = Layout::NoLayout;
}

pub fn gui_bind_layout(gui: &mut Gui, layout: Layout) {
    //processing last bound layout
    process_layout(gui);
    gui.bound_layout = layout;
}

pub fn draw_text(gui: &mut Gui, label: String) {
    gui.widgets.push(Widget {
        label: label,
        computed: false,
        computed_position: v2::zero(),
        computed_size: v2::new(0.0,LINE_HEIGHT),
        widget_type: WidgetType::Label,
    });
    gui.cur_layout_widgets.push(gui.widgets.len() - 1);
}

pub fn draw_text_pos(gui: &mut Gui, label: String, pos: v2) {
    gui.widgets.push(Widget {
        label: label,
        computed: true,
        computed_position: pos,
        computed_size: v2::zero(),
        widget_type: WidgetType::Label,
    });

    gui.cur_layout_widgets.push(gui.widgets.len() - 1);
}

pub fn do_button(gui: &mut Gui, label: String) -> bool {
    let id = label;

    // Check whether the button should be hot
    if let Some(w) = gui.widgets_prev.get(&id) {
        if regionhit(gui, w.computed_position.x, w.computed_position.y, w.computed_size.x, w.computed_size.y) {
            gui.hot = Some(id.clone());
            if gui.active.is_none() && gui.mouse_down {
                gui.active = Some(id.clone());
            }
        }     
    }

    gui.widgets.push(Widget {
        label: id.clone(),
        computed: false,
        computed_position: v2::zero(),
        // giving it a default size in the case that the layout does not change it.
        computed_size: v2::new(200.0,50.0),
        widget_type: WidgetType::Button,
    });

    gui.cur_layout_widgets.push(gui.widgets.len() - 1);

    // If button is hot and active, but mouse button is not
    // down, the user must have clicked the button.
    if gui.mouse_down == false && gui.hot == Some(id.clone()) && gui.active == Some(id.clone()) {
        return true;
    }

    // Otherwise, no clicky.
    return false;
}
