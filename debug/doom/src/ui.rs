use crate::game::Game;
use crate::levels::levels;
use crate::settings::{DifficultyPreset, setting_category_name, settings_for_category};
use macroquad::prelude::*;

pub(crate) fn draw_overlay(
    _view_x: f32,
    _view_y: f32,
    _view_w: f32,
    _view_h: f32,
    end_screen: bool,
    victory: bool,
    game: &Game,
) {
    let sw = screen_width();
    let sh = screen_height();
    draw_rectangle(0.0, 0.0, sw, sh, Color::new(0.015, 0.012, 0.01, 0.82));
    let base_w = if end_screen { 620.0 } else { 900.0 };
    let base_h = if end_screen { 290.0 } else { 540.0 };
    let scale = ((sw - 20.0).max(1.0) / base_w)
        .min((sh - 20.0).max(1.0) / base_h)
        .min(1.0);
    let panel_w = (base_w * scale).min(sw - 20.0);
    let panel_h = (base_h * scale).min(sh - 20.0);
    let panel_x = (sw - panel_w) * 0.5;
    let panel_y = (sh - panel_h) * 0.5;
    draw_rectangle(
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        Color::new(0.055, 0.047, 0.041, 0.98),
    );
    draw_rectangle(
        panel_x,
        panel_y,
        panel_w,
        4.0,
        Color::new(0.72, 0.17, 0.10, 1.0),
    );
    draw_rectangle_lines(
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        1.0,
        Color::new(0.43, 0.27, 0.19, 1.0),
    );

    if end_screen {
        let (title, sub) = if victory {
            ("MISSION COMPLETE", "MARS IS QUIET. FOR NOW.")
        } else {
            ("YOU DIED", "THE DEMONS GOT YOU.")
        };
        draw_text(
            title,
            panel_x + 30.0 * scale,
            panel_y + 76.0 * scale,
            (30.0 * scale).max(10.0),
            Color::new(0.91, 0.31, 0.20, 1.0),
        );
        draw_text(
            sub,
            panel_x + 32.0 * scale,
            panel_y + 111.0 * scale,
            (17.0 * scale).max(9.0),
            Color::new(0.83, 0.76, 0.64, 1.0),
        );
        draw_text(
            "PRESS ENTER OR R TO PLAY AGAIN",
            panel_x + 32.0 * scale,
            panel_y + 174.0 * scale,
            (18.0 * scale).max(9.0),
            Color::new(0.95, 0.77, 0.39, 1.0),
        );
        draw_text(
            "CLOSE THE WINDOW TO QUIT",
            panel_x + 32.0 * scale,
            panel_y + 213.0 * scale,
            (15.0 * scale).max(8.0),
            GRAY,
        );
        return;
    }

    draw_text(
        "DOOM: HELL ON MARS",
        panel_x + 28.0 * scale,
        panel_y + 43.0 * scale,
        (30.0 * scale).max(11.0),
        Color::new(0.91, 0.29, 0.16, 1.0),
    );
    draw_text(
        "CONTROLS",
        panel_x + 30.0 * scale,
        panel_y + 80.0 * scale,
        (17.0 * scale).max(8.0),
        Color::new(0.98, 0.73, 0.34, 1.0),
    );
    draw_text(
        "SELECT A LEVEL",
        panel_x + 480.0 * scale,
        panel_y + 80.0 * scale,
        (17.0 * scale).max(8.0),
        Color::new(0.98, 0.73, 0.34, 1.0),
    );

    let rows = [
        ("W / S", "Move forward / back"),
        ("A / D", "Strafe left / right"),
        ("LEFT / RIGHT", "Turn"),
        ("SHIFT", "Run"),
        ("CTRL / SPACE / CLICK", "Fire"),
        ("1-4 / MOUSE WHEEL", "Choose a weapon"),
        ("E", "Activate the exit portal"),
        ("F1 / FN + F1 / ESC", "Pause and open this help"),
        ("SHIFT + P", "Open the settings menu"),
        ("ESC / SPACE / F1", "Resume from help"),
        ("F10", "Quit the game"),
        ("R AFTER END", "Restart after death or win"),
    ];
    let font = (15.0 * scale).max(7.0);
    let row_step = 27.0 * scale;
    let mut y = panel_y + 108.0 * scale;
    for (key, desc) in rows {
        draw_text(
            key,
            panel_x + 32.0 * scale,
            y,
            font,
            Color::new(0.91, 0.73, 0.45, 1.0),
        );
        draw_text(
            desc,
            panel_x + 195.0 * scale,
            y,
            font,
            Color::new(0.80, 0.79, 0.74, 1.0),
        );
        y += row_step;
    }

    let list_start_y = panel_y + 108.0 * scale;
    for (index, level) in levels().iter().enumerate() {
        let level_y = list_start_y + index as f32 * 31.0 * scale;
        let selected = index == game.selected_level;
        if selected {
            draw_rectangle(
                panel_x + 470.0 * scale,
                level_y - 18.0 * scale,
                395.0 * scale,
                25.0 * scale,
                Color::new(0.32, 0.12, 0.08, 0.95),
            );
        }
        let color = if selected {
            Color::new(1.0, 0.82, 0.42, 1.0)
        } else {
            Color::new(0.82, 0.79, 0.73, 1.0)
        };
        draw_text(
            &format!("{:02}. {}  {}", index + 1, level.code, level.title),
            panel_x + 480.0 * scale,
            level_y,
            font,
            color,
        );
    }
    draw_rectangle(
        panel_x + 28.0 * scale,
        panel_y + panel_h - 63.0 * scale,
        panel_w - 56.0 * scale,
        1.0,
        Color::new(0.30, 0.26, 0.22, 1.0),
    );
    draw_text(
        "UP/DOWN SELECT    ENTER: START SELECTED LEVEL",
        panel_x + 30.0 * scale,
        panel_y + panel_h - 40.0 * scale,
        (14.0 * scale).max(8.0),
        Color::new(0.77, 0.73, 0.66, 1.0),
    );
    draw_text(
        "Q: QUICKPLAY FROM LEVEL 1    ESC / SPACE / F1: RESUME",
        panel_x + 30.0 * scale,
        panel_y + panel_h - 17.0 * scale,
        (13.0 * scale).max(8.0),
        Color::new(0.96, 0.77, 0.37, 1.0),
    );
}

pub(crate) fn draw_settings_overlay(game: &Game) {
    let sw = screen_width();
    let sh = screen_height();
    draw_rectangle(0.0, 0.0, sw, sh, Color::new(0.012, 0.01, 0.009, 0.88));
    let scale = ((sw - 24.0).max(1.0) / 780.0)
        .min((sh - 24.0).max(1.0) / 560.0)
        .min(1.0);
    let panel_w = (780.0 * scale).min(sw - 24.0);
    let panel_h = (560.0 * scale).min(sh - 24.0);
    let panel_x = (sw - panel_w) * 0.5;
    let panel_y = (sh - panel_h) * 0.5;
    draw_rectangle(
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        Color::new(0.045, 0.04, 0.036, 0.99),
    );
    draw_rectangle(
        panel_x,
        panel_y,
        panel_w,
        (4.0 * scale).max(2.0),
        Color::new(0.72, 0.17, 0.10, 1.0),
    );
    draw_rectangle_lines(
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        1.0,
        Color::new(0.43, 0.27, 0.19, 1.0),
    );

    let small_font = (15.0 * scale).max(8.0);
    draw_text(
        "SETTINGS",
        panel_x + 30.0 * scale,
        panel_y + 45.0 * scale,
        (28.0 * scale).max(12.0),
        Color::new(0.94, 0.37, 0.23, 1.0),
    );
    draw_text(
        "DIFFICULTY PRESETS",
        panel_x + 30.0 * scale,
        panel_y + 77.0 * scale,
        small_font,
        Color::new(0.77, 0.72, 0.64, 1.0),
    );

    let preset_x = panel_x + 207.0 * scale;
    let preset_y = panel_y + 77.0 * scale;
    for (preset, label, offset) in [
        (DifficultyPreset::Easy, "1 EASY", 0.0),
        (DifficultyPreset::Medium, "2 MEDIUM", 84.0),
        (DifficultyPreset::Hard, "3 HARD", 190.0),
    ] {
        let color = if game.difficulty == preset {
            Color::new(1.0, 0.77, 0.34, 1.0)
        } else {
            Color::new(0.68, 0.67, 0.62, 1.0)
        };
        draw_text(
            label,
            preset_x + offset * scale,
            preset_y,
            small_font,
            color,
        );
    }
    draw_text(
        game.difficulty.label(),
        panel_x + panel_w - 112.0 * scale,
        preset_y,
        small_font,
        Color::new(0.67, 0.8, 0.69, 1.0),
    );

    let category_y = panel_y + 112.0 * scale;
    for category in 0..4 {
        let x = panel_x + (28.0 + category as f32 * 178.0) * scale;
        let width = 148.0 * scale;
        let color = if category == game.settings_category {
            Color::new(0.18, 0.08, 0.06, 1.0)
        } else {
            Color::new(0.075, 0.068, 0.059, 1.0)
        };
        draw_rectangle(x, category_y - 18.0 * scale, width, 25.0 * scale, color);
        draw_text(
            setting_category_name(category),
            x + 9.0 * scale,
            category_y,
            small_font,
            if category == game.settings_category {
                Color::new(1.0, 0.79, 0.39, 1.0)
            } else {
                Color::new(0.68, 0.67, 0.62, 1.0)
            },
        );
    }

    let definitions = settings_for_category(game.settings_category);
    let visible_rows = 8;
    let first_row = (game.settings_selection / visible_rows) * visible_rows;
    let end_row = (first_row + visible_rows).min(definitions.len());
    let row_top = panel_y + 157.0 * scale;
    let row_step = 34.0 * scale;
    let value_x = panel_x + panel_w - 95.0 * scale;
    for index in first_row..end_row {
        let y = row_top + (index - first_row) as f32 * row_step;
        let selected = index == game.settings_selection;
        if selected {
            draw_rectangle(
                panel_x + 20.0 * scale,
                y - 20.0 * scale,
                panel_w - 40.0 * scale,
                27.0 * scale,
                Color::new(0.24, 0.085, 0.06, 0.96),
            );
        }
        draw_text(
            definitions[index].label,
            panel_x + 34.0 * scale,
            y,
            (17.0 * scale).max(8.0),
            if selected {
                Color::new(1.0, 0.9, 0.72, 1.0)
            } else {
                Color::new(0.78, 0.77, 0.72, 1.0)
            },
        );
        let value = game.setting_value(definitions[index].key);
        let display = format!("{:.*}", definitions[index].decimals, value);
        draw_text(
            &display,
            value_x,
            y,
            (17.0 * scale).max(8.0),
            if selected {
                Color::new(1.0, 0.77, 0.34, 1.0)
            } else {
                Color::new(0.67, 0.8, 0.69, 1.0)
            },
        );
    }

    let page = format!(
        "{}  {}-{} / {}",
        setting_category_name(game.settings_category),
        first_row + 1,
        end_row,
        definitions.len()
    );
    draw_text(
        &page,
        panel_x + 30.0 * scale,
        panel_y + panel_h - 68.0 * scale,
        small_font,
        Color::new(0.72, 0.69, 0.63, 1.0),
    );
    draw_rectangle(
        panel_x + 24.0 * scale,
        panel_y + panel_h - 56.0 * scale,
        panel_w - 48.0 * scale,
        1.0,
        Color::new(0.28, 0.25, 0.22, 1.0),
    );
    draw_text(
        "UP/DOWN SELECT   LEFT/RIGHT EDIT   TAB CHANGE SECTION",
        panel_x + 30.0 * scale,
        panel_y + panel_h - 34.0 * scale,
        small_font,
        Color::new(0.8, 0.78, 0.71, 1.0),
    );
    draw_text(
        "1 EASY   2 MEDIUM   3 HARD   SHIFT+P OR ESC BACK",
        panel_x + 30.0 * scale,
        panel_y + panel_h - 12.0 * scale,
        small_font,
        Color::new(0.98, 0.77, 0.38, 1.0),
    );
}
