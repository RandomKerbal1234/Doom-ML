#![allow(dead_code)]

use std::io::{self, BufRead};

const WIDTH: usize = 320;
const HEIGHT: usize = 200;
const VIEW_HEIGHT: usize = 170;
const HUD_TOP: usize = 172;

#[path = "../game.rs"]
mod game;
#[path = "../levels.rs"]
mod levels;
#[path = "../render.rs"]
mod render;
#[path = "../settings.rs"]
mod settings;

fn main() {
    if let Err(error) = levels::initialize_levels() {
        eprintln!("level initialization failed: {error}");
        std::process::exit(1);
    }
    let stdin = io::stdin();
    let mut game = game::Game::new();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let fields = line.trim().split(',').collect::<Vec<_>>();
        match fields.first().copied() {
            Some("reset") => {
                let level = fields.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                game.agent_start(level, fields.get(2).copied().unwrap_or("medium"));
                println!("{}", game.agent_observation());
            }
            Some("step") if fields.len() >= 9 => {
                let n = |i: usize| fields[i].parse::<f32>().unwrap_or(0.0);
                game.agent_step(
                    n(1),
                    n(2),
                    n(3),
                    n(4) != 0.0,
                    n(5) != 0.0,
                    n(6).max(0.0) as usize,
                    n(7) != 0.0,
                    n(8),
                );
                println!("{}", game.agent_observation());
            }
            Some("quit") => break,
            _ => println!("{{\"error\":\"invalid command\"}}"),
        }
    }
}
