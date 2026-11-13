#![allow(dead_code)]

use std::io::{self, BufRead, Read};

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
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().is_some_and(|arg| arg == "--evaluate") {
        if let Err(error) = evaluate_episode(&arguments[1..]) {
            eprintln!("evaluation failed: {error}");
            std::process::exit(2);
        }
        return;
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

#[derive(Clone, Copy)]
struct FitnessWeights {
    level_completion: f32,
    partial_level: f32,
    health_per_point: f32,
    kill: f32,
    seconds: f32,
    exit_progress: f32,
    damage_dealt: f32,
    damage_taken: f32,
    armor_damage_taken: f32,
    health_gain: f32,
    armor_gain: f32,
    ammo_spent: f32,
    death: f32,
}

fn distance_to_exit(game: &game::Game) -> f32 {
    let exit = game.current_level().exit;
    let dx = exit.0 - game.x;
    let dy = exit.1 - game.y;
    (dx * dx + dy * dy).sqrt()
}

fn evaluate_episode(arguments: &[String]) -> Result<(), String> {
    if arguments.len() != 18 {
        return Err(
            "expected start_level, difficulty, dt, max_steps, max_levels_per_agent, and thirteen fitness weights"
                .into(),
        );
    }
    let parse = |index: usize| {
        arguments[index]
            .parse::<f32>()
            .map_err(|error| format!("invalid argument {}: {error}", index + 1))
    };
    let start_level = arguments[0]
        .parse::<usize>()
        .map_err(|error| format!("invalid start level: {error}"))?;
    let difficulty = &arguments[1];
    let dt = parse(2)?;
    let max_steps = arguments[3]
        .parse::<usize>()
        .map_err(|error| format!("invalid max steps: {error}"))?;
    let max_levels = arguments[4]
        .parse::<usize>()
        .map_err(|error| format!("invalid max levels per agent: {error}"))?;
    if max_levels == 0 {
        return Err("max_levels_per_agent must be positive".into());
    }
    let fitness = FitnessWeights {
        level_completion: parse(5)?,
        partial_level: parse(6)?,
        health_per_point: parse(7)?,
        kill: parse(8)?,
        seconds: parse(9)?,
        exit_progress: parse(10)?,
        damage_dealt: parse(11)?,
        damage_taken: parse(12)?,
        armor_damage_taken: parse(13)?,
        health_gain: parse(14)?,
        armor_gain: parse(15)?,
        ammo_spent: parse(16)?,
        death: parse(17)?,
    };

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("could not read policy weights: {error}"))?;
    let weights = input
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if weights.len() != 430 {
        return Err(format!(
            "expected 430 policy weights, got {}",
            weights.len()
        ));
    }

    let mut game = game::Game::new();
    game.agent_start(start_level, difficulty);
    let mut level_count = 0;
    let mut level_kills = 0;
    let mut total_kills = 0;
    let mut total_time = 0.0;
    let mut level_start_time = 0.0;
    let mut score = 0.0;
    let mut steps = 0;
    let mut last_exit_distance = distance_to_exit(&game);
    while steps < max_steps {
        let enemies_before = game.enemies.iter().filter(|enemy| enemy.alive).count();
        let previous_level = game.level_index;
        let action = game.agent_policy_action(&weights);
        game.agent_step(
            action.0, action.1, action.2, action.3, action.4, action.5, action.6, dt,
        );
        let enemies_after = game.enemies.iter().filter(|enemy| enemy.alive).count();
        let killed = if game.level_index == previous_level {
            enemies_before.saturating_sub(enemies_after)
        } else {
            0
        };
        level_kills += killed;
        total_kills += killed;
        steps += 1;
        total_time += dt;
        if game.level_index == previous_level {
            let distance_now = distance_to_exit(&game);
            score += (last_exit_distance - distance_now).max(0.0) * fitness.exit_progress;
            last_exit_distance = distance_now;
        } else {
            last_exit_distance = distance_to_exit(&game);
        }
        if game.level_index != previous_level {
            level_count += 1;
            score += fitness.level_completion;
            score += game.health as f32 * fitness.health_per_point;
            score += level_kills as f32 * fitness.kill;
            score -= (total_time - level_start_time) * fitness.seconds;
            level_kills = 0;
            level_start_time = total_time;
        }
        if game.dead || level_count >= max_levels {
            break;
        }
    }
    if level_count < max_levels {
        score += level_count as f32 * fitness.partial_level;
        score += game.health as f32 * fitness.health_per_point;
        score += level_kills as f32 * fitness.kill;
        score -= (total_time - level_start_time) * fitness.seconds;
    }
    score += game.damage_dealt * fitness.damage_dealt;
    score -= game.damage_taken as f32 * fitness.damage_taken;
    score -= game.armor_damage_taken as f32 * fitness.armor_damage_taken;
    score += game.health_gain as f32 * fitness.health_gain;
    score += game.armor_gain as f32 * fitness.armor_gain;
    score -= game.ammo_spent as f32 * fitness.ammo_spent;
    if game.dead {
        score -= fitness.death;
    }
    println!(
        "{score},{level_count},{total_kills},{},{total_time:.2},{start_level}",
        game.health
    );
    Ok(())
}
