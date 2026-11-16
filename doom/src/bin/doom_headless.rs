#![allow(dead_code)]

use std::io::{self, BufRead, Read, Write};
use std::time::{Duration, Instant};

const WIDTH: usize = 320;
const HEIGHT: usize = 200;
const VIEW_HEIGHT: usize = 170;
const HUD_TOP: usize = 172;

#[path = "../game.rs"]
mod game;
#[path = "../levels.rs"]
mod levels;
#[path = "../settings.rs"]
mod settings;

fn main() {
    let main_start = Instant::now();
    if let Err(error) = levels::initialize_levels() {
        eprintln!("level initialization failed: {error}");
        std::process::exit(1);
    }
    let levels_elapsed = main_start.elapsed();
    let profiling = std::env::var_os("DOOM_PROFILE").is_some_and(|value| value != "0");
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    let (shape, arguments) = match take_shape(&raw) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("invalid --arch: {error}");
            std::process::exit(2);
        }
    };
    match arguments.first().map(String::as_str) {
        Some("--evaluate") => {
            if let Err(error) = evaluate_episode(
                &arguments[1..],
                &shape,
                main_start,
                levels_elapsed,
                profiling,
            ) {
                eprintln!("evaluation failed: {error}");
                std::process::exit(2);
            }
            return;
        }
        Some("--evaluate-server") => {
            if let Err(error) = evaluate_server(&shape, profiling) {
                eprintln!("evaluation server failed: {error}");
                std::process::exit(2);
            }
            return;
        }
        Some("--probe") => {
            if let Err(error) = probe_policy(&arguments[1..], &shape) {
                eprintln!("probe failed: {error}");
                std::process::exit(2);
            }
            return;
        }
        _ => {}
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

/// Extracts `--arch a,b,c` (or `--arch=a,b,c`) from the command line.
///
/// The architecture is a launch-time property of the process, so it never has
/// to travel inside a job line. Omitting it selects the single-layer policy.
fn take_shape(arguments: &[String]) -> Result<(Vec<usize>, Vec<String>), String> {
    let mut shape = vec![game::POLICY_INPUTS, game::POLICY_OUTPUTS];
    let mut remaining = Vec::with_capacity(arguments.len());
    let mut index = 0;
    while index < arguments.len() {
        let current = &arguments[index];
        let value = if let Some(inline) = current.strip_prefix("--arch=") {
            Some(inline.to_string())
        } else if current == "--arch" {
            index += 1;
            Some(
                arguments
                    .get(index)
                    .cloned()
                    .ok_or("--arch needs a shape such as 64,16,8,10")?,
            )
        } else {
            None
        };
        match value {
            Some(text) => {
                let widths = text
                    .split(',')
                    .map(|item| {
                        item.trim()
                            .parse::<usize>()
                            .map_err(|error| format!("invalid layer width: {error}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if widths.len() < 2 {
                    return Err("a shape needs at least an input and an output layer".into());
                }
                shape = widths;
            }
            None => remaining.push(current.clone()),
        }
        index += 1;
    }
    Ok((shape, remaining))
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

struct EpisodeParams {
    start_level: usize,
    difficulty: String,
    dt: f32,
    max_steps: usize,
    max_levels: usize,
    fitness: FitnessWeights,
}

fn distance_to_exit(game: &game::Game) -> f32 {
    let exit = game.current_level().exit;
    let dx = exit.0 - game.x;
    let dy = exit.1 - game.y;
    (dx * dx + dy * dy).sqrt()
}

struct ProfileSnapshot {
    steps: usize,
    levels: usize,
    main_ns: u64,
    levels_ns: u64,
    setup_ns: u64,
    policy_ns: u64,
    sim_ns: u64,
    killcount_ns: u64,
    loop_other_ns: u64,
    total_ns: u64,
    level_load_ns: u64,
    movement_ns: u64,
    fire_ns: u64,
    update_ns: u64,
    enemy_ns: u64,
    danger_ns: u64,
    pickup_ns: u64,
}

impl ProfileSnapshot {
    fn to_json(&self) -> String {
        format!(
            "{{\"steps\":{},\"levels\":{},\"main_ns\":{},\"levels_ns\":{},\
             \"setup_ns\":{},\"policy_ns\":{},\"sim_ns\":{},\"killcount_ns\":{},\
             \"loop_other_ns\":{},\"total_ns\":{},\"level_load_ns\":{},\
             \"movement_ns\":{},\"fire_ns\":{},\"update_ns\":{},\"enemy_ns\":{},\
             \"danger_ns\":{},\"pickup_ns\":{}}}",
            self.steps,
            self.levels,
            self.main_ns,
            self.levels_ns,
            self.setup_ns,
            self.policy_ns,
            self.sim_ns,
            self.killcount_ns,
            self.loop_other_ns,
            self.total_ns,
            self.level_load_ns,
            self.movement_ns,
            self.fire_ns,
            self.update_ns,
            self.enemy_ns,
            self.danger_ns,
            self.pickup_ns,
        )
    }
}

struct EpisodeResult {
    score: f32,
    levels: usize,
    kills: usize,
    health: i32,
    seconds: f32,
    start_level: usize,
    profile: Option<ProfileSnapshot>,
}

fn parse_cli_params(arguments: &[String]) -> Result<EpisodeParams, String> {
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
    let max_levels = arguments[4]
        .parse::<usize>()
        .map_err(|error| format!("invalid max levels per agent: {error}"))?;
    if max_levels == 0 {
        return Err("max_levels_per_agent must be positive".into());
    }
    Ok(EpisodeParams {
        start_level: arguments[0]
            .parse::<usize>()
            .map_err(|error| format!("invalid start level: {error}"))?,
        difficulty: arguments[1].clone(),
        dt: parse(2)?,
        max_steps: arguments[3]
            .parse::<usize>()
            .map_err(|error| format!("invalid max steps: {error}"))?,
        max_levels,
        fitness: FitnessWeights {
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
        },
    })
}

fn parse_job(
    line: &str,
    shape: &[usize],
    parameters: usize,
    reader: &mut impl BufRead,
) -> Result<(EpisodeParams, game::Policy), String> {
    let mut fields = line.split(',').collect::<Vec<_>>();
    // A leading `b` marks a binary weight blob that follows the header line.
    // Text job lines (no marker) stay supported for hand-written debugging.
    let binary_weights = fields.first().is_some_and(|field| *field == "b");
    if binary_weights {
        fields.remove(0);
    }
    if fields.len() < 18 {
        return Err(format!("expected at least 18 fields, got {}", fields.len()));
    }
    let number = |index: usize, name: &str| {
        fields[index]
            .trim()
            .parse::<f32>()
            .map_err(|error| format!("invalid {name}: {error}"))
    };
    let start_level = number(0, "start_level")? as usize;
    let difficulty = fields[1].trim().to_string();
    let dt = number(2, "dt")?;
    let max_steps = number(3, "max_steps")? as usize;
    let max_levels = number(4, "max_levels_per_agent")? as usize;
    if max_levels == 0 {
        return Err("max_levels_per_agent must be positive".into());
    }
    let fitness = FitnessWeights {
        level_completion: number(5, "level_completion")?,
        partial_level: number(6, "partial_level")?,
        health_per_point: number(7, "health_per_point")?,
        kill: number(8, "kill")?,
        seconds: number(9, "seconds")?,
        exit_progress: number(10, "exit_progress")?,
        damage_dealt: number(11, "damage_dealt")?,
        damage_taken: number(12, "damage_taken")?,
        armor_damage_taken: number(13, "armor_damage_taken")?,
        health_gain: number(14, "health_gain")?,
        armor_gain: number(15, "armor_gain")?,
        ammo_spent: number(16, "ammo_spent")?,
        death: number(17, "death")?,
    };
    let weights = if binary_weights {
        read_binary_weights(reader, parameters)?
    } else {
        fields[18..]
            .iter()
            .map(|value| {
                value
                    .trim()
                    .parse::<f32>()
                    .map_err(|error| format!("invalid weight: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let policy = game::Policy::new(shape, weights)?;
    Ok((
        EpisodeParams {
            start_level,
            difficulty,
            dt,
            max_steps,
            max_levels,
            fitness,
        },
        policy,
    ))
}

/// Runs one complete episode. The caller owns the `Game`, which lets the
/// evaluate-server reuse the same allocation across many jobs.
fn run_episode(
    game: &mut game::Game,
    params: &EpisodeParams,
    policy: &game::Policy,
    profiling: bool,
) -> EpisodeResult {
    let total_start = Instant::now();
    let setup_start = Instant::now();
    game.agent_start(params.start_level, &params.difficulty);
    let setup_ns = setup_start.elapsed().as_nanos() as u64;

    let mut level_count = 0;
    let mut level_kills = 0;
    let mut total_kills = 0;
    let mut total_time = 0.0;
    let mut level_start_time = 0.0;
    let mut score = 0.0;
    let mut steps = 0;
    let mut last_exit_distance = distance_to_exit(game);
    let mut policy_ns = 0u64;
    let mut sim_ns = 0u64;
    let mut killcount_ns = 0u64;
    let loop_start = Instant::now();
    while steps < params.max_steps {
        let scan_start = Instant::now();
        let kills_before = game.kills;
        killcount_ns += scan_start.elapsed().as_nanos() as u64;
        let previous_level = game.level_index;
        let policy_start = Instant::now();
        let action = game.agent_policy_action(policy);
        policy_ns += policy_start.elapsed().as_nanos() as u64;
        let sim_start = Instant::now();
        game.agent_step(
            action.0, action.1, action.2, action.3, action.4, action.5, action.6, params.dt,
        );
        sim_ns += sim_start.elapsed().as_nanos() as u64;
        let scan_start = Instant::now();
        let killed = if game.level_index == previous_level {
            (game.kills - kills_before) as usize
        } else {
            0
        };
        killcount_ns += scan_start.elapsed().as_nanos() as u64;
        level_kills += killed;
        total_kills += killed;
        steps += 1;
        total_time += params.dt;
        if game.level_index == previous_level {
            let distance_now = distance_to_exit(game);
            score += (last_exit_distance - distance_now).max(0.0) * params.fitness.exit_progress;
            last_exit_distance = distance_now;
        } else {
            last_exit_distance = distance_to_exit(game);
        }
        if game.level_index != previous_level {
            level_count += 1;
            score += params.fitness.level_completion;
            score += game.health as f32 * params.fitness.health_per_point;
            score += level_kills as f32 * params.fitness.kill;
            score -= (total_time - level_start_time) * params.fitness.seconds;
            level_kills = 0;
            level_start_time = total_time;
        }
        if game.dead || level_count >= params.max_levels {
            break;
        }
    }
    let loop_ns = loop_start.elapsed().as_nanos() as u64;
    if level_count < params.max_levels {
        score += level_count as f32 * params.fitness.partial_level;
        score += game.health as f32 * params.fitness.health_per_point;
        score += level_kills as f32 * params.fitness.kill;
        score -= (total_time - level_start_time) * params.fitness.seconds;
    }
    score += game.damage_dealt * params.fitness.damage_dealt;
    score -= game.damage_taken as f32 * params.fitness.damage_taken;
    score -= game.armor_damage_taken as f32 * params.fitness.armor_damage_taken;
    score += game.health_gain as f32 * params.fitness.health_gain;
    score += game.armor_gain as f32 * params.fitness.armor_gain;
    score -= game.ammo_spent as f32 * params.fitness.ammo_spent;
    if game.dead {
        score -= params.fitness.death;
    }

    let profile = profiling.then(|| {
        let p = game.profile;
        let loop_other_ns = loop_ns
            .saturating_sub(policy_ns)
            .saturating_sub(sim_ns)
            .saturating_sub(killcount_ns);
        ProfileSnapshot {
            steps,
            levels: level_count,
            main_ns: 0,
            levels_ns: 0,
            setup_ns,
            policy_ns,
            sim_ns,
            killcount_ns,
            loop_other_ns,
            total_ns: total_start.elapsed().as_nanos() as u64,
            level_load_ns: p.level_load_ns,
            movement_ns: p.movement_ns,
            fire_ns: p.fire_ns,
            update_ns: p.update_ns,
            enemy_ns: p.enemy_ns,
            danger_ns: p.danger_ns,
            pickup_ns: p.pickup_ns,
        }
    });
    EpisodeResult {
        score,
        levels: level_count,
        kills: total_kills,
        health: game.health,
        seconds: total_time,
        start_level: params.start_level,
        profile,
    }
}

fn read_stdin_weights() -> Result<Vec<f32>, String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("could not read policy weights: {error}"))?;
    input
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()
}

fn evaluate_episode(
    arguments: &[String],
    shape: &[usize],
    main_start: Instant,
    levels_elapsed: Duration,
    profiling: bool,
) -> Result<(), String> {
    let params = parse_cli_params(arguments)?;
    let policy = game::Policy::new(shape, read_stdin_weights()?)?;

    let mut game = game::Game::new();
    game.set_headless();
    if profiling {
        game.enable_profiling();
    }
    let result = run_episode(&mut game, &params, &policy, profiling);
    println!(
        "{},{},{},{},{:.2},{}",
        result.score, result.levels, result.kills, result.health, result.seconds,
        result.start_level
    );
    if let Some(profile) = result.profile {
        let mut profile = profile;
        profile.main_ns = main_start.elapsed().as_nanos() as u64;
        profile.levels_ns = levels_elapsed.as_nanos() as u64;
        eprintln!("PROFILE {}", profile.to_json());
    }
    Ok(())
}

fn read_binary_weights(reader: &mut impl BufRead, count: usize) -> Result<Vec<f32>, String> {
    let bytes_len = count
        .checked_mul(4)
        .ok_or("policy weight count is too large")?;
    let mut bytes = vec![0u8; bytes_len];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("could not read binary policy weights: {error}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn evaluate_server(shape: &[usize], profiling: bool) -> Result<(), String> {
    let mut game = game::Game::new();
    game.set_headless();
    if profiling {
        game.enable_profiling();
    }
    // Reading the header line and the weight blob must go through the same
    // buffered reader, or the blob's first bytes stay stranded in the buffer.
    let parameters = game::Policy::parameters(shape);
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    let mut header = Vec::with_capacity(4096);
    loop {
        header.clear();
        let read = reader
            .read_until(b'\n', &mut header)
            .map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        while matches!(header.last(), Some(b'\n') | Some(b'\r')) {
            header.pop();
        }
        if header.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let job_start = Instant::now();
        let parsed = match std::str::from_utf8(&header) {
            Ok(line) => parse_job(line, shape, parameters, &mut reader),
            Err(error) => Err(format!("job header is not valid utf-8: {error}")),
        };
        let (params, policy) = match parsed {
            Ok(job) => job,
            Err(error) => {
                writeln!(writer, "{{\"error\":\"{}\"}}", error.replace('"', "'"))
                    .map_err(|error| error.to_string())?;
                continue;
            }
        };
        let result = run_episode(&mut game, &params, &policy, profiling);
        if let Some(profile) = result.profile {
            let mut profile = profile;
            profile.main_ns = job_start.elapsed().as_nanos() as u64;
            writeln!(
                writer,
                "{{\"score\":{},\"levels\":{},\"kills\":{},\"health\":{},\"seconds\":{:.2},\
                 \"start_level\":{},\"profile\":{}}}",
                result.score,
                result.levels,
                result.kills,
                result.health,
                result.seconds,
                result.start_level,
                profile.to_json(),
            )
            .map_err(|error| error.to_string())?;
        } else {
            writeln!(
                writer,
                "{{\"score\":{},\"levels\":{},\"kills\":{},\"health\":{},\"seconds\":{:.2},\
                 \"start_level\":{}}}",
                result.score, result.levels, result.kills, result.health, result.seconds,
                result.start_level,
            )
            .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

/// Prints the policy inputs and outputs for the first step of an episode.
///
/// `training/env.py` computes the same vector in Python; comparing the two
/// keeps the duplicated encoder and forward pass from drifting apart.
fn probe_policy(arguments: &[String], shape: &[usize]) -> Result<(), String> {
    let level = arguments
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let difficulty = arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| "medium".to_string());
    let policy = game::Policy::new(shape, read_stdin_weights()?)?;
    let mut game = game::Game::new();
    game.set_headless();
    game.agent_start(level, &difficulty);
    let inputs = game.agent_inputs();
    let outputs = policy.forward(&inputs);
    let inputs_json = inputs
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let outputs_json = outputs
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("{{\"shape\":{shape:?},\"inputs\":[{inputs_json}],\"outputs\":[{outputs_json}]}}");
    Ok(())
}
