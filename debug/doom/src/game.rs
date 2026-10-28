use crate::levels::{
    Level, MAP_HEIGHT, MAP_WIDTH, PickupKind, level_tile, levels,
};
use crate::settings::{DifficultyPreset, GameSettings, SettingKey};
use crate::{HEIGHT, WIDTH};
use std::time::Instant;

/// Accumulated timings for the headless trainer. Every value is nanoseconds.
#[derive(Clone, Copy, Default)]
pub(crate) struct GameProfile {
    pub(crate) level_load_ns: u64,
    pub(crate) movement_ns: u64,
    pub(crate) fire_ns: u64,
    pub(crate) update_ns: u64,
    pub(crate) enemy_ns: u64,
    pub(crate) danger_ns: u64,
    pub(crate) pickup_ns: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct Enemy {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) hp: f32,
    pub(crate) kind: u8,
    pub(crate) attack_timer: f32,
    pub(crate) hurt_timer: f32,
    pub(crate) alive: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct Pickup {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) kind: PickupKind,
    pub(crate) active: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct Sprite {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) kind: SpriteKind,
    pub(crate) size: f32,
    pub(crate) depth: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct HitEffect {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) weapon: usize,
    pub(crate) kind: ImpactKind,
    pub(crate) timer: f32,
    pub(crate) duration: f32,
}

#[derive(Clone, Copy)]
pub(crate) enum ImpactKind {
    Enemy,
    Wall,
    Miss,
}

#[derive(Clone, Copy)]
pub(crate) enum SpriteKind {
    Enemy(u8, bool),
    Pickup(PickupKind),
    Exit,
}

pub(crate) struct Game {
    pub(crate) level_index: usize,
    pub(crate) selected_level: usize,
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) angle: f32,
    pub(crate) health: i32,
    pub(crate) armor: i32,
    pub(crate) ammo: [i32; 4],
    pub(crate) unlocked: [bool; 4],
    pub(crate) weapon: usize,
    pub(crate) enemies: Vec<Enemy>,
    pub(crate) pickups: Vec<Pickup>,
    pub(crate) hit_effects: Vec<HitEffect>,
    pub(crate) paused: bool,
    pub(crate) settings_open: bool,
    pub(crate) settings_return_paused: bool,
    pub(crate) settings_category: usize,
    pub(crate) settings_selection: usize,
    pub(crate) difficulty: DifficultyPreset,
    pub(crate) settings: GameSettings,
    pub(crate) dead: bool,
    pub(crate) won: bool,
    pub(crate) shot_timer: f32,
    pub(crate) recoil: f32,
    pub(crate) muzzle_timer: f32,
    pub(crate) muzzle_weapon: usize,
    pub(crate) damage_flash: f32,
    pub(crate) danger: bool,
    pub(crate) danger_angle: f32,
    pub(crate) message: String,
    pub(crate) message_timer: f32,
    pub(crate) time: f32,
    pub(crate) kills: u64,
    pub(crate) profiling: bool,
    pub(crate) profile: GameProfile,
    pub(crate) track_danger: bool,
    pub(crate) visual_effects: bool,
    pub(crate) pixels: Vec<u8>,
    pub(crate) z_buffer: [f32; WIDTH],
    pub(crate) wall_grid: [bool; MAP_WIDTH * MAP_HEIGHT],
}

impl Game {
    pub(crate) fn new() -> Self {
        let settings = GameSettings::medium();
        let mut game = Self {
            level_index: 0,
            selected_level: 0,
            x: 2.5,
            y: 2.5,
            angle: 0.0,
            health: settings.max_health.round() as i32,
            armor: 0,
            ammo: [0, 0, 0, 0],
            unlocked: [true, false, false, false],
            weapon: 0,
            enemies: Vec::with_capacity(8),
            pickups: Vec::with_capacity(8),
            hit_effects: Vec::with_capacity(8),
            paused: true,
            settings_open: false,
            settings_return_paused: false,
            settings_category: 0,
            settings_selection: 0,
            difficulty: DifficultyPreset::Medium,
            settings,
            dead: false,
            won: false,
            shot_timer: 0.0,
            recoil: 0.0,
            muzzle_timer: 0.0,
            muzzle_weapon: 0,
            damage_flash: 0.0,
            danger: false,
            danger_angle: 0.0,
            message: String::new(),
            message_timer: 0.0,
            time: 0.0,
            kills: 0,
            profiling: false,
            profile: GameProfile::default(),
            track_danger: true,
            visual_effects: true,
            pixels: vec![0; WIDTH * HEIGHT * 4],
            z_buffer: [0.0; WIDTH],
            wall_grid: [true; MAP_WIDTH * MAP_HEIGHT],
        };
        game.ammo[0] = game.settings.starting_pistol_ammo.round() as i32;
        game.load_level(0);
        game
    }

    /// Turns on nanosecond timers around the simulation phases used by training.
    #[allow(dead_code)] // Used by the headless training binary.
    pub(crate) fn enable_profiling(&mut self) {
        self.profiling = true;
    }

    /// Disables work that only affects the windowed presentation.
    #[allow(dead_code)] // Used by the headless training binary.
    pub(crate) fn set_headless(&mut self) {
        self.track_danger = false;
        self.visual_effects = false;
    }

    pub(crate) fn load_level(&mut self, index: usize) {
        let profile_start = if self.profiling {
            Some(Instant::now())
        } else {
            None
        };
        self.level_index = index;
        self.selected_level = index;
        self.danger = false;
        let level = &levels()[index];
        self.wall_grid.fill(false);
        for (y, row) in level.map.iter().enumerate() {
            for (x, tile) in row.bytes().enumerate() {
                self.wall_grid[y * MAP_WIDTH + x] = matches!(tile, b'#' | b'B' | b'M');
            }
        }
        self.x = level.spawn.0;
        self.y = level.spawn.1;
        self.angle = level.spawn.2;
        self.enemies.clear();
        self.enemies
            .extend(level.enemies.iter().map(|&(x, y, kind)| Enemy {
                x,
                y,
                hp: if kind == 0 {
                    self.settings.imp_health
                } else {
                    self.settings.demon_health
                },
                kind,
                attack_timer: 0.8,
                hurt_timer: 0.0,
                alive: true,
            }));
        self.pickups.clear();
        self.pickups.extend(level.pickups.iter().map(|p| Pickup {
            x: p.x,
            y: p.y,
            kind: p.kind,
            active: true,
        }));
        self.message = format!("{}  //  {}", level.code, level.title);
        self.message_timer = 3.0;
        if let Some(start) = profile_start {
            self.profile.level_load_ns += start.elapsed().as_nanos() as u64;
        }
    }

    pub(crate) fn current_level(&self) -> &Level {
        &levels()[self.level_index]
    }

    pub(crate) fn start_new_run(&mut self, first_level: usize) {
        self.health = self.settings.max_health.round() as i32;
        self.armor = 0;
        self.ammo = [self.settings.starting_pistol_ammo.round() as i32, 0, 0, 0];
        self.unlocked = [true, false, false, false];
        self.weapon = 0;
        self.dead = false;
        self.won = false;
        self.paused = false;
        self.settings_open = false;
        self.selected_level = first_level;
        self.kills = 0;
        self.load_level(first_level);
    }

    /// Starts a fresh headless training episode at the requested level.
    #[allow(dead_code)] // Used by the separate headless training binary.
    pub(crate) fn agent_start(&mut self, first_level: usize, preset: &str) {
        let preset = match preset {
            "easy" => DifficultyPreset::Easy,
            "hard" => DifficultyPreset::Hard,
            _ => DifficultyPreset::Medium,
        };
        self.settings = GameSettings::preset(preset);
        self.difficulty = preset;
        self.time = 0.0;
        self.shot_timer = 0.0;
        self.recoil = 0.0;
        self.muzzle_timer = 0.0;
        self.damage_flash = 0.0;
        self.message_timer = 0.0;
        self.hit_effects.clear();
        self.danger = false;
        self.profile = GameProfile::default();
        self.kills = 0;
        self.start_new_run(first_level % levels().len());
    }

    /// Applies one normalized agent action and advances the actual game simulation.
    /// turn/forward/strafe are in [-1, 1]; dt is bounded by the caller.
    #[allow(dead_code)] // Used by the separate headless training binary.
    pub(crate) fn agent_step(
        &mut self,
        turn: f32,
        forward: f32,
        strafe: f32,
        fire: bool,
        activate_exit: bool,
        weapon: usize,
        run: bool,
        dt: f32,
    ) {
        if self.dead || self.won || self.paused || self.settings_open {
            return;
        }
        let movement_start = if self.profiling {
            Some(Instant::now())
        } else {
            None
        };
        let dt = dt.clamp(0.0, 0.05);
        if weapon < self.unlocked.len() && self.unlocked[weapon] {
            self.weapon = weapon;
        }
        let turn_speed = if run {
            self.settings.run_turn_speed
        } else {
            self.settings.turn_speed
        };
        self.angle = (self.angle + turn.clamp(-1.0, 1.0) * turn_speed * dt)
            .rem_euclid(std::f32::consts::TAU);
        let forward = forward.clamp(-1.0, 1.0);
        let strafe = strafe.clamp(-1.0, 1.0);
        let mut vx = self.angle.cos() * forward - self.angle.sin() * strafe;
        let mut vy = self.angle.sin() * forward + self.angle.cos() * strafe;
        let magnitude = (vx * vx + vy * vy).sqrt();
        if magnitude > 1.0 {
            vx /= magnitude;
            vy /= magnitude;
        }
        let speed = if run {
            self.settings.run_speed
        } else {
            self.settings.walk_speed
        };
        let (self_x, self_y) = self.try_move(self.x + vx * speed * dt, self.y + vy * speed * dt);
        self.x = self_x;
        self.y = self_y;

        if activate_exit {
            let exit = self.current_level().exit;
            if distance(self.x, self.y, exit.0, exit.1) < 1.35 {
                self.load_level((self.level_index + 1) % levels().len());
            }
        }
        if let Some(start) = movement_start {
            self.profile.movement_ns += start.elapsed().as_nanos() as u64;
        }
        let fire_start = if self.profiling {
            Some(Instant::now())
        } else {
            None
        };
        if fire && self.shot_timer <= 0.0 {
            self.fire();
        }
        if let Some(start) = fire_start {
            self.profile.fire_ns += start.elapsed().as_nanos() as u64;
        }
        let update_start = if self.profiling {
            Some(Instant::now())
        } else {
            None
        };
        self.update(dt);
        if let Some(start) = update_start {
            self.profile.update_ns += start.elapsed().as_nanos() as u64;
        }
    }

    #[allow(dead_code)] // Used by the separate headless training binary.
    pub(crate) fn agent_observation(&self) -> String {
        let level = self.current_level();
        let enemies = self
            .enemies
            .iter()
            .filter(|e| e.alive)
            .map(|e| {
                format!(
                    "{{\"x\":{:.3},\"y\":{:.3},\"hp\":{:.2},\"kind\":{}}}",
                    e.x, e.y, e.hp, e.kind
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let pickups = self
            .pickups
            .iter()
            .filter(|p| p.active)
            .map(|p| {
                format!(
                    "{{\"x\":{:.3},\"y\":{:.3},\"kind\":\"{}\"}}",
                    p.x,
                    p.y,
                    p.kind.label()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let map = level
            .map
            .iter()
            .map(|r| format!("\"{}\"", r))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"level\":{},\"x\":{:.3},\"y\":{:.3},\"angle\":{:.4},\"health\":{},\"armor\":{},\"ammo\":[{},{},{},{}],\"unlocked\":[{},{},{},{}],\"weapon\":{},\"time\":{:.3},\"dead\":{},\"exit\":[{:.3},{:.3}],\"map\":[{}],\"enemies\":[{}],\"pickups\":[{}]}}",
            self.level_index,
            self.x,
            self.y,
            self.angle,
            self.health,
            self.armor,
            self.ammo[0],
            self.ammo[1],
            self.ammo[2],
            self.ammo[3],
            self.unlocked[0],
            self.unlocked[1],
            self.unlocked[2],
            self.unlocked[3],
            self.weapon,
            self.time,
            self.dead,
            level.exit.0,
            level.exit.1,
            map,
            enemies,
            pickups
        )
    }

    /// Computes the same normalized 42-value input vector used by training/env.py.
    pub(crate) fn agent_policy_action(
        &self,
        weights: &[f32],
    ) -> (f32, f32, f32, bool, bool, usize, bool) {
        const INPUTS: usize = 42;
        const OUTPUTS: usize = 10;
        let width = INPUTS + 1;
        if weights.len() != OUTPUTS * width {
            return (0.0, 0.0, 0.0, false, false, 0, true);
        }
        let mut values = [0.0f32; INPUTS];
        let mut n = 0usize;
        for ray in 0..8 {
            let direction = self.angle + ray as f32 * std::f32::consts::TAU / 8.0;
            let mut distance = 0.0;
            for k in 1..=80 {
                distance = k as f32 * 0.1;
                let sx = (self.x + direction.cos() * distance) as i32;
                let sy = (self.y + direction.sin() * distance) as i32;
                if self.is_wall(sx, sy) {
                    break;
                }
            }
            values[n] = (distance / 8.0).min(1.0);
            n += 1;
        }

        // Find the two nearest living enemies in one pass. This is equivalent
        // to the old stable sort by distance but avoids the allocation.
        let mut nearest: Option<(f32, usize)> = None;
        let mut second: Option<(f32, usize)> = None;
        for (index, enemy) in self.enemies.iter().enumerate() {
            if !enemy.alive {
                continue;
            }
            let dist = distance(self.x, self.y, enemy.x, enemy.y);
            if nearest.is_none_or(|(best, _)| dist < best) {
                second = nearest;
                nearest = Some((dist, index));
            } else if second.is_none_or(|(best, _)| dist < best) {
                second = Some((dist, index));
            }
        }
        for slot in 0..2 {
            let found = if slot == 0 { nearest } else { second };
            if let Some((dist, index)) = found {
                let enemy = &self.enemies[index];
                let dx = enemy.x - self.x;
                let dy = enemy.y - self.y;
                let bearing = dy.atan2(dx) - self.angle;
                values[n] = bearing.sin();
                values[n + 1] = bearing.cos();
                values[n + 2] = (dist / 12.0).min(1.0);
                values[n + 3] = (enemy.hp / 200.0).min(1.0);
                values[n + 4] = enemy.kind as f32;
            } else {
                values[n] = 0.0;
                values[n + 1] = 0.0;
                values[n + 2] = 1.0;
                values[n + 3] = 0.0;
                values[n + 4] = 0.0;
            }
            n += 5;
        }

        let mut nearest_pickup: Option<(f32, usize)> = None;
        for (index, pickup) in self.pickups.iter().enumerate() {
            if !pickup.active {
                continue;
            }
            let dist = distance(self.x, self.y, pickup.x, pickup.y);
            if nearest_pickup.is_none_or(|(best, _)| dist < best) {
                nearest_pickup = Some((dist, index));
            }
        }
        if let Some((dist, index)) = nearest_pickup {
            let pickup = &self.pickups[index];
            let dx = pickup.x - self.x;
            let dy = pickup.y - self.y;
            let bearing = dy.atan2(dx) - self.angle;
            values[n] = bearing.sin();
            values[n + 1] = bearing.cos();
            values[n + 2] = (dist / 12.0).min(1.0);
            values[n + 3] = (pickup.kind == PickupKind::Shotgun) as u8 as f32;
            values[n + 4] = (pickup.kind == PickupKind::Chaingun) as u8 as f32;
            values[n + 5] = (pickup.kind == PickupKind::RocketLauncher) as u8 as f32;
            values[n + 6] = (pickup.kind == PickupKind::Medkit) as u8 as f32;
            values[n + 7] = (pickup.kind == PickupKind::Armor) as u8 as f32;
            values[n + 8] = (pickup.kind == PickupKind::Ammo) as u8 as f32;
        } else {
            values[n] = 0.0;
            values[n + 1] = 0.0;
            values[n + 2] = 1.0;
            for slot in 3..9 {
                values[n + slot] = 0.0;
            }
        }
        n += 9;

        let exit = self.current_level().exit;
        let dx = exit.0 - self.x;
        let dy = exit.1 - self.y;
        let bearing = dy.atan2(dx) - self.angle;
        values[n] = bearing.sin();
        values[n + 1] = bearing.cos();
        values[n + 2] = (distance(self.x, self.y, exit.0, exit.1) / 16.0).min(1.0);
        values[n + 3] = self.health as f32 / 200.0;
        values[n + 4] = self.armor as f32 / 200.0;
        n += 5;

        for ammo in self.ammo {
            values[n] = (ammo.max(0) as f32 / 100.0).min(1.0);
            n += 1;
        }
        for unlocked in self.unlocked {
            values[n] = unlocked as u8 as f32;
            n += 1;
        }
        values[n] = self.weapon as f32 / 3.0;
        values[n + 1] = (self.time / 120.0).min(1.0);
        n += 2;
        debug_assert_eq!(n, INPUTS);

        let mut output = [0.0f32; OUTPUTS];
        for (out, value) in output.iter_mut().enumerate() {
            let start = out * width;
            let row = &weights[start..start + INPUTS];
            let mut sum = weights[start + INPUTS];
            for (weight, input) in row.iter().zip(values.iter()) {
                sum += weight * input;
            }
            *value = sum.tanh();
        }
        let mut weapon = (0..4)
            .find(|&candidate| self.unlocked[candidate])
            .unwrap_or(0);
        for candidate in 0..4 {
            if self.unlocked[candidate] && output[5 + candidate] > output[5 + weapon] {
                weapon = candidate;
            }
        }
        (
            output[0],
            output[1],
            output[2],
            output[3] > 0.0,
            output[4] > 0.0,
            weapon,
            output[9] > 0.0,
        )
    }

    pub(crate) fn toggle_settings(&mut self) {
        if self.settings_open {
            self.settings_open = false;
            self.paused = self.settings_return_paused;
        } else {
            self.settings_return_paused = self.paused;
            self.settings_open = true;
        }
    }

    pub(crate) fn apply_preset(&mut self, preset: DifficultyPreset) {
        let next = GameSettings::preset(preset);
        let imp_scale = next.imp_health / self.settings.imp_health.max(1.0);
        let demon_scale = next.demon_health / self.settings.demon_health.max(1.0);
        for enemy in &mut self.enemies {
            if enemy.alive {
                enemy.hp *= if enemy.kind == 0 {
                    imp_scale
                } else {
                    demon_scale
                };
            }
        }
        self.settings = next;
        self.health = self.health.min(self.settings.max_health.round() as i32);
        self.armor = self.armor.min(self.settings.max_armor.round() as i32);
        self.dead = self.health <= 0;
        self.difficulty = preset;
    }

    pub(crate) fn setting_value(&self, key: SettingKey) -> f32 {
        match key {
            SettingKey::CurrentHealth => self.health as f32,
            SettingKey::CurrentArmor => self.armor as f32,
            SettingKey::CurrentPistolAmmo => self.ammo[0] as f32,
            SettingKey::CurrentShotgunAmmo => self.ammo[1] as f32,
            SettingKey::CurrentChaingunAmmo => self.ammo[2] as f32,
            SettingKey::CurrentRocketAmmo => self.ammo[3] as f32,
            _ => self
                .settings
                .setting_value(key)
                .expect("setting key has a field"),
        }
    }

    pub(crate) fn write_setting_value(&mut self, key: SettingKey, value: f32) {
        match key {
            SettingKey::CurrentHealth => {
                self.health = value.round().min(self.settings.max_health.round()) as i32;
                self.dead = self.health <= 0;
            }
            SettingKey::CurrentArmor => {
                self.armor = value.round().min(self.settings.max_armor.round()) as i32;
            }
            SettingKey::CurrentPistolAmmo => self.ammo[0] = value.round() as i32,
            SettingKey::CurrentShotgunAmmo => self.ammo[1] = value.round() as i32,
            SettingKey::CurrentChaingunAmmo => self.ammo[2] = value.round() as i32,
            SettingKey::CurrentRocketAmmo => self.ammo[3] = value.round() as i32,
            _ => {
                let old_value = self
                    .settings
                    .setting_value(key)
                    .expect("setting key has a field");
                self.settings
                    .set_setting_value(key, value)
                    .expect("setting key has a writable field");
                match key {
                    SettingKey::MaxHealth => {
                        self.health = self.health.min(value.round() as i32);
                        self.dead = self.health <= 0;
                    }
                    SettingKey::MaxArmor => self.armor = self.armor.min(value.round() as i32),
                    SettingKey::StartingPistolAmmo => self.ammo[0] = value.round() as i32,
                    SettingKey::ImpHealth | SettingKey::DemonHealth => {
                        let scale = value / old_value.max(1.0);
                        let imp = matches!(key, SettingKey::ImpHealth);
                        for enemy in &mut self.enemies {
                            if enemy.alive && (enemy.kind == 0) == imp {
                                enemy.hp *= scale;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub(crate) fn map_tile(&self, x: i32, y: i32) -> u8 {
        level_tile(self.level_index, x, y)
    }

    pub(crate) fn is_wall(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x >= MAP_WIDTH as i32 || y >= MAP_HEIGHT as i32 {
            return true;
        }
        self.wall_grid[y as usize * MAP_WIDTH + x as usize]
    }

    fn can_stand(&self, x: f32, y: f32) -> bool {
        let r = self.settings.player_radius;
        !self.is_wall((x - r).floor() as i32, (y - r).floor() as i32)
            && !self.is_wall((x + r).floor() as i32, (y - r).floor() as i32)
            && !self.is_wall((x - r).floor() as i32, (y + r).floor() as i32)
            && !self.is_wall((x + r).floor() as i32, (y + r).floor() as i32)
    }

    pub(crate) fn try_move(&self, x: f32, y: f32) -> (f32, f32) {
        let mut nx = self.x;
        let mut ny = self.y;
        if self.can_stand(x, self.y) {
            nx = x;
        }
        if self.can_stand(nx, y) {
            ny = y;
        }
        (nx, ny)
    }

    pub(crate) fn update(&mut self, dt: f32) {
        if self.paused || self.settings_open || self.dead || self.won {
            return;
        }
        self.time += dt;
        self.shot_timer = (self.shot_timer - dt).max(0.0);
        if self.visual_effects {
            self.recoil = (self.recoil - dt * 4.0).max(0.0);
            self.muzzle_timer = (self.muzzle_timer - dt).max(0.0);
            for effect in &mut self.hit_effects {
                effect.timer = (effect.timer - dt).max(0.0);
            }
            self.hit_effects.retain(|effect| effect.timer > 0.0);
            self.damage_flash = (self.damage_flash - dt).max(0.0);
            self.message_timer = (self.message_timer - dt).max(0.0);
        }

        let mut incoming_damage = 0;
        let walls = self.wall_grid;
        let player_x = self.x;
        let player_y = self.y;
        let enemy_start = if self.profiling {
            Some(Instant::now())
        } else {
            None
        };
        for enemy in &mut self.enemies {
            if !enemy.alive {
                continue;
            }
            enemy.attack_timer = (enemy.attack_timer - dt).max(0.0);
            enemy.hurt_timer = (enemy.hurt_timer - dt).max(0.0);
            let dx = player_x - enemy.x;
            let dy = player_y - enemy.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < self.settings.chase_range && dist > self.settings.enemy_stop_distance {
                let speed = if enemy.kind == 0 {
                    self.settings.imp_speed
                } else {
                    self.settings.demon_speed
                };
                let step = speed * dt / dist;
                let nx = enemy.x + dx * step;
                let ny = enemy.y + dy * step;
                if !wall_in(&walls, nx.floor() as i32, enemy.y.floor() as i32) {
                    enemy.x = nx;
                }
                if !wall_in(&walls, enemy.x.floor() as i32, ny.floor() as i32) {
                    enemy.y = ny;
                }
            }
            if dist < self.settings.attack_range
                && enemy.attack_timer <= 0.0
                && line_of_sight_in(&walls, player_x, player_y, enemy.x, enemy.y)
            {
                let damage = if enemy.kind == 0 {
                    self.settings.imp_damage
                } else {
                    self.settings.demon_damage
                };
                incoming_damage += damage.round() as i32;
                enemy.attack_timer = if enemy.kind == 0 {
                    self.settings.imp_attack_delay
                } else {
                    self.settings.demon_attack_delay
                };
            }
        }
        if let Some(start) = enemy_start {
            self.profile.enemy_ns += start.elapsed().as_nanos() as u64;
        }
        let danger_start = if self.profiling {
            Some(Instant::now())
        } else {
            None
        };
        if self.track_danger {
            let mut nearest_threat: Option<(f32, f32, f32)> = None;
            for enemy in &self.enemies {
                if !enemy.alive {
                    continue;
                }
                let dist = distance(player_x, player_y, enemy.x, enemy.y);
                if dist < self.settings.attack_range
                    && line_of_sight_in(&walls, player_x, player_y, enemy.x, enemy.y)
                    && nearest_threat.map_or(true, |(nearest_dist, _, _)| dist < nearest_dist)
                {
                    nearest_threat = Some((dist, enemy.x, enemy.y));
                }
            }
            if let Some((_, threat_x, threat_y)) = nearest_threat {
                self.danger = true;
                self.danger_angle =
                    wrap_angle((threat_y - player_y).atan2(threat_x - player_x) - self.angle);
            } else {
                self.danger = false;
            }
        }
        if let Some(start) = danger_start {
            self.profile.danger_ns += start.elapsed().as_nanos() as u64;
        }
        if incoming_damage > 0 {
            self.take_damage(incoming_damage);
        }

        let pickup_start = if self.profiling {
            Some(Instant::now())
        } else {
            None
        };
        for pickup in &mut self.pickups {
            if !pickup.active
                || distance(self.x, self.y, pickup.x, pickup.y) > self.settings.pickup_radius
            {
                continue;
            }
            pickup.active = false;
            match pickup.kind {
                PickupKind::Shotgun => {
                    self.unlocked[1] = true;
                    self.ammo[1] += self.settings.shotgun_pickup_ammo.round() as i32;
                    self.weapon = 1;
                    self.message = "SHOTGUN ACQUIRED".to_string();
                }
                PickupKind::Chaingun => {
                    self.unlocked[2] = true;
                    self.ammo[2] += self.settings.chaingun_pickup_ammo.round() as i32;
                    self.weapon = 2;
                    self.message = "CHAINGUN ACQUIRED".to_string();
                }
                PickupKind::RocketLauncher => {
                    self.unlocked[3] = true;
                    self.ammo[3] += self.settings.rocket_pickup_ammo.round() as i32;
                    self.weapon = 3;
                    self.message = "ROCKET LAUNCHER ACQUIRED".to_string();
                }
                PickupKind::Medkit => {
                    self.health = (self.health + self.settings.medkit_heal.round() as i32)
                        .min(self.settings.max_health.round() as i32);
                    self.message = "MEDIKIT +30".to_string();
                }
                PickupKind::Armor => {
                    self.armor = (self.armor + self.settings.armor_pickup.round() as i32)
                        .min(self.settings.max_armor.round() as i32);
                    self.message = "ARMOR +40".to_string();
                }
                PickupKind::Ammo => {
                    self.ammo[0] += self.settings.ammo_cache_pistol.round() as i32;
                    if self.unlocked[1] {
                        self.ammo[1] += self.settings.ammo_cache_shotgun.round() as i32;
                    }
                    if self.unlocked[2] {
                        self.ammo[2] += self.settings.ammo_cache_chaingun.round() as i32;
                    }
                    if self.unlocked[3] {
                        self.ammo[3] += self.settings.ammo_cache_rocket.round() as i32;
                    }
                    self.message = "AMMO CACHE".to_string();
                }
            }
            self.message_timer = 1.8;
        }
        if let Some(start) = pickup_start {
            self.profile.pickup_ns += start.elapsed().as_nanos() as u64;
        }
    }

    pub(crate) fn add_impact_effect(&mut self, x: f32, y: f32, kind: ImpactKind) {
        const HIT_DURATIONS: [f32; 4] = [0.18, 0.28, 0.14, 0.42];
        const IMPACT_DURATIONS: [f32; 4] = [0.14, 0.20, 0.12, 0.30];
        if self.hit_effects.len() == self.hit_effects.capacity() {
            self.hit_effects.remove(0);
        }
        let weapon = self.weapon.min(3);
        let duration = if matches!(kind, ImpactKind::Enemy) {
            HIT_DURATIONS[weapon]
        } else {
            IMPACT_DURATIONS[weapon]
        };
        self.hit_effects.push(HitEffect {
            x,
            y,
            weapon,
            kind,
            timer: duration,
            duration,
        });
    }

    pub(crate) fn restart(&mut self) {
        self.level_index = 0;
        self.health = self.settings.max_health.round() as i32;
        self.armor = 0;
        self.ammo = [self.settings.starting_pistol_ammo.round() as i32, 0, 0, 0];
        self.unlocked = [true, false, false, false];
        self.weapon = 0;
        self.dead = false;
        self.won = false;
        self.paused = false;
        self.load_level(0);
    }

    pub(crate) fn choose_weapon(&mut self, weapon: usize) {
        if weapon < self.unlocked.len() && self.unlocked[weapon] {
            self.weapon = weapon;
        }
    }

    pub(crate) fn cycle_weapon(&mut self, direction: i32) {
        for _ in 0..self.unlocked.len() {
            self.weapon = (self.weapon as i32 + direction).rem_euclid(4) as usize;
            if self.unlocked[self.weapon] {
                break;
            }
        }
    }

    pub(crate) fn fire(&mut self) {
        let (damage, range, cone, delay, ammo_cost) = match self.weapon {
            0 => (
                self.settings.pistol_damage,
                self.settings.pistol_range,
                self.settings.pistol_spread,
                self.settings.pistol_delay,
                self.settings.pistol_ammo_cost,
            ),
            1 => (
                self.settings.shotgun_damage,
                self.settings.shotgun_range,
                self.settings.shotgun_spread,
                self.settings.shotgun_delay,
                self.settings.shotgun_ammo_cost,
            ),
            2 => (
                self.settings.chaingun_damage,
                self.settings.chaingun_range,
                self.settings.chaingun_spread,
                self.settings.chaingun_delay,
                self.settings.chaingun_ammo_cost,
            ),
            _ => (
                self.settings.rocket_damage,
                self.settings.rocket_range,
                self.settings.rocket_spread,
                self.settings.rocket_delay,
                self.settings.rocket_ammo_cost,
            ),
        };
        let ammo_cost = ammo_cost.round() as i32;
        if self.ammo[self.weapon] < ammo_cost {
            self.message = "OUT OF AMMO".to_string();
            self.message_timer = 0.8;
            self.shot_timer = 0.22;
            return;
        }
        self.ammo[self.weapon] -= ammo_cost;
        self.shot_timer = delay;
        self.recoil = 1.0;
        self.muzzle_timer = 0.12;
        self.muzzle_weapon = self.weapon;
        let mut target = None;
        let mut best_dist = range;
        for (i, enemy) in self.enemies.iter().enumerate() {
            if !enemy.alive {
                continue;
            }
            let dx = enemy.x - self.x;
            let dy = enemy.y - self.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist >= best_dist {
                continue;
            }
            let diff = wrap_angle(dy.atan2(dx) - self.angle).abs();
            // Enemies have a visible width, so accept a shot that intersects
            // their hit area instead of requiring it to pass through the center.
            // This matters most up close, where the same world-space width
            // covers a much larger angle on screen.
            let enemy_hit_radius = 0.36;
            let hit_cone = cone + (enemy_hit_radius / dist.max(0.01)).atan();
            if diff <= hit_cone && self.line_of_sight(enemy.x, enemy.y) {
                best_dist = dist;
                target = Some(i);
            }
        }

        if let Some(index) = target {
            let hit_x = self.enemies[index].x;
            let hit_y = self.enemies[index].y;
            if self.weapon == 3 {
                for enemy in &mut self.enemies {
                    if enemy.alive
                        && distance(hit_x, hit_y, enemy.x, enemy.y)
                            < self.settings.rocket_blast_radius
                    {
                        enemy.hp -= damage;
                        enemy.hurt_timer = 0.18;
                        if enemy.hp <= 0.0 {
                            enemy.alive = false;
                            self.kills += 1;
                        }
                    }
                }
            } else {
                let falloff = if self.weapon == 1 {
                    (1.0 - best_dist / 15.0).max(0.55)
                } else {
                    1.0
                };
                let enemy = &mut self.enemies[index];
                enemy.hp -= damage * falloff;
                enemy.hurt_timer = 0.18;
                if enemy.hp <= 0.0 {
                    enemy.alive = false;
                    self.kills += 1;
                }
            }
            if self.visual_effects {
                self.add_impact_effect(hit_x, hit_y, ImpactKind::Enemy);
            }
        } else {
            let (impact_x, impact_y, kind) = self.find_shot_endpoint(range);
            if self.visual_effects {
                self.add_impact_effect(impact_x, impact_y, kind);
            }
        }
    }

    fn find_shot_endpoint(&self, range: f32) -> (f32, f32, ImpactKind) {
        let ray_x = self.angle.cos();
        let ray_y = self.angle.sin();
        let mut map_x = self.x.floor() as i32;
        let mut map_y = self.y.floor() as i32;
        let delta_x = if ray_x.abs() < f32::EPSILON {
            f32::INFINITY
        } else {
            (1.0 / ray_x).abs()
        };
        let delta_y = if ray_y.abs() < f32::EPSILON {
            f32::INFINITY
        } else {
            (1.0 / ray_y).abs()
        };
        let step_x = if ray_x < 0.0 { -1 } else { 1 };
        let step_y = if ray_y < 0.0 { -1 } else { 1 };
        let mut side_x = if ray_x < 0.0 {
            (self.x - map_x as f32) * delta_x
        } else {
            (map_x as f32 + 1.0 - self.x) * delta_x
        };
        let mut side_y = if ray_y < 0.0 {
            (self.y - map_y as f32) * delta_y
        } else {
            (map_y as f32 + 1.0 - self.y) * delta_y
        };

        for _ in 0..(MAP_WIDTH + MAP_HEIGHT + 8) {
            let distance_to_tile = if side_x < side_y {
                let distance_to_tile = side_x;
                side_x += delta_x;
                map_x += step_x;
                distance_to_tile
            } else {
                let distance_to_tile = side_y;
                side_y += delta_y;
                map_y += step_y;
                distance_to_tile
            };
            if distance_to_tile > range {
                break;
            }
            if self.is_wall(map_x, map_y) {
                let distance_to_surface = (distance_to_tile - 0.015).max(0.0);
                return (
                    self.x + ray_x * distance_to_surface,
                    self.y + ray_y * distance_to_surface,
                    ImpactKind::Wall,
                );
            }
        }

        (
            self.x + ray_x * range,
            self.y + ray_y * range,
            ImpactKind::Miss,
        )
    }

    fn take_damage(&mut self, damage: i32) {
        let absorbed = (damage as f32 * self.settings.armor_absorption).round() as i32;
        let absorbed = absorbed.min(self.armor);
        self.armor -= absorbed;
        self.health -= damage - absorbed;
        self.damage_flash = 0.20;
        if self.health <= 0 {
            self.health = 0;
            self.dead = true;
        }
    }

    fn line_of_sight(&self, x: f32, y: f32) -> bool {
        let dx = x - self.x;
        let dy = y - self.y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 0.25 {
            return true;
        }
        let steps = (dist * 8.0) as usize;
        for i in 1..steps {
            let t = i as f32 / steps as f32;
            if self.is_wall(
                (self.x + dx * t).floor() as i32,
                (self.y + dy * t).floor() as i32,
            ) {
                return false;
            }
        }
        true
    }
}
pub(crate) fn distance(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ((ax - bx) * (ax - bx) + (ay - by) * (ay - by)).sqrt()
}

fn wall_in(walls: &[bool; MAP_WIDTH * MAP_HEIGHT], x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || x >= MAP_WIDTH as i32 || y >= MAP_HEIGHT as i32 {
        return true;
    }
    walls[y as usize * MAP_WIDTH + x as usize]
}

fn line_of_sight_in(
    walls: &[bool; MAP_WIDTH * MAP_HEIGHT],
    from_x: f32,
    from_y: f32,
    to_x: f32,
    to_y: f32,
) -> bool {
    let dx = to_x - from_x;
    let dy = to_y - from_y;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 0.25 {
        return true;
    }
    let steps = (dist * 8.0) as usize;
    for i in 1..steps {
        let t = i as f32 / steps as f32;
        if wall_in(
            walls,
            (from_x + dx * t).floor() as i32,
            (from_y + dy * t).floor() as i32,
        ) {
            return false;
        }
    }
    true
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}
