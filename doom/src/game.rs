use crate::levels::{
    Level, MAP_HEIGHT, MAP_WIDTH, PickupKind, has_line_of_sight, level_tile, levels, tile_is_wall,
};
use crate::settings::{DifficultyPreset, GameSettings, SettingKey, settings_for_category};
use crate::{HEIGHT, WIDTH};
use macroquad::prelude::*;

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
    pub(crate) pixels: Vec<u8>,
    pub(crate) z_buffer: [f32; WIDTH],
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
            pixels: vec![0; WIDTH * HEIGHT * 4],
            z_buffer: [0.0; WIDTH],
        };
        game.ammo[0] = game.settings.starting_pistol_ammo.round() as i32;
        game.load_level(0);
        game
    }

    fn load_level(&mut self, index: usize) {
        self.level_index = index;
        self.selected_level = index;
        self.danger = false;
        let level = &levels()[index];
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
    }

    pub(crate) fn current_level(&self) -> &Level {
        &levels()[self.level_index]
    }

    fn start_new_run(&mut self, first_level: usize) {
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
        self.load_level(first_level);
    }

    fn toggle_settings(&mut self) {
        if self.settings_open {
            self.settings_open = false;
            self.paused = self.settings_return_paused;
        } else {
            self.settings_return_paused = self.paused;
            self.settings_open = true;
        }
    }

    fn handle_settings_input(&mut self) {
        if is_key_pressed(KeyCode::Escape) {
            self.toggle_settings();
            return;
        }
        if is_key_pressed(KeyCode::Key1) {
            self.apply_preset(DifficultyPreset::Easy);
            return;
        }
        if is_key_pressed(KeyCode::Key2) {
            self.apply_preset(DifficultyPreset::Medium);
            return;
        }
        if is_key_pressed(KeyCode::Key3) {
            self.apply_preset(DifficultyPreset::Hard);
            return;
        }

        if is_key_pressed(KeyCode::Tab) || is_key_pressed(KeyCode::PageDown) {
            self.settings_category = (self.settings_category + 1) % 4;
            self.settings_selection = 0;
        } else if is_key_pressed(KeyCode::PageUp) {
            self.settings_category = (self.settings_category + 3) % 4;
            self.settings_selection = 0;
        }

        let count = settings_for_category(self.settings_category).len();
        if is_key_pressed(KeyCode::Down) {
            self.settings_selection = (self.settings_selection + 1) % count;
        }
        if is_key_pressed(KeyCode::Up) {
            self.settings_selection = (self.settings_selection + count - 1) % count;
        }
        let direction = if is_key_pressed(KeyCode::Right) {
            1.0
        } else if is_key_pressed(KeyCode::Left) {
            -1.0
        } else {
            0.0
        };
        if direction != 0.0 {
            let definition = settings_for_category(self.settings_category)[self.settings_selection];
            let old_value = self.setting_value(definition.key);
            let value =
                (old_value + definition.step * direction).clamp(definition.min, definition.max);
            self.write_setting_value(definition.key, value);
            self.difficulty = DifficultyPreset::Custom;
        }
    }

    fn apply_preset(&mut self, preset: DifficultyPreset) {
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

    fn write_setting_value(&mut self, key: SettingKey, value: f32) {
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
        matches!(self.map_tile(x, y), b'#' | b'B' | b'M')
    }

    fn can_stand(&self, x: f32, y: f32) -> bool {
        let r = self.settings.player_radius;
        !self.is_wall((x - r).floor() as i32, (y - r).floor() as i32)
            && !self.is_wall((x + r).floor() as i32, (y - r).floor() as i32)
            && !self.is_wall((x - r).floor() as i32, (y + r).floor() as i32)
            && !self.is_wall((x + r).floor() as i32, (y + r).floor() as i32)
    }

    fn try_move(&self, x: f32, y: f32) -> (f32, f32) {
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

    pub(crate) fn handle_input(&mut self, dt: f32) {
        let shift_down = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
        if shift_down && is_key_pressed(KeyCode::P) {
            self.toggle_settings();
            return;
        }
        if self.settings_open {
            self.handle_settings_input();
            return;
        }

        if is_key_pressed(KeyCode::F1) {
            if !self.dead && !self.won {
                self.paused = !self.paused;
            }
        }

        if self.paused {
            if is_key_pressed(KeyCode::F10) {
                macroquad::miniquad::window::quit();
                return;
            }
            if is_key_pressed(KeyCode::Up) {
                self.selected_level = (self.selected_level + levels().len() - 1) % levels().len();
            }
            if is_key_pressed(KeyCode::Down) {
                self.selected_level = (self.selected_level + 1) % levels().len();
            }
            if is_key_pressed(KeyCode::Q) {
                self.start_new_run(0);
            } else if is_key_pressed(KeyCode::Enter) {
                self.start_new_run(self.selected_level);
            } else if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Space) {
                self.paused = false;
            }
            return;
        }

        if self.dead || self.won {
            if is_key_pressed(KeyCode::R) || is_key_pressed(KeyCode::Enter) {
                self.restart();
            }
            return;
        }

        if is_key_pressed(KeyCode::Escape) {
            self.paused = true;
            return;
        }

        if is_key_pressed(KeyCode::Key1) {
            self.choose_weapon(0);
        }
        if is_key_pressed(KeyCode::Key2) {
            self.choose_weapon(1);
        }
        if is_key_pressed(KeyCode::Key3) {
            self.choose_weapon(2);
        }
        if is_key_pressed(KeyCode::Key4) {
            self.choose_weapon(3);
        }
        let (_, wheel_y) = mouse_wheel();
        if wheel_y > 0.0 {
            self.cycle_weapon(1);
        }
        if wheel_y < 0.0 {
            self.cycle_weapon(-1);
        }

        let turn = (is_key_down(KeyCode::Right) as i8 - is_key_down(KeyCode::Left) as i8) as f32;
        let running = shift_down;
        let turn_speed = if running {
            self.settings.run_turn_speed
        } else {
            self.settings.turn_speed
        };
        self.angle = (self.angle + turn * turn_speed * dt).rem_euclid(std::f32::consts::TAU);

        let forward = (is_key_down(KeyCode::W) as i8 - is_key_down(KeyCode::S) as i8) as f32;
        let strafe = (is_key_down(KeyCode::D) as i8 - is_key_down(KeyCode::A) as i8) as f32;
        let mut vx = self.angle.cos() * forward - self.angle.sin() * strafe;
        let mut vy = self.angle.sin() * forward + self.angle.cos() * strafe;
        let magnitude = (vx * vx + vy * vy).sqrt();
        if magnitude > 1.0 {
            vx /= magnitude;
            vy /= magnitude;
        }
        let speed = if running {
            self.settings.run_speed
        } else {
            self.settings.walk_speed
        };
        let (nx, ny) = self.try_move(self.x + vx * speed * dt, self.y + vy * speed * dt);
        self.x = nx;
        self.y = ny;

        if is_key_pressed(KeyCode::E) {
            let exit = self.current_level().exit;
            if distance(self.x, self.y, exit.0, exit.1) < 1.35 {
                if self.level_index + 1 == levels().len() {
                    self.won = true;
                } else {
                    self.load_level(self.level_index + 1);
                }
            } else {
                self.message = "FIND THE GREEN EXIT PORTAL".to_string();
                self.message_timer = 1.0;
            }
        }

        if (is_key_down(KeyCode::Space)
            || is_key_down(KeyCode::LeftControl)
            || is_key_down(KeyCode::RightControl)
            || is_mouse_button_down(MouseButton::Left))
            && self.shot_timer <= 0.0
        {
            self.fire();
        }
    }

    pub(crate) fn update(&mut self, dt: f32) {
        if self.paused || self.settings_open || self.dead || self.won {
            return;
        }
        self.time += dt;
        self.shot_timer = (self.shot_timer - dt).max(0.0);
        self.recoil = (self.recoil - dt * 4.0).max(0.0);
        self.muzzle_timer = (self.muzzle_timer - dt).max(0.0);
        for effect in &mut self.hit_effects {
            effect.timer = (effect.timer - dt).max(0.0);
        }
        self.hit_effects.retain(|effect| effect.timer > 0.0);
        self.damage_flash = (self.damage_flash - dt).max(0.0);
        self.message_timer = (self.message_timer - dt).max(0.0);

        let mut incoming_damage = 0;
        let level_index = self.level_index;
        let player_x = self.x;
        let player_y = self.y;
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
                if !tile_is_wall(level_tile(
                    level_index,
                    nx.floor() as i32,
                    enemy.y.floor() as i32,
                )) {
                    enemy.x = nx;
                }
                if !tile_is_wall(level_tile(
                    level_index,
                    enemy.x.floor() as i32,
                    ny.floor() as i32,
                )) {
                    enemy.y = ny;
                }
            }
            if dist < self.settings.attack_range
                && enemy.attack_timer <= 0.0
                && has_line_of_sight(level_index, player_x, player_y, enemy.x, enemy.y)
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
        let mut nearest_threat: Option<(f32, f32, f32)> = None;
        for enemy in &self.enemies {
            if !enemy.alive {
                continue;
            }
            let dist = distance(player_x, player_y, enemy.x, enemy.y);
            if dist < self.settings.attack_range
                && has_line_of_sight(level_index, player_x, player_y, enemy.x, enemy.y)
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
        if incoming_damage > 0 {
            self.take_damage(incoming_damage);
        }

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
    }

    fn restart(&mut self) {
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

    fn choose_weapon(&mut self, weapon: usize) {
        if weapon < self.unlocked.len() && self.unlocked[weapon] {
            self.weapon = weapon;
        }
    }

    fn cycle_weapon(&mut self, direction: i32) {
        for _ in 0..self.unlocked.len() {
            self.weapon = (self.weapon as i32 + direction).rem_euclid(4) as usize;
            if self.unlocked[self.weapon] {
                break;
            }
        }
    }

    fn fire(&mut self) {
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
                }
            }
            self.add_impact_effect(hit_x, hit_y, ImpactKind::Enemy);
        } else {
            let (impact_x, impact_y, kind) = self.find_shot_endpoint(range);
            self.add_impact_effect(impact_x, impact_y, kind);
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
fn distance(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ((ax - bx) * (ax - bx) + (ay - by) * (ay - by)).sqrt()
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}
