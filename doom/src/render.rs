use crate::game::{Game, ImpactKind, Sprite, SpriteKind};
use crate::levels::{PickupKind, levels};
use crate::{HEIGHT, HUD_TOP, VIEW_HEIGHT, WIDTH};

const FOV_PLANE: f32 = 0.66;

impl Game {
    pub(crate) fn render(&mut self) {
        self.render_floor_and_ceiling();
        self.cast_walls();
        self.render_sprites();
        self.render_hit_effects();
        self.render_weapon();
        self.render_danger_indicator();
        self.render_hud();
    }

    fn render_danger_indicator(&mut self) {
        if !self.danger {
            return;
        }

        let pulse = 0.72 + self.time.mul_add(7.0, 0.0).sin().abs() * 0.28;
        let red = (190.0 + 65.0 * pulse) as u8;
        self.fill_rect(124, 6, 72, 14, (50, 10, 9));
        self.fill_rect(124, 6, 72, 1, (red, 40, 26));
        self.fill_rect(124, 19, 72, 1, (red, 40, 26));
        self.text_3x5("DEMON IN RANGE", 132, 10, (255, 224, 186));

        let (x, arrow) = if self.danger_angle < -0.25 {
            (5, "<")
        } else if self.danger_angle > 0.25 {
            (WIDTH as i32 - 9, ">")
        } else {
            return;
        };
        self.text_3x5(arrow, x, 82, (red, 45, 32));
    }

    fn render_floor_and_ceiling(&mut self) {
        let half = VIEW_HEIGHT / 2;
        for y in 0..half {
            let shade = 20 + (y * 11 / half) as u8;
            let band = if y % 24 == 0 { 5 } else { 0 };
            for x in 0..WIDTH {
                let side = if (x / 40) % 2 == 0 { 0 } else { 2 };
                self.set_pixel(
                    x as i32,
                    y as i32,
                    shade + band + side,
                    shade + band,
                    shade + band + 3,
                );
            }
        }

        let dir_x = self.angle.cos();
        let dir_y = self.angle.sin();
        let plane_x = -dir_y * FOV_PLANE;
        let plane_y = dir_x * FOV_PLANE;
        let left_x = dir_x - plane_x;
        let left_y = dir_y - plane_y;
        let right_x = dir_x + plane_x;
        let right_y = dir_y + plane_y;

        for y in (half + 1)..VIEW_HEIGHT {
            let row_dist = (0.5 * VIEW_HEIGHT as f32) / (y - half) as f32;
            let step_x = row_dist * (right_x - left_x) / WIDTH as f32;
            let step_y = row_dist * (right_y - left_y) / WIDTH as f32;
            let mut floor_x = self.x + row_dist * left_x;
            let mut floor_y = self.y + row_dist * left_y;
            let shade = (1.0 / (1.0 + row_dist * 0.075)).clamp(0.28, 0.88);
            for x in 0..WIDTH {
                let cell_x = floor_x.floor() as i32;
                let cell_y = floor_y.floor() as i32;
                let fx = ((floor_x - floor_x.floor()) * 8.0) as i32;
                let fy = ((floor_y - floor_y.floor()) * 8.0) as i32;
                let grid = fx == 0 || fy == 0;
                let checker = (cell_x ^ cell_y) & 1;
                let base = if checker == 0 { 76.0 } else { 65.0 };
                let texture = if grid {
                    0.66
                } else if (fx + fy) % 7 == 0 {
                    0.86
                } else {
                    1.0
                };
                let tile = self.map_tile(cell_x, cell_y);
                let stain = if tile == b'B' { 1.08 } else { 1.0 };
                let value = (base * shade * texture * stain) as u8;
                self.set_pixel(
                    x as i32,
                    y as i32,
                    value,
                    (value as f32 * 0.98) as u8,
                    (value as f32 * 0.92) as u8,
                );
                floor_x += step_x;
                floor_y += step_y;
            }
        }
    }

    fn cast_walls(&mut self) {
        let dir_x = self.angle.cos();
        let dir_y = self.angle.sin();
        let plane_x = -dir_y * FOV_PLANE;
        let plane_y = dir_x * FOV_PLANE;

        for screen_x in 0..WIDTH {
            let camera_x = 2.0 * screen_x as f32 / WIDTH as f32 - 1.0;
            let ray_x = dir_x + plane_x * camera_x;
            let ray_y = dir_y + plane_y * camera_x;
            let mut map_x = self.x.floor() as i32;
            let mut map_y = self.y.floor() as i32;
            let delta_x = if ray_x.abs() < 0.00001 {
                1.0e30
            } else {
                (1.0 / ray_x).abs()
            };
            let delta_y = if ray_y.abs() < 0.00001 {
                1.0e30
            } else {
                (1.0 / ray_y).abs()
            };
            let (step_x, mut side_x) = if ray_x < 0.0 {
                (-1, (self.x - map_x as f32) * delta_x)
            } else {
                (1, (map_x as f32 + 1.0 - self.x) * delta_x)
            };
            let (step_y, mut side_y) = if ray_y < 0.0 {
                (-1, (self.y - map_y as f32) * delta_y)
            } else {
                (1, (map_y as f32 + 1.0 - self.y) * delta_y)
            };
            let mut side = 0;
            let mut tile = b'#';
            for _ in 0..40 {
                if side_x < side_y {
                    side_x += delta_x;
                    map_x += step_x;
                    side = 0;
                } else {
                    side_y += delta_y;
                    map_y += step_y;
                    side = 1;
                }
                tile = self.map_tile(map_x, map_y);
                if matches!(tile, b'#' | b'B' | b'M') {
                    break;
                }
            }
            let distance = if side == 0 {
                (map_x as f32 - self.x + (1 - step_x) as f32 * 0.5) / ray_x
            } else {
                (map_y as f32 - self.y + (1 - step_y) as f32 * 0.5) / ray_y
            }
            .max(0.05);
            self.z_buffer[screen_x] = distance;
            let wall_height = (VIEW_HEIGHT as f32 / distance) as i32;
            let top = (VIEW_HEIGHT as i32 / 2 - wall_height / 2).max(0);
            let bottom = (VIEW_HEIGHT as i32 / 2 + wall_height / 2).min(VIEW_HEIGHT as i32 - 1);
            let mut wall_x = if side == 0 {
                self.y + distance * ray_y
            } else {
                self.x + distance * ray_x
            };
            wall_x -= wall_x.floor();
            let mut tex_x = (wall_x * 32.0) as i32;
            if (side == 0 && ray_x > 0.0) || (side == 1 && ray_y < 0.0) {
                tex_x = 31 - tex_x;
            }

            let (base_r, base_g, base_b) = match tile {
                b'B' => (133.0, 57.0, 37.0),
                b'M' => (55.0, 104.0, 128.0),
                _ => (116.0, 112.0, 108.0),
            };
            let fog = (1.0 / (1.0 + distance * 0.095)).clamp(0.36, 1.0);
            let face = if side == 1 { 0.74 } else { 1.0 };
            for y in top..=bottom {
                let tex_y = ((y - (VIEW_HEIGHT as i32 / 2 - wall_height / 2)) * 32
                    / wall_height.max(1))
                    & 31;
                let mortar = if tile == b'M' {
                    tex_y % 16 == 0 || tex_x % 16 == 0
                } else {
                    let row = tex_y / 8;
                    tex_y % 8 == 0 || (tex_x + if row % 2 == 0 { 0 } else { 16 }) % 32 < 2
                };
                let fleck = (tex_x * 13 + tex_y * 7 + map_x * 3 + map_y * 5) & 15 == 0;
                let texture = if mortar {
                    0.52
                } else if fleck {
                    1.13
                } else {
                    1.0
                };
                let light = fog * face * texture;
                self.set_pixel(
                    screen_x as i32,
                    y,
                    (base_r * light) as u8,
                    (base_g * light) as u8,
                    (base_b * light) as u8,
                );
            }
        }
    }

    fn render_sprites(&mut self) {
        let mut sprites = Vec::with_capacity(self.enemies.len() + self.pickups.len() + 1);
        let dir_x = self.angle.cos();
        let dir_y = self.angle.sin();
        let plane_x = -dir_y * FOV_PLANE;
        let plane_y = dir_x * FOV_PLANE;
        let inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y);

        for enemy in &self.enemies {
            if enemy.alive {
                sprites.push(Sprite {
                    x: enemy.x,
                    y: enemy.y,
                    kind: SpriteKind::Enemy(enemy.kind, enemy.hurt_timer > 0.0),
                    size: 0.86,
                    depth: 0.0,
                });
            }
        }
        for pickup in &self.pickups {
            if pickup.active {
                sprites.push(Sprite {
                    x: pickup.x,
                    y: pickup.y,
                    kind: SpriteKind::Pickup(pickup.kind),
                    size: 0.44,
                    depth: 0.0,
                });
            }
        }
        let exit = self.current_level().exit;
        sprites.push(Sprite {
            x: exit.0,
            y: exit.1,
            kind: SpriteKind::Exit,
            size: 0.72,
            depth: 0.0,
        });

        for sprite in &mut sprites {
            let rel_x = sprite.x - self.x;
            let rel_y = sprite.y - self.y;
            sprite.depth = inv_det * (-plane_y * rel_x + plane_x * rel_y);
        }
        sprites.sort_by(|a, b| b.depth.total_cmp(&a.depth));

        for sprite in sprites {
            if sprite.depth <= 0.12 {
                continue;
            }
            let rel_x = sprite.x - self.x;
            let rel_y = sprite.y - self.y;
            let transform_x = inv_det * (dir_y * rel_x - dir_x * rel_y);
            let transform_y = sprite.depth;
            let center_x = (WIDTH as f32 / 2.0 * (1.0 + transform_x / transform_y)) as i32;
            let projected_h = (VIEW_HEIGHT as f32 / transform_y * sprite.size) as i32;
            if projected_h <= 0 || projected_h > 1800 {
                continue;
            }
            let aspect = if matches!(sprite.kind, SpriteKind::Enemy(..)) {
                0.80
            } else {
                0.72
            };
            let projected_w = (projected_h as f32 * aspect) as i32;
            let top = VIEW_HEIGHT as i32 / 2 - projected_h / 2;
            let left = center_x - projected_w / 2;
            let right = center_x + projected_w / 2;
            for stripe in left.max(0)..right.min(WIDTH as i32) {
                if transform_y >= self.z_buffer[stripe as usize] {
                    continue;
                }
                let u = ((stripe - left) * 16 / projected_w.max(1)).clamp(0, 15) as usize;
                for y in top.max(0)..(top + projected_h).min(VIEW_HEIGHT as i32) {
                    let v = ((y - top) * 16 / projected_h.max(1)).clamp(0, 15) as usize;
                    if let Some((r, g, b)) = sprite_pixel(sprite.kind, u, v, self.time) {
                        let shade = (1.0 / (1.0 + transform_y * 0.07)).clamp(0.48, 1.0);
                        self.set_pixel(
                            stripe,
                            y,
                            (r as f32 * shade) as u8,
                            (g as f32 * shade) as u8,
                            (b as f32 * shade) as u8,
                        );
                    }
                }
            }
        }
    }

    fn render_weapon(&mut self) {
        let lift = (self.recoil * 5.0) as i32;
        let y = VIEW_HEIGHT as i32 - 24 + lift;
        let skin = (172, 111, 78);
        let metal = (91, 91, 91);
        let dark = (38, 39, 40);
        let light = (145, 145, 139);
        self.fill_rect(139, y + 12, 42, 13, skin);
        match self.weapon {
            0 => {
                self.fill_rect(150, y + 2, 21, 18, dark);
                self.fill_rect(153, y, 15, 11, metal);
                self.fill_rect(155, y - 5, 11, 8, light);
                self.fill_rect(155, y + 15, 11, 8, dark);
            }
            1 => {
                self.fill_rect(143, y + 4, 35, 16, dark);
                self.fill_rect(146, y, 29, 13, metal);
                self.fill_rect(147, y - 5, 9, 8, light);
                self.fill_rect(164, y - 5, 9, 8, light);
                self.fill_rect(154, y + 14, 12, 10, dark);
            }
            2 => {
                self.fill_rect(143, y + 6, 35, 16, dark);
                self.fill_rect(148, y - 1, 24, 16, metal);
                self.fill_rect(148, y - 7, 8, 11, light);
                self.fill_rect(163, y - 7, 8, 11, light);
                self.fill_rect(153, y + 17, 13, 9, dark);
            }
            _ => {
                self.fill_rect(139, y + 4, 43, 18, dark);
                self.fill_rect(144, y, 33, 13, (76, 102, 72));
                self.fill_rect(148, y - 6, 25, 9, light);
                self.fill_rect(154, y + 16, 14, 10, dark);
            }
        }
        self.render_muzzle_flash(y);
        self.fill_rect(157, 83, 6, 1, (230, 210, 165));
        self.fill_rect(159, 81, 1, 5, (230, 210, 165));
    }

    fn render_muzzle_flash(&mut self, weapon_y: i32) {
        if self.muzzle_timer <= 0.0 {
            return;
        }
        let cx = 160;
        let cy = weapon_y - 10;
        let pulse = (self.muzzle_timer / 0.12).clamp(0.0, 1.0);
        match self.muzzle_weapon {
            0 => {
                // Pistol: a tight, star-shaped yellow muzzle flash.
                let width = (2.0 + pulse * 2.0) as i32;
                self.fill_rect(cx - width, cy - 1, width * 2 + 1, 3, (255, 176, 57));
                self.fill_rect(cx - 1, cy - 4, 3, 9, (255, 237, 158));
                self.fill_rect(cx - 5, cy - 4, 2, 2, (255, 207, 105));
                self.fill_rect(cx + 4, cy + 3, 2, 2, (255, 207, 105));
            }
            1 => {
                // Shotgun: a larger layered blast with branching tongues and hot sparks.
                self.fill_rect(cx - 13, cy - 3, 27, 7, (201, 67, 29));
                self.fill_rect(cx - 10, cy - 6, 21, 12, (245, 119, 39));
                self.fill_rect(cx - 7, cy - 7, 15, 13, (255, 183, 67));
                self.fill_rect(cx - 4, cy - 6, 9, 11, (255, 241, 163));
                self.fill_rect(cx - 12, cy - 9, 4, 4, (255, 202, 85));
                self.fill_rect(cx + 8, cy - 8, 4, 4, (255, 215, 99));
                self.fill_rect(cx - 15, cy + 3, 5, 2, (255, 151, 48));
                self.fill_rect(cx + 10, cy + 4, 5, 2, (255, 151, 48));
                self.set_pixel(cx - 8, cy - 11, 255, 235, 151);
                self.set_pixel(cx + 7, cy - 12, 255, 230, 135);
                self.set_pixel(cx - 13, cy + 8, 255, 183, 72);
                self.set_pixel(cx + 14, cy + 7, 255, 183, 72);
            }
            2 => {
                // Chaingun: twin green-white flashes at alternating barrels.
                let offset = ((self.time * 42.0) as i32 & 1) * 4;
                for barrel_x in [cx - 7 + offset, cx + 7 - offset] {
                    self.fill_rect(barrel_x - 2, cy - 3, 5, 7, (77, 232, 133));
                    self.fill_rect(barrel_x - 1, cy - 5, 3, 9, (202, 255, 197));
                }
            }
            _ => {
                // Rocket launcher: a long red-orange flame and bright exhaust core.
                let length = (8.0 + pulse * 5.0) as i32;
                for row in 0..length {
                    let half_width = (row / 3 + 1).min(5);
                    let flame_y = cy - row;
                    self.fill_rect(
                        cx - half_width,
                        flame_y,
                        half_width * 2 + 1,
                        1,
                        if row < length / 2 {
                            (255, 111, 38)
                        } else {
                            (203, 53, 34)
                        },
                    );
                }
                self.fill_rect(cx - 2, cy - 5, 5, 7, (255, 238, 153));
                self.fill_rect(cx - 6, cy - length - 1, 3, 2, (83, 79, 72));
                self.fill_rect(cx + 4, cy - length - 4, 3, 2, (83, 79, 72));
            }
        }
    }

    fn render_hit_effects(&mut self) {
        if self.hit_effects.is_empty() {
            return;
        }
        let dir_x = self.angle.cos();
        let dir_y = self.angle.sin();
        let plane_x = -dir_y * FOV_PLANE;
        let plane_y = dir_x * FOV_PLANE;
        let inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y);

        for index in 0..self.hit_effects.len() {
            let effect = self.hit_effects[index];
            let rel_x = effect.x - self.x;
            let rel_y = effect.y - self.y;
            let depth = inv_det * (-plane_y * rel_x + plane_x * rel_y);
            if depth <= 0.12 {
                continue;
            }
            let transform_x = inv_det * (dir_y * rel_x - dir_x * rel_y);
            let center_x = (WIDTH as f32 / 2.0 * (1.0 + transform_x / depth)).round() as i32;
            if !(0..WIDTH as i32).contains(&center_x) {
                continue;
            }
            let wall_impact = matches!(effect.kind, ImpactKind::Wall);
            let wall_depth = self.z_buffer[center_x as usize];
            if (wall_impact && depth > wall_depth + 0.08) || (!wall_impact && depth >= wall_depth) {
                continue;
            }

            let progress = 1.0 - effect.timer / effect.duration;
            let pixel_scale = (4.0 / depth).clamp(1.0, 3.0);
            let radius = if effect.weapon == 0 {
                (0.5 + progress * 2.0 * pixel_scale).round() as i32
            } else {
                (1.0 + progress * 4.0 * pixel_scale).round() as i32
            };
            let center_y = VIEW_HEIGHT as i32 / 2 + (10.0 / depth) as i32;
            match effect.kind {
                ImpactKind::Enemy => match effect.weapon {
                    0 => {
                        // Pistol: a tiny, brief cross of hot sparks.
                        self.fill_rect(
                            center_x - radius,
                            center_y,
                            radius * 2 + 1,
                            1,
                            (255, 173, 59),
                        );
                        self.fill_rect(
                            center_x,
                            center_y - radius,
                            1,
                            radius * 2 + 1,
                            (255, 235, 145),
                        );
                        self.set_pixel(center_x - radius, center_y - radius, 255, 212, 104);
                        self.set_pixel(center_x + radius, center_y + radius, 255, 212, 104);
                    }
                    1 => {
                        // Shotgun: a row of separate pellet bursts across the target.
                        for pellet in -2..=2 {
                            let pellet_x = center_x + pellet * radius;
                            let pellet_y = center_y + (pellet.abs() % 2);
                            let color = if pellet == 0 {
                                (255, 230, 132)
                            } else {
                                (255, 135, 54)
                            };
                            self.fill_rect(pellet_x - 1, pellet_y - 1, 3, 3, color);
                            self.set_pixel(pellet_x, pellet_y - 2, 255, 245, 196);
                        }
                    }
                    2 => {
                        // Chaingun: fast green electrical-looking impact spokes.
                        let spokes = [
                            (-radius, -radius),
                            (0, -radius - 1),
                            (radius, -radius),
                            (-radius - 1, 0),
                            (radius + 1, 0),
                            (-radius, radius),
                            (0, radius + 1),
                            (radius, radius),
                        ];
                        for (spoke_x, spoke_y) in spokes {
                            self.fill_rect(
                                center_x + spoke_x,
                                center_y + spoke_y,
                                2,
                                2,
                                (103, 255, 153),
                            );
                        }
                        self.set_pixel(center_x, center_y, 230, 255, 209);
                    }
                    _ => {
                        // The sprite grows from half-size to 1.5x a demon's projected height.
                        let demon_height = VIEW_HEIGHT as f32 * 0.86 / depth;
                        let diameter = (demon_height * (0.55 + progress * 0.95))
                            .round()
                            .clamp(16.0, 256.0) as i32;
                        self.render_rocket_blast(center_x, center_y, diameter, progress);
                    }
                },
                ImpactKind::Wall => {
                    self.render_wall_impact(center_x, center_y, radius, effect.weapon);
                }
                ImpactKind::Miss => {
                    self.render_miss_effect(center_x, center_y, radius, effect.weapon, progress);
                }
            }
        }
    }

    fn render_wall_impact(&mut self, x: i32, y: i32, radius: i32, weapon: usize) {
        match weapon {
            0 => {
                // Pistol ricochet: a small star of yellow-white sparks.
                self.fill_rect(x - radius, y, radius * 2 + 1, 1, (255, 190, 74));
                self.fill_rect(x, y - radius, 1, radius * 2 + 1, (255, 239, 171));
                self.set_pixel(x - radius, y - radius, 255, 220, 116);
                self.set_pixel(x + radius, y + radius, 255, 220, 116);
            }
            1 => {
                // Shotgun ricochet: a wide fan of orange fragments.
                for fragment in -3..=3 {
                    let fragment_x = x + fragment * (radius.max(1));
                    let fragment_y = y + (fragment.abs() % 2) * radius.max(1);
                    self.fill_rect(fragment_x, fragment_y, 2, 2, (255, 145, 55));
                    self.set_pixel(fragment_x + 1, fragment_y - 2, 255, 221, 135);
                }
            }
            2 => {
                // Chaingun ricochet: sharp green sparks in alternating directions.
                for (spark_x, spark_y) in [
                    (-radius, -radius),
                    (radius, -radius),
                    (-radius - 1, 1),
                    (radius + 1, -1),
                    (-radius, radius),
                    (radius, radius),
                ] {
                    self.fill_rect(x + spark_x, y + spark_y, 2, 2, (103, 255, 153));
                }
                self.set_pixel(x, y, 225, 255, 218);
            }
            _ => {
                let demon_height = VIEW_HEIGHT as f32 * 0.86
                    / self.z_buffer[x.clamp(0, WIDTH as i32 - 1) as usize].max(0.12);
                let diameter = (demon_height * 1.35).round().clamp(16.0, 256.0) as i32;
                self.render_rocket_blast(x, y, diameter, 0.72);
            }
        }
    }

    fn render_rocket_blast(&mut self, x: i32, y: i32, diameter: i32, progress: f32) {
        // A 16x16 hand-authored pixel sprite: smoke, irregular flame silhouette,
        // molten outer ring, gold body, and a white-hot center. Two frames flicker
        // the silhouette without needing external image assets.
        const FRAME_A: [&str; 16] = [
            "....ss..ss......",
            "...ssRRss.......",
            "..sRROORRr......",
            ".sRROYYORRr......",
            ".RROYYYYORr..s...",
            "sRROYWWYORRr.ss..",
            "sRROWWWWYORRrr...",
            "RROYYWWWWYORRr...",
            "RROYYYYWWYYORr...",
            "sRROYYYYYYYORr...",
            ".RROOYYYYYYORr...",
            ".sRROOYYYYORr....",
            "..sRROOOOORr..s..",
            "...ssRRRRRr..ss..",
            "....srrrrr.......",
            ".....rr..........",
        ];
        const FRAME_B: [&str; 16] = [
            ".....ss.ss.......",
            "....ssRRRr.......",
            "...sRROORRr..s....",
            "..sRROYYORRr.ss...",
            ".sRROYYYYORRr.....",
            ".RROYYWWYORRr.....",
            "sRROYWWWWYORRr.s..",
            "RROOYYWWWWYORRr...",
            "RROOYYYYWWYYORr...",
            "sRROOYYYYYYYORr...",
            ".sRROOYYYYYYORr...",
            "..sRROOOYYYORr....",
            "...sRRROOOORr..s..",
            "....ssRRRRrr.ss...",
            ".....srrrrr.......",
            "......rr..........",
        ];
        let frame = if (self.time * 18.0) as i32 % 2 == 0 {
            &FRAME_A
        } else {
            &FRAME_B
        };
        let cell = (diameter / 16).max(1);
        let left = x - cell * 8;
        let top = y - cell * 8;
        for (row, pixels) in frame.iter().enumerate() {
            for (col, pixel) in pixels.bytes().enumerate() {
                let color: Option<(u8, u8, u8)> = match pixel {
                    b's' if progress > 0.45 => Some((76, 70, 68)),
                    b's' => Some((112, 78, 58)),
                    b'R' => Some((190, 45, 25)),
                    b'r' => Some((241, 78, 27)),
                    b'O' => Some((255, 137, 35)),
                    b'Y' => Some((255, 205, 65)),
                    b'W' => Some(if progress < 0.35 {
                        (255, 250, 196)
                    } else {
                        (255, 232, 141)
                    }),
                    _ => None,
                };
                if let Some((red, green, blue)) = color {
                    let flicker: u8 =
                        if pixel == b'W' && (row + col + (self.time * 30.0) as usize) % 5 == 0 {
                            18
                        } else {
                            0
                        };
                    self.fill_rect(
                        left + col as i32 * cell,
                        top + row as i32 * cell,
                        cell,
                        cell,
                        (red, green.saturating_add(flicker), blue),
                    );
                }
            }
        }
    }

    fn render_miss_effect(&mut self, x: i32, y: i32, radius: i32, weapon: usize, progress: f32) {
        match weapon {
            0 => {
                // Pistol miss: a brief amber streak with a fading spark at its tip.
                self.fill_rect(x - 1, y + 2, 3, radius.max(2), (207, 127, 47));
                self.fill_rect(x, y, 1, 3, (255, 219, 119));
            }
            1 => {
                // Shotgun miss: several spread-out pellets fade separately.
                for pellet in -2..=2 {
                    let pellet_x = x + pellet * radius.max(1);
                    self.fill_rect(pellet_x, y + pellet.abs() % 2, 2, 2, (255, 153, 61));
                }
            }
            2 => {
                // Chaingun miss: a string of quick green tracer dots.
                for dot in 0..4 {
                    if (dot as f32 / 4.0) <= progress + 0.35 {
                        self.fill_rect(x, y + dot * radius.max(1), 2, 2, (91, 235, 133));
                    }
                }
            }
            _ => {
                // Rocket miss: a short ember trail ending in two drifting smoke pixels.
                self.fill_rect(x - 1, y + 1, 3, (radius * 2).max(3), (222, 75, 38));
                self.fill_rect(x - 4, y - 2, 2, 2, (87, 82, 76));
                self.fill_rect(x + 3, y - 5, 2, 2, (102, 94, 83));
            }
        }
    }

    fn render_hud(&mut self) {
        self.fill_rect(
            0,
            HUD_TOP as i32,
            WIDTH as i32,
            (HEIGHT - HUD_TOP) as i32,
            (24, 22, 20),
        );
        self.fill_rect(0, HUD_TOP as i32, WIDTH as i32, 2, (128, 82, 52));
        self.fill_rect(8, 182, 304, 1, (57, 52, 47));
        let max_health = self.settings.max_health.round().max(1.0) as i32;
        let max_armor = self.settings.max_armor.round().max(1.0) as i32;
        let health = self.health.clamp(0, max_health);
        let armor = self.armor.clamp(0, max_armor);
        let ammo = if self.weapon == 0 {
            self.ammo[0]
        } else {
            self.ammo[self.weapon]
        };
        self.text_3x5("HP", 8, 176, (238, 92, 66));
        self.text_3x5(&format!("{:03}", health), 19, 176, (238, 222, 190));
        self.fill_rect(34, 177, 46, 5, (63, 39, 33));
        self.fill_rect(35, 178, health * 44 / max_health, 3, (174, 45, 37));
        self.text_3x5("ARMOR", 88, 176, (82, 174, 184));
        self.text_3x5(&format!("{:03}", armor), 113, 176, (218, 222, 206));
        self.text_3x5("AMMO", 159, 176, (224, 177, 69));
        self.text_3x5(&format!("{:03}", ammo.max(0)), 183, 176, (238, 222, 190));
        self.text_3x5(&levels()[self.level_index].code, 278, 176, (196, 181, 157));
        self.text_3x5(weapon_name(self.weapon), 8, 188, (205, 197, 176));
        if self.message_timer > 0.0 {
            let msg = self.message.to_ascii_uppercase();
            let x = (WIDTH as i32 - msg.len() as i32 * 4) / 2;
            self.text_3x5(&msg, x.max(94), 188, (255, 213, 117));
        } else {
            self.text_3x5("ARROWS TURN  E USE EXIT", 170, 188, (132, 126, 116));
        }
        if self.damage_flash > 0.0 {
            let intensity = (self.damage_flash * 520.0) as u8;
            for x in 0..WIDTH {
                self.set_pixel(x as i32, 0, intensity, 0, 0);
                self.set_pixel(x as i32, 1, intensity / 2, 0, 0);
            }
        }
    }

    fn text_3x5(&mut self, text: &str, x: i32, y: i32, color: (u8, u8, u8)) {
        let mut cursor = x;
        for ch in text.chars() {
            let rows = glyph(ch);
            for (row, bits) in rows.iter().enumerate() {
                for col in 0..3 {
                    if bits & (1 << (2 - col)) != 0 {
                        self.set_pixel(cursor + col, y + row as i32, color.0, color.1, color.2);
                    }
                }
            }
            cursor += 4;
        }
    }

    fn set_pixel(&mut self, x: i32, y: i32, r: u8, g: u8, b: u8) {
        if x < 0 || y < 0 || x >= WIDTH as i32 || y >= HEIGHT as i32 {
            return;
        }
        let index = (y as usize * WIDTH + x as usize) * 4;
        self.pixels[index] = r;
        self.pixels[index + 1] = g;
        self.pixels[index + 2] = b;
        self.pixels[index + 3] = 255;
    }

    fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: (u8, u8, u8)) {
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(WIDTH as i32);
        let y1 = (y + h).min(HEIGHT as i32);
        for py in y0..y1 {
            for px in x0..x1 {
                self.set_pixel(px, py, color.0, color.1, color.2);
            }
        }
    }
}
fn weapon_name(index: usize) -> &'static str {
    match index {
        0 => "PISTOL",
        1 => "SHOTGUN",
        2 => "CHAINGUN",
        _ => "ROCKET LAUNCHER",
    }
}

fn sprite_pixel(kind: SpriteKind, x: usize, y: usize, time: f32) -> Option<(u8, u8, u8)> {
    match kind {
        SpriteKind::Enemy(enemy_kind, hurt) => {
            const DEMON: [&str; 12] = [
                "..R......R..",
                ".RRR....RRR.",
                "GGGGGGGGGGGG",
                "GGGGGGGGGGGG",
                "GGWWGGGGWWGG",
                "GGWWGGGGWWGG",
                ".GGGGGGGGGG.",
                "..GGGGGGGG..",
                "..GGGGGGGG..",
                ".GGGGGGGGGG.",
                "..GGG..GGG..",
                "RRRR....RRRR",
            ];
            let sprite_y = y * DEMON.len() / 16;
            let sprite_x = x * 12 / 16;
            let c = DEMON
                .get(sprite_y)
                .and_then(|row| row.as_bytes().get(sprite_x))
                .copied()
                .unwrap_or(b'.');
            if c == b'.' {
                return None;
            }
            if hurt {
                return Some((255, 226, 194));
            }
            let body = if enemy_kind == 0 {
                (104, 139, 77)
            } else {
                (145, 70, 57)
            };
            Some(match c {
                b'R' => (177, 62, 43),
                b'W' => (255, 220, 120),
                b'G' => body,
                _ => (102, 50, 39),
            })
        }
        SpriteKind::Exit => {
            let dx = x as f32 - 7.5;
            let dy = y as f32 - 7.5;
            let r = (dx * dx + dy * dy).sqrt();
            if r > 7.0 || r < 3.0 {
                return None;
            }
            let pulse = ((time * 5.0 + r) as i32 & 1) == 0;
            Some(if pulse { (47, 235, 126) } else { (24, 126, 80) })
        }
        SpriteKind::Pickup(pickup) => {
            if x < 3 || x > 12 || y < 3 || y > 12 {
                return None;
            }
            let edge = x == 3 || x == 12 || y == 3 || y == 12;
            let inside = (x + y + (time * 2.0) as usize) % 4 == 0;
            let base = match pickup {
                PickupKind::Shotgun => (215, 162, 69),
                PickupKind::Chaingun => (89, 184, 119),
                PickupKind::RocketLauncher => (207, 84, 57),
                PickupKind::Medkit => (230, 222, 190),
                PickupKind::Armor => (91, 159, 201),
                PickupKind::Ammo => (193, 142, 61),
            };
            Some(if edge {
                (44, 40, 35)
            } else if inside {
                (244, 229, 179)
            } else {
                base
            })
        }
    }
}

fn glyph(ch: char) -> [u8; 5] {
    match ch {
        'A' => [2, 5, 7, 5, 5],
        'B' => [6, 5, 6, 5, 6],
        'C' => [3, 4, 4, 4, 3],
        'D' => [6, 5, 5, 5, 6],
        'E' => [7, 4, 6, 4, 7],
        'F' => [7, 4, 6, 4, 4],
        'G' => [3, 4, 5, 5, 3],
        'H' => [5, 5, 7, 5, 5],
        'I' => [7, 2, 2, 2, 7],
        'J' => [1, 1, 1, 5, 2],
        'K' => [5, 5, 6, 5, 5],
        'L' => [4, 4, 4, 4, 7],
        'M' => [5, 7, 7, 5, 5],
        'N' => [5, 7, 7, 7, 5],
        'O' => [2, 5, 5, 5, 2],
        'P' => [6, 5, 6, 4, 4],
        'Q' => [2, 5, 5, 3, 1],
        'R' => [6, 5, 6, 5, 5],
        'S' => [3, 4, 2, 1, 6],
        'T' => [7, 2, 2, 2, 2],
        'U' => [5, 5, 5, 5, 7],
        'V' => [5, 5, 5, 5, 2],
        'W' => [5, 5, 7, 7, 5],
        'X' => [5, 5, 2, 5, 5],
        'Y' => [5, 5, 2, 2, 2],
        'Z' => [7, 1, 2, 4, 7],
        '0' => [7, 5, 5, 5, 7],
        '1' => [2, 6, 2, 2, 7],
        '2' => [6, 1, 2, 4, 7],
        '3' => [6, 1, 3, 1, 6],
        '4' => [5, 5, 7, 1, 1],
        '5' => [7, 4, 6, 1, 6],
        '6' => [3, 4, 6, 5, 2],
        '7' => [7, 1, 2, 2, 2],
        '8' => [2, 5, 2, 5, 2],
        '9' => [2, 5, 3, 1, 6],
        ':' => [0, 2, 0, 2, 0],
        '/' => [1, 1, 2, 4, 4],
        '-' => [0, 0, 7, 0, 0],
        '+' => [0, 2, 7, 2, 0],
        '<' => [1, 2, 4, 2, 1],
        '>' => [4, 2, 1, 2, 4],
        '.' => [0, 0, 0, 0, 2],
        ' ' => [0; 5],
        _ => [0; 5],
    }
}
