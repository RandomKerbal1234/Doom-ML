#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DifficultyPreset {
    Easy,
    Medium,
    Hard,
    Custom,
}

impl DifficultyPreset {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Easy => "EASY",
            Self::Medium => "MEDIUM",
            Self::Hard => "HARD",
            Self::Custom => "CUSTOM",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum SettingKey {
    CurrentHealth,
    MaxHealth,
    CurrentArmor,
    MaxArmor,
    WalkSpeed,
    RunSpeed,
    TurnSpeed,
    RunTurnSpeed,
    PlayerRadius,
    ArmorAbsorption,
    CurrentPistolAmmo,
    StartingPistolAmmo,
    CurrentShotgunAmmo,
    CurrentChaingunAmmo,
    CurrentRocketAmmo,
    PistolDamage,
    PistolRange,
    PistolDelay,
    PistolSpread,
    PistolAmmoCost,
    ShotgunDamage,
    ShotgunRange,
    ShotgunDelay,
    ShotgunSpread,
    ShotgunAmmoCost,
    ChaingunDamage,
    ChaingunRange,
    ChaingunDelay,
    ChaingunSpread,
    ChaingunAmmoCost,
    RocketDamage,
    RocketRange,
    RocketDelay,
    RocketSpread,
    RocketAmmoCost,
    RocketBlastRadius,
    ImpHealth,
    DemonHealth,
    ImpSpeed,
    DemonSpeed,
    ChaseRange,
    AttackRange,
    EnemyStopDistance,
    ImpDamage,
    DemonDamage,
    ImpAttackDelay,
    DemonAttackDelay,
    MedkitHeal,
    ArmorPickup,
    PickupRadius,
    ShotgunPickupAmmo,
    ChaingunPickupAmmo,
    RocketPickupAmmo,
    AmmoCachePistol,
    AmmoCacheShotgun,
    AmmoCacheChaingun,
    AmmoCacheRocket,
}

#[derive(Clone, Copy)]
pub(crate) struct SettingDef {
    pub(crate) key: SettingKey,
    pub(crate) label: &'static str,
    pub(crate) step: f32,
    pub(crate) min: f32,
    pub(crate) max: f32,
    pub(crate) decimals: usize,
}

const PLAYER_SETTINGS: &[SettingDef] = &[
    SettingDef {
        key: SettingKey::CurrentHealth,
        label: "Current health",
        step: 5.0,
        min: 0.0,
        max: 300.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::MaxHealth,
        label: "Maximum health",
        step: 5.0,
        min: 25.0,
        max: 300.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::CurrentArmor,
        label: "Current armor",
        step: 5.0,
        min: 0.0,
        max: 300.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::MaxArmor,
        label: "Maximum armor",
        step: 5.0,
        min: 0.0,
        max: 300.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::WalkSpeed,
        label: "Walk speed (tiles/s)",
        step: 0.1,
        min: 0.5,
        max: 8.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::RunSpeed,
        label: "Run speed (tiles/s)",
        step: 0.1,
        min: 0.5,
        max: 12.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::TurnSpeed,
        label: "Turn speed (rad/s)",
        step: 0.1,
        min: 0.5,
        max: 8.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::RunTurnSpeed,
        label: "Run turn speed",
        step: 0.1,
        min: 0.5,
        max: 10.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::PlayerRadius,
        label: "Player collision radius",
        step: 0.01,
        min: 0.08,
        max: 0.45,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::ArmorAbsorption,
        label: "Armor damage absorption",
        step: 0.05,
        min: 0.0,
        max: 1.0,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::StartingPistolAmmo,
        label: "Starting pistol ammo",
        step: 5.0,
        min: 0.0,
        max: 500.0,
        decimals: 0,
    },
];

const WEAPON_SETTINGS: &[SettingDef] = &[
    SettingDef {
        key: SettingKey::PistolDamage,
        label: "Pistol damage",
        step: 5.0,
        min: 1.0,
        max: 300.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::PistolRange,
        label: "Pistol range (tiles)",
        step: 0.5,
        min: 1.0,
        max: 30.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::PistolDelay,
        label: "Pistol fire delay (s)",
        step: 0.05,
        min: 0.05,
        max: 3.0,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::PistolSpread,
        label: "Pistol spread (rad)",
        step: 0.01,
        min: 0.0,
        max: 0.5,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::PistolAmmoCost,
        label: "Pistol ammo per shot",
        step: 1.0,
        min: 0.0,
        max: 10.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::ShotgunDamage,
        label: "Shotgun damage",
        step: 5.0,
        min: 1.0,
        max: 300.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::ShotgunRange,
        label: "Shotgun range (tiles)",
        step: 0.5,
        min: 1.0,
        max: 30.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::ShotgunDelay,
        label: "Shotgun fire delay (s)",
        step: 0.05,
        min: 0.05,
        max: 3.0,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::ShotgunSpread,
        label: "Shotgun spread (rad)",
        step: 0.01,
        min: 0.0,
        max: 0.8,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::ShotgunAmmoCost,
        label: "Shotgun ammo per shot",
        step: 1.0,
        min: 0.0,
        max: 10.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::ChaingunDamage,
        label: "Chaingun damage",
        step: 2.0,
        min: 1.0,
        max: 200.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::ChaingunRange,
        label: "Chaingun range (tiles)",
        step: 0.5,
        min: 1.0,
        max: 30.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::ChaingunDelay,
        label: "Chaingun fire delay (s)",
        step: 0.01,
        min: 0.03,
        max: 2.0,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::ChaingunSpread,
        label: "Chaingun spread (rad)",
        step: 0.01,
        min: 0.0,
        max: 0.5,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::ChaingunAmmoCost,
        label: "Chaingun ammo per shot",
        step: 1.0,
        min: 0.0,
        max: 10.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::RocketDamage,
        label: "Rocket damage",
        step: 5.0,
        min: 1.0,
        max: 500.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::RocketRange,
        label: "Rocket range (tiles)",
        step: 0.5,
        min: 1.0,
        max: 30.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::RocketDelay,
        label: "Rocket fire delay (s)",
        step: 0.05,
        min: 0.05,
        max: 3.0,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::RocketSpread,
        label: "Rocket spread (rad)",
        step: 0.01,
        min: 0.0,
        max: 0.5,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::RocketAmmoCost,
        label: "Rocket ammo per shot",
        step: 1.0,
        min: 0.0,
        max: 10.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::RocketBlastRadius,
        label: "Rocket blast radius",
        step: 0.1,
        min: 0.2,
        max: 6.0,
        decimals: 1,
    },
];

const DEMON_SETTINGS: &[SettingDef] = &[
    SettingDef {
        key: SettingKey::ImpHealth,
        label: "Imp health",
        step: 5.0,
        min: 1.0,
        max: 500.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::DemonHealth,
        label: "Demon health",
        step: 5.0,
        min: 1.0,
        max: 500.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::ImpSpeed,
        label: "Imp speed (tiles/s)",
        step: 0.05,
        min: 0.0,
        max: 5.0,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::DemonSpeed,
        label: "Demon speed (tiles/s)",
        step: 0.05,
        min: 0.0,
        max: 5.0,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::ChaseRange,
        label: "Demon chase range",
        step: 0.5,
        min: 1.0,
        max: 30.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::AttackRange,
        label: "Demon attack range",
        step: 0.5,
        min: 0.5,
        max: 30.0,
        decimals: 1,
    },
    SettingDef {
        key: SettingKey::EnemyStopDistance,
        label: "Demon stop distance",
        step: 0.05,
        min: 0.3,
        max: 2.0,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::ImpDamage,
        label: "Imp attack damage",
        step: 1.0,
        min: 0.0,
        max: 100.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::DemonDamage,
        label: "Demon attack damage",
        step: 1.0,
        min: 0.0,
        max: 100.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::ImpAttackDelay,
        label: "Imp attack delay (s)",
        step: 0.05,
        min: 0.1,
        max: 10.0,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::DemonAttackDelay,
        label: "Demon attack delay (s)",
        step: 0.05,
        min: 0.1,
        max: 10.0,
        decimals: 2,
    },
];

const SUPPLY_SETTINGS: &[SettingDef] = &[
    SettingDef {
        key: SettingKey::CurrentPistolAmmo,
        label: "Current pistol ammo",
        step: 5.0,
        min: 0.0,
        max: 5000.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::CurrentShotgunAmmo,
        label: "Current shotgun ammo",
        step: 2.0,
        min: 0.0,
        max: 5000.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::CurrentChaingunAmmo,
        label: "Current chaingun ammo",
        step: 10.0,
        min: 0.0,
        max: 5000.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::CurrentRocketAmmo,
        label: "Current rockets",
        step: 1.0,
        min: 0.0,
        max: 5000.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::MedkitHeal,
        label: "Medkit healing",
        step: 5.0,
        min: 0.0,
        max: 100.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::ArmorPickup,
        label: "Armor pickup amount",
        step: 5.0,
        min: 0.0,
        max: 100.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::PickupRadius,
        label: "Pickup reach (tiles)",
        step: 0.05,
        min: 0.25,
        max: 1.5,
        decimals: 2,
    },
    SettingDef {
        key: SettingKey::ShotgunPickupAmmo,
        label: "Shotgun pickup shells",
        step: 1.0,
        min: 0.0,
        max: 100.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::ChaingunPickupAmmo,
        label: "Chaingun pickup ammo",
        step: 5.0,
        min: 0.0,
        max: 500.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::RocketPickupAmmo,
        label: "Rocket pickup amount",
        step: 1.0,
        min: 0.0,
        max: 100.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::AmmoCachePistol,
        label: "Ammo cache bullets",
        step: 5.0,
        min: 0.0,
        max: 500.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::AmmoCacheShotgun,
        label: "Ammo cache shells",
        step: 1.0,
        min: 0.0,
        max: 100.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::AmmoCacheChaingun,
        label: "Ammo cache chaingun",
        step: 5.0,
        min: 0.0,
        max: 500.0,
        decimals: 0,
    },
    SettingDef {
        key: SettingKey::AmmoCacheRocket,
        label: "Ammo cache rockets",
        step: 1.0,
        min: 0.0,
        max: 100.0,
        decimals: 0,
    },
];

pub(crate) fn settings_for_category(category: usize) -> &'static [SettingDef] {
    match category {
        0 => PLAYER_SETTINGS,
        1 => WEAPON_SETTINGS,
        2 => DEMON_SETTINGS,
        _ => SUPPLY_SETTINGS,
    }
}

pub(crate) fn setting_category_name(category: usize) -> &'static str {
    match category {
        0 => "PLAYER",
        1 => "WEAPONS",
        2 => "DEMONS",
        _ => "SUPPLIES",
    }
}

#[derive(Clone)]
pub(crate) struct GameSettings {
    pub(crate) max_health: f32,
    pub(crate) max_armor: f32,
    pub(crate) walk_speed: f32,
    pub(crate) run_speed: f32,
    pub(crate) turn_speed: f32,
    pub(crate) run_turn_speed: f32,
    pub(crate) player_radius: f32,
    pub(crate) armor_absorption: f32,
    pub(crate) starting_pistol_ammo: f32,
    pub(crate) pistol_damage: f32,
    pub(crate) pistol_range: f32,
    pub(crate) pistol_delay: f32,
    pub(crate) pistol_spread: f32,
    pub(crate) pistol_ammo_cost: f32,
    pub(crate) shotgun_damage: f32,
    pub(crate) shotgun_range: f32,
    pub(crate) shotgun_delay: f32,
    pub(crate) shotgun_spread: f32,
    pub(crate) shotgun_ammo_cost: f32,
    pub(crate) chaingun_damage: f32,
    pub(crate) chaingun_range: f32,
    pub(crate) chaingun_delay: f32,
    pub(crate) chaingun_spread: f32,
    pub(crate) chaingun_ammo_cost: f32,
    pub(crate) rocket_damage: f32,
    pub(crate) rocket_range: f32,
    pub(crate) rocket_delay: f32,
    pub(crate) rocket_spread: f32,
    pub(crate) rocket_ammo_cost: f32,
    pub(crate) rocket_blast_radius: f32,
    pub(crate) imp_health: f32,
    pub(crate) demon_health: f32,
    pub(crate) imp_speed: f32,
    pub(crate) demon_speed: f32,
    pub(crate) chase_range: f32,
    pub(crate) attack_range: f32,
    pub(crate) enemy_stop_distance: f32,
    pub(crate) imp_damage: f32,
    pub(crate) demon_damage: f32,
    pub(crate) imp_attack_delay: f32,
    pub(crate) demon_attack_delay: f32,
    pub(crate) medkit_heal: f32,
    pub(crate) armor_pickup: f32,
    pub(crate) pickup_radius: f32,
    pub(crate) shotgun_pickup_ammo: f32,
    pub(crate) chaingun_pickup_ammo: f32,
    pub(crate) rocket_pickup_ammo: f32,
    pub(crate) ammo_cache_pistol: f32,
    pub(crate) ammo_cache_shotgun: f32,
    pub(crate) ammo_cache_chaingun: f32,
    pub(crate) ammo_cache_rocket: f32,
}

macro_rules! setting_field_accessors {
    ($($key:ident => $field:ident),+ $(,)?) => {
        pub(crate) fn setting_value(&self, key: SettingKey) -> Option<f32> {
            match key {
                $(SettingKey::$key => Some(self.$field),)+
                _ => None,
            }
        }

        pub(crate) fn set_setting_value(&mut self, key: SettingKey, value: f32) -> Option<()> {
            match key {
                $(SettingKey::$key => {
                    self.$field = value;
                    Some(())
                },)+
                _ => None,
            }
        }
    };
}

impl GameSettings {
    pub(crate) fn medium() -> Self {
        Self {
            max_health: 100.0,
            max_armor: 100.0,
            walk_speed: 2.7,
            run_speed: 4.3,
            turn_speed: 2.1,
            run_turn_speed: 2.8,
            player_radius: 0.19,
            armor_absorption: 0.5,
            starting_pistol_ammo: 50.0,
            pistol_damage: 30.0,
            pistol_range: 9.5,
            pistol_delay: 0.32,
            pistol_spread: 0.045,
            pistol_ammo_cost: 1.0,
            shotgun_damage: 64.0,
            shotgun_range: 7.0,
            shotgun_delay: 0.60,
            shotgun_spread: 0.15,
            shotgun_ammo_cost: 1.0,
            chaingun_damage: 18.0,
            chaingun_range: 10.5,
            chaingun_delay: 0.105,
            chaingun_spread: 0.035,
            chaingun_ammo_cost: 1.0,
            rocket_damage: 95.0,
            rocket_range: 10.5,
            rocket_delay: 0.82,
            rocket_spread: 0.04,
            rocket_ammo_cost: 1.0,
            rocket_blast_radius: 1.7,
            imp_health: 55.0,
            demon_health: 80.0,
            imp_speed: 0.72,
            demon_speed: 0.94,
            chase_range: 8.5,
            attack_range: 6.5,
            enemy_stop_distance: 0.82,
            imp_damage: 7.0,
            demon_damage: 10.0,
            imp_attack_delay: 1.25,
            demon_attack_delay: 1.55,
            medkit_heal: 30.0,
            armor_pickup: 40.0,
            pickup_radius: 0.52,
            shotgun_pickup_ammo: 12.0,
            chaingun_pickup_ammo: 70.0,
            rocket_pickup_ammo: 6.0,
            ammo_cache_pistol: 25.0,
            ammo_cache_shotgun: 6.0,
            ammo_cache_chaingun: 30.0,
            ammo_cache_rocket: 2.0,
        }
    }

    pub(crate) fn preset(preset: DifficultyPreset) -> Self {
        let mut settings = Self::medium();
        match preset {
            DifficultyPreset::Easy => {
                settings.max_health = 125.0;
                settings.max_armor = 125.0;
                settings.armor_absorption = 0.6;
                settings.imp_health = 40.0;
                settings.demon_health = 60.0;
                settings.imp_speed = 0.56;
                settings.demon_speed = 0.72;
                settings.chase_range = 7.5;
                settings.attack_range = 5.0;
                settings.imp_damage = 5.0;
                settings.demon_damage = 7.0;
                settings.imp_attack_delay = 1.6;
                settings.demon_attack_delay = 1.9;
                settings.medkit_heal = 40.0;
                settings.armor_pickup = 50.0;
            }
            DifficultyPreset::Hard => {
                settings.max_health = 90.0;
                settings.max_armor = 75.0;
                settings.armor_absorption = 0.4;
                settings.imp_health = 80.0;
                settings.demon_health = 110.0;
                settings.imp_speed = 0.9;
                settings.demon_speed = 1.1;
                settings.chase_range = 10.0;
                settings.attack_range = 7.8;
                settings.imp_damage = 11.0;
                settings.demon_damage = 15.0;
                settings.imp_attack_delay = 0.95;
                settings.demon_attack_delay = 1.15;
                settings.medkit_heal = 20.0;
                settings.armor_pickup = 30.0;
                settings.ammo_cache_pistol = 15.0;
                settings.ammo_cache_shotgun = 4.0;
                settings.ammo_cache_chaingun = 20.0;
                settings.ammo_cache_rocket = 1.0;
            }
            DifficultyPreset::Medium | DifficultyPreset::Custom => {}
        }
        settings
    }

    setting_field_accessors! {
        MaxHealth => max_health,
        MaxArmor => max_armor,
        WalkSpeed => walk_speed,
        RunSpeed => run_speed,
        TurnSpeed => turn_speed,
        RunTurnSpeed => run_turn_speed,
        PlayerRadius => player_radius,
        ArmorAbsorption => armor_absorption,
        StartingPistolAmmo => starting_pistol_ammo,
        PistolDamage => pistol_damage,
        PistolRange => pistol_range,
        PistolDelay => pistol_delay,
        PistolSpread => pistol_spread,
        PistolAmmoCost => pistol_ammo_cost,
        ShotgunDamage => shotgun_damage,
        ShotgunRange => shotgun_range,
        ShotgunDelay => shotgun_delay,
        ShotgunSpread => shotgun_spread,
        ShotgunAmmoCost => shotgun_ammo_cost,
        ChaingunDamage => chaingun_damage,
        ChaingunRange => chaingun_range,
        ChaingunDelay => chaingun_delay,
        ChaingunSpread => chaingun_spread,
        ChaingunAmmoCost => chaingun_ammo_cost,
        RocketDamage => rocket_damage,
        RocketRange => rocket_range,
        RocketDelay => rocket_delay,
        RocketSpread => rocket_spread,
        RocketAmmoCost => rocket_ammo_cost,
        RocketBlastRadius => rocket_blast_radius,
        ImpHealth => imp_health,
        DemonHealth => demon_health,
        ImpSpeed => imp_speed,
        DemonSpeed => demon_speed,
        ChaseRange => chase_range,
        AttackRange => attack_range,
        EnemyStopDistance => enemy_stop_distance,
        ImpDamage => imp_damage,
        DemonDamage => demon_damage,
        ImpAttackDelay => imp_attack_delay,
        DemonAttackDelay => demon_attack_delay,
        MedkitHeal => medkit_heal,
        ArmorPickup => armor_pickup,
        PickupRadius => pickup_radius,
        ShotgunPickupAmmo => shotgun_pickup_ammo,
        ChaingunPickupAmmo => chaingun_pickup_ammo,
        RocketPickupAmmo => rocket_pickup_ammo,
        AmmoCachePistol => ammo_cache_pistol,
        AmmoCacheShotgun => ammo_cache_shotgun,
        AmmoCacheChaingun => ammo_cache_chaingun,
        AmmoCacheRocket => ammo_cache_rocket,
    }
}
