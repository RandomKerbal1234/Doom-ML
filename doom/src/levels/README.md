# Level files

Each `.lvl` file describes one level. Edit these files in `src/levels/` before compiling. The build automatically embeds every `.lvl` file in the executable in filename order. After compiling, the game does not read or write these source files, so changing them requires another build to take effect.

Files are played in filename order, so the numeric prefix controls campaign order (`01_...`, `02_...`, and so on). Give each level a unique `code`.

```ini
code=E1M1
title=HANGAR 13
spawn=2.5,2.5,0.0
exit=14.5,10.5

[map]
################
#......#.......#
#......#..##...#
#......#.......#
#..##..#.......#
#......#.......#
#..............#
#...#..........#
#...#..........#
#..............#
#.............X#
################

[enemies]
5.5,3.5,imp
10.5,8.5,demon

[pickups]
8.5,6.5,shotgun
5.5,6.5,ammo
12.5,8.5,medkit
```

The map is exactly 16 columns by 12 rows. `#`, `B`, and `M` are solid wall tiles; `.` and `X` are walkable floor (`X` is a visual marker only—the exit position comes from `exit=`). Coordinates are map positions, usually tile centers such as `2.5,2.5`. The spawn has an angle in radians.

Enemy types are `imp` and `demon`. Pickup types are `shotgun`, `chaingun`, `rocket_launcher`, `medkit`, `armor`, and `ammo`. Lines beginning with `;` or `//` are comments. Keep the spawn, exit, enemies, and pickups on open tiles.

In the pause/help screen, use Up/Down to select a level and Enter to start it. Press Q to quickplay from the first level and continue through the campaign via the exits. Escape, Space, or F1 resumes the paused game.
