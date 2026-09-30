"""Builds the MuJoCo scene: Menagerie's Panda plus a table, three bins and three trash objects."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from meshes import can_mesh, crumpled_paper_mesh
from setup_assets import panda_path

CLASS_NAMES = ("can", "bottle", "paper")

# Bin centres. Bin i takes class i. Each bin is 20 cm square inside, rim at z=0.14.
BIN_XY = np.array([[0.57, -0.40], [0.70, 0.0], [0.57, 0.40]])
BIN_COLORS = ("0.85 0.3 0.2 1", "0.2 0.5 0.85 1", "0.3 0.7 0.35 1")
# Drop target: 10 cm above the correct bin's rim.
DROP_HEIGHT = 0.24

# Object sizes, used by the shapes below and by scripted_sort.
OBJECT_RADIUS = np.array([0.03, 0.025, 0.025])
OBJECT_HEIGHT = np.array([0.12, 0.14, 0.062])
OBJECT_FRICTION = np.array([0.7, 0.6, 1.0])
# Unused objects rest on the floor under the table.
PARK_XY = np.array([[0.3, 0.3], [0.45, 0.3], [0.6, 0.3]])
FLOOR_Z = -0.55

# Bin walls, relative to the bin centre: (name, pos, half-size).
BIN_PARTS = (
    ("floor", "0 0 0.01", "0.11 0.11 0.01"),
    ("left", "0 -0.105 0.08", "0.11 0.005 0.06"),
    ("right", "0 0.105 0.08", "0.11 0.005 0.06"),
    ("back", "0.105 0 0.08", "0.005 0.1 0.06"),
    ("front", "-0.105 0 0.08", "0.005 0.1 0.06"),
)

# Origin is each object's bottom. Geoms with contype="0" conaffinity="0" mass="0" are visual only.
# Group 3 is hidden in the viewer, so those geoms collide but you see the mesh instead.
# Each object's collision masses add up to its total mass (can 14 g, bottle 10 g, paper 5 g).
r, h = OBJECT_RADIUS[0], OBJECT_HEIGHT[0]
lid = h - 0.003
br, bh = OBJECT_RADIUS[1], OBJECT_HEIGHT[1]
pr, ph = OBJECT_RADIUS[2], OBJECT_HEIGHT[2]
SHAPES = [
    # Can: one hidden collision cylinder, a mesh body, a red label, and a ring-pull on the lid.
    f"""<geom type="cylinder" size="{r} {h / 2}" pos="0 0 {h / 2}" group="3" rgba="0.75 0.77 0.8 1" mass="0.014"/>
    <geom type="mesh" mesh="can" rgba="0.78 0.8 0.83 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="cylinder" size="{r + 0.0002} {0.25 * h}" pos="0 0 {h / 2}" rgba="0.85 0.2 0.12 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="ellipsoid" size="0.0065 0.0045 0.0002" pos="-0.009 0 {lid + 0.0001}" rgba="0.5 0.51 0.55 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="cylinder" size="0.0018 0.0006" pos="0 0 {lid + 0.0005}" rgba="0.7 0.72 0.75 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="box" size="0.0075 0.0045 0.0004" pos="0.005 0 {lid + 0.0005}" rgba="0.85 0.87 0.9 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="ellipsoid" size="0.003 0.0022 0.0002" pos="0.008 0 {lid + 0.0009}" rgba="0.3 0.3 0.33 1" contype="0" conaffinity="0" mass="0"/>""",
    # Bottle: body, shoulder, neck and cap collide; the rest is decoration.
    f"""<geom type="cylinder" size="{br} {0.31 * bh}" pos="0 0 {0.31 * bh}" rgba="0.3 0.6 0.9 1" mass="0.007"/>
    <geom type="ellipsoid" size="{br} {br} {0.138 * bh}" pos="0 0 {0.62 * bh}" rgba="0.3 0.6 0.9 1" mass="0.002"/>
    <geom type="cylinder" size="{0.46 * br} {0.155 * bh}" pos="0 0 {0.828 * bh}" rgba="0.3 0.6 0.9 1" mass="0.0008"/>
    <geom type="cylinder" size="{0.54 * br} {0.0345 * bh}" pos="0 0 {0.9655 * bh}" rgba="0.1 0.25 0.65 1" mass="0.0002"/>
    <geom type="ellipsoid" size="{0.923 * br} {0.923 * br} {0.207 * bh}" pos="0 0 {0.655 * bh}" rgba="0.3 0.6 0.9 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="cylinder" size="{0.596 * br} 0.0012" pos="0 0 {0.917 * bh}" rgba="0.1 0.25 0.65 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="cylinder" size="{br + 0.0005} {0.083 * bh}" pos="0 0 {0.469 * bh}" rgba="0.1 0.25 0.65 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="cylinder" size="{br + 0.0002} 0.001" pos="0 0 {0.138 * bh}" rgba="0.2 0.45 0.8 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="cylinder" size="{br + 0.0002} 0.001" pos="0 0 {0.207 * bh}" rgba="0.2 0.45 0.8 1" contype="0" conaffinity="0" mass="0"/>""",
    # Paper: a hidden box (flat base) and ellipsoid collide; the crumpled mesh is what you see.
    f"""<geom type="box" size="{0.767 * pr} {0.733 * pr} {0.129 * ph}" pos="0 0 {0.129 * ph}" group="3" rgba="0.85 0.82 0.71 1" mass="0.0014"/>
    <geom type="ellipsoid" size="{0.933 * pr} {0.9 * pr} {0.419 * ph}" pos="0 0 {0.565 * ph}" group="3" rgba="0.85 0.82 0.71 1" mass="0.0036"/>
    <geom type="mesh" mesh="crumpled_paper" rgba="0.93 0.91 0.84 1" contype="0" conaffinity="0" mass="0"/>""",
]


def add_robot_tuning(root):
    """Physics and servo settings that keep grasps stable. Changing these can make objects slip."""
    # noslip_iterations: extra solver pass that removes contact slip. Without it the 5 g paper
    # slides out of the fingers during the sideways carry.
    root.find("option").attrib.update(
        timestep="0.002",
        integrator="implicitfast",
        cone="elliptic",
        iterations="30",
        noslip_iterations="3",
    )
    # Softer arm servos keep contact gentle; with Menagerie's stiff ones the arm squeezed the
    # paper out of the grasp. A stiffer gripper squeezes harder at the same opening.
    # for actuator, kp, kd in zip(
    #     root.find("actuator")[:7],
    #     [600, 600, 500, 500, 250, 250, 200],
    #     [50, 50, 40, 40, 20, 20, 15],
    # ):
    #     actuator.set("gainprm", str(kp))
    #     actuator.set("biasprm", f"0 {-kp} {-kd}")
    # gripper = root.find("actuator")[7]
    # gripper.set("gainprm", str(400 * 0.04 / 255))
    # gripper.set("biasprm", "0 -400 -10")
    # gripper.set("forcerange", "-40 40")
    # Gravity compensation holds the arm up. Only robot bodies exist at this point,
    # so bins and trash added later are not affected.
    for body in root.findall(".//body"):
        body.set("gravcomp", "1")
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(
        hand, "site", name="grasp", pos="0 0 0.103", size="0.004", rgba="1 0.3 0.1 0"
    )


def add_room(world):
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="2 2 0.1",
        pos=f"0 0 {FLOOR_Z}",
        rgba="0.18 0.2 0.23 1",
    )
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        size="0.6 0.55 0.04",
        pos="0.4 0 -0.04",
        rgba="0.55 0.40 0.25 1",
        friction="0.8 0.01 0.001",
    )
    ET.SubElement(
        world,
        "camera",
        name="overview",
        pos="-0.1 -1.3 1.4",
        xyaxes="0.852 -0.524 0 0.331 0.539 0.775",
    )


def add_bins(world):
    for i, (x, y) in enumerate(BIN_XY):
        body = ET.SubElement(world, "body", name=f"bin{i}", pos=f"{x} {y} 0")
        for name, pos, size in BIN_PARTS:
            ET.SubElement(
                body,
                "geom",
                name=f"bin{i}_{name}",
                type="box",
                pos=pos,
                size=size,
                rgba=BIN_COLORS[i],
            )


def add_objects(world):
    """One free body per class, starting parked on the floor. reset() moves one onto the table."""
    for i, shape in enumerate(SHAPES):
        x, y = PARK_XY[i]
        body = ET.SubElement(world, "body", name=f"object{i}", pos=f"{x} {y} {FLOOR_Z}")
        ET.SubElement(body, "freejoint", name=f"object{i}_joint")
        for j, geom in enumerate(ET.fromstring(f"<shapes>{shape}</shapes>")):
            geom.set("name", f"object{i}_geom{j}")
            geom.set("friction", f"{OBJECT_FRICTION[i]} 0.01 0.004")
            geom.set("condim", "6")  # Contacts resist rolling and twisting too.
            geom.set(
                "solref", "0.005 1"
            )  # Stiffer contact, less sinking into the fingers.
            body.append(geom)


def make_scene():
    """Extend Menagerie's Panda XML in memory; cached assets stay untouched."""
    folder = Path(panda_path())
    root = ET.parse(folder / "panda.xml").getroot()
    root.find("compiler").set("meshdir", str(folder / "assets"))
    root.remove(root.find("keyframe"))  # Added free joints change qpos length.
    ET.SubElement(root, "visual").append(
        ET.Element("global", offwidth="640", offheight="480")
    )

    add_robot_tuning(root)

    asset = root.find("asset")
    vertex, face = crumpled_paper_mesh(radius=0.967 * OBJECT_RADIUS[2])
    ET.SubElement(asset, "mesh", name="crumpled_paper", vertex=vertex, face=face)
    vertex, face = can_mesh(OBJECT_RADIUS[0], OBJECT_HEIGHT[0])
    ET.SubElement(asset, "mesh", name="can", vertex=vertex, face=face)

    world = root.find("worldbody")
    add_room(world)
    add_bins(world)
    add_objects(world)
    return ET.tostring(root, encoding="unicode")
