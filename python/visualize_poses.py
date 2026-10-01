#!/usr/bin/env python3

import sys
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_pose(filename):
    """
    Load one pose from CSV.

    Format:
        qw,qx,qy,qz,x,y,z
    """

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)

        row = next(reader)

        # Support CSV with header
        try:
            values = [float(value) for value in row]
        except ValueError:
            row = next(reader)
            values = [float(value) for value in row]

    if len(values) != 7:
        raise ValueError(
            f"{filename}: expected 7 values "
            "(qw,qx,qy,qz,x,y,z)"
        )

    qw, qx, qy, qz, x, y, z = values

    return (
        np.array([qw, qx, qy, qz], dtype=float),
        np.array([x, y, z], dtype=float),
    )


def quaternion_to_rotation_matrix(q):
    """
    Convert [qw, qx, qy, qz] to a 3x3 rotation matrix.
    """

    q = np.asarray(q, dtype=float)

    norm = np.linalg.norm(q)

    if norm == 0:
        raise ValueError("Quaternion has zero length")

    qw, qx, qy, qz = q / norm

    return np.array([
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qx * qw),
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx**2 + qy**2),
        ],
    ])


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to [qw, qx, qy, qz].
    """

    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)

        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])

        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])

        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])

        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz])

    return q / np.linalg.norm(q)


def quaternion_slerp(q0, q1, t):
    """
    Spherical linear interpolation between two quaternions.
    """

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)

    # Use the shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    # Small angle -> linear interpolation
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0

    return s0 * q0 + s1 * q1


def interpolate_pose(q0, p0, q1, p1, t):
    """
    Interpolate position linearly and orientation with SLERP.
    """

    position = (1.0 - t) * p0 + t * p1
    quaternion = quaternion_slerp(q0, q1, t)

    return quaternion, position


def draw_coordinate_frame(
    ax,
    position,
    quaternion,
    axis_length=1.0,
    linewidth=2.0,
):
    """
    Draw X/Y/Z coordinate frame.

    X -> red
    Y -> green
    Z -> blue
    """

    R = quaternion_to_rotation_matrix(quaternion)

    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    ax.quiver(
        position[0],
        position[1],
        position[2],
        x_axis[0],
        x_axis[1],
        x_axis[2],
        length=axis_length,
        color="red",
        linewidth=linewidth,
        arrow_length_ratio=0.15,
    )

    ax.quiver(
        position[0],
        position[1],
        position[2],
        y_axis[0],
        y_axis[1],
        y_axis[2],
        length=axis_length,
        color="green",
        linewidth=linewidth,
        arrow_length_ratio=0.15,
    )

    ax.quiver(
        position[0],
        position[1],
        position[2],
        z_axis[0],
        z_axis[1],
        z_axis[2],
        length=axis_length,
        color="blue",
        linewidth=linewidth,
        arrow_length_ratio=0.15,
    )


def draw_static_frame(
    ax,
    position,
    quaternion,
    label,
    axis_length,
):
    """
    Draw a static coordinate frame with a label.
    """

    draw_coordinate_frame(
        ax,
        position,
        quaternion,
        axis_length,
        linewidth=2,
    )

    ax.scatter(
        position[0],
        position[1],
        position[2],
        color="black",
        s=50,
    )

    ax.text(
        position[0],
        position[1],
        position[2],
        f"  {label}",
        fontsize=10,
    )


def set_equal_axes(ax, positions, axis_length):
    """
    Make X/Y/Z scales equal.
    """

    positions = np.asarray(positions)

    min_values = positions.min(axis=0) - axis_length
    max_values = positions.max(axis=0) + axis_length

    center = (min_values + max_values) / 2
    radius = np.max(max_values - min_values) / 2

    ax.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )

    ax.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )

    ax.set_zlim(
        center[2] - radius,
        center[2] + radius,
    )


def main():

    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} "
            "pose1.csv pose2.csv relative_pose.csv"
        )
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    relative_file = sys.argv[3]

    # ---------------------------------------------------------
    # Load poses
    # ---------------------------------------------------------

    q1, p1 = load_pose(file1)
    q2, p2 = load_pose(file2)

    q_relative, p_relative = load_pose(relative_file)

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    label1 = os.path.splitext(
        os.path.basename(file1)
    )[0]

    label2 = os.path.splitext(
        os.path.basename(file2)
    )[0]

    relative_label = os.path.splitext(
        os.path.basename(relative_file)
    )[0]

    # ---------------------------------------------------------
    # Calculate transformation predicted by relative pose
    #
    # Assumption:
    #
    #     T_12 = relative transformation from pose 1 to pose 2
    #
    #     T_2 = T_1 * T_12
    # ---------------------------------------------------------

    R1 = quaternion_to_rotation_matrix(q1)
    R_relative = quaternion_to_rotation_matrix(q_relative)

    predicted_R2 = R1 @ R_relative
    predicted_p2 = p1 + R1 @ p_relative

    predicted_q2 = rotation_matrix_to_quaternion(
        predicted_R2
    )

    print()
    print("=" * 60)

    print(f"{label1}")
    print(f"  q = {q1}")
    print(f"  p = {p1}")

    print()

    print(f"{label2}")
    print(f"  q = {q2}")
    print(f"  p = {p2}")

    print()

    print(f"{relative_label}")
    print(f"  q = {q_relative}")
    print(f"  p = {p_relative}")

    print()

    print("Pose 2 predicted from Pose 1 + Relative Pose:")
    print(f"  q = {predicted_q2}")
    print(f"  p = {predicted_p2}")

    print()

    print("Position error:")
    print(f"  {np.linalg.norm(predicted_p2 - p2):.6f}")

    print("=" * 60)
    print()

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    axis_length = 1.0

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Static pose 1
    draw_static_frame(
        ax,
        p1,
        q1,
        label1,
        axis_length,
    )

    # Static pose 2
    draw_static_frame(
        ax,
        p2,
        q2,
        label2,
        axis_length,
    )

    # ---------------------------------------------------------
    # Relative transformation
    #
    # Relative pose is visualized at pose 1's origin.
    # ---------------------------------------------------------

    relative_position_world = p1 + R1 @ p_relative
    relative_rotation_world = R1 @ R_relative

    relative_q_world = rotation_matrix_to_quaternion(
        relative_rotation_world
    )

    # Relative pose frame
    draw_static_frame(
        ax,
        relative_position_world,
        relative_q_world,
        relative_label,
        axis_length * 0.7,
    )

    # ---------------------------------------------------------
    # Draw line between absolute poses
    # ---------------------------------------------------------

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Pose displacement",
    )

    # ---------------------------------------------------------
    # Animation objects
    # ---------------------------------------------------------

    animated_quivers = []
    animated_origin = None
    animated_label = None

    # We use the relative transformation as the target.
    #
    # Starting pose:
    #     T = T1
    #
    # Final pose:
    #     T = T1 * T_relative
    #
    # This allows the transformation itself to be visualized.

    animation_frames = 100

    def update(frame):

        nonlocal animated_origin
        nonlocal animated_label

        t = frame / (animation_frames - 1)

        # Interpolate using the relative transformation.
        q_current, p_current = interpolate_pose(
            q1,
            p1,
            predicted_q2,
            predicted_p2,
            t,
        )

        # Remove previous animated frame
        for artist in animated_quivers:
            artist.remove()

        animated_quivers.clear()

        if animated_origin is not None:
            animated_origin.remove()

        if animated_label is not None:
            animated_label.remove()

        # Current rotation
        R_current = quaternion_to_rotation_matrix(
            q_current
        )

        x_axis = R_current[:, 0]
        y_axis = R_current[:, 1]
        z_axis = R_current[:, 2]

        # X axis
        animated_quivers.append(
            ax.quiver(
                p_current[0],
                p_current[1],
                p_current[2],
                x_axis[0],
                x_axis[1],
                x_axis[2],
                length=axis_length,
                color="red",
                linewidth=3,
                arrow_length_ratio=0.15,
            )
        )

        # Y axis
        animated_quivers.append(
            ax.quiver(
                p_current[0],
                p_current[1],
                p_current[2],
                y_axis[0],
                y_axis[1],
                y_axis[2],
                length=axis_length,
                color="green",
                linewidth=3,
                arrow_length_ratio=0.15,
            )
        )

        # Z axis
        animated_quivers.append(
            ax.quiver(
                p_current[0],
                p_current[1],
                p_current[2],
                z_axis[0],
                z_axis[1],
                z_axis[2],
                length=axis_length,
                color="blue",
                linewidth=3,
                arrow_length_ratio=0.15,
            )
        )

        animated_origin = ax.scatter(
            p_current[0],
            p_current[1],
            p_current[2],
            color="black",
            s=70,
        )

        animated_label = ax.text(
            p_current[0],
            p_current[1],
            p_current[2],
            f"  Transform: {t:.2f}",
            fontsize=10,
        )

        return animated_quivers + [
            animated_origin,
            animated_label,
        ]

    # ---------------------------------------------------------
    # Axis configuration
    # ---------------------------------------------------------

    all_positions = [
        p1,
        p2,
        relative_position_world,
        predicted_p2,
    ]

    set_equal_axes(
        ax,
        all_positions,
        axis_length,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        "3D Pose Transformation"
    )

    ax.grid(True)

    # Axis legend
    ax.plot(
        [],
        [],
        color="red",
        label="X axis",
    )

    ax.plot(
        [],
        [],
        color="green",
        label="Y axis",
    )

    ax.plot(
        [],
        [],
        color="blue",
        label="Z axis",
    )

    ax.legend()

    plt.tight_layout()

    # ---------------------------------------------------------
    # Start animation
    # ---------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=animation_frames,
        interval=40,
        blit=False,
        repeat=True,
        repeat_delay=1000,
    )

    # Keep animation alive
    fig._animation = animation

    # save animation
    animation.save(
    "/home/aliaydin/Desktop/awesome_cpp/slam_optimization_tutorial/data/pose_transformation.gif",
    writer="pillow",
    fps=25,
    )

    plt.show()


if __name__ == "__main__":
    main()
