import taichi as ti
import math

ti.init(arch=ti.cpu)

# =========================
# 立方体：8个点，12条边
# =========================
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

# =========================
# Model（多轴旋转）
# =========================
@ti.func
def get_model_matrix(ax: ti.f32, ay: ti.f32, az: ti.f32):
    rx = ax * math.pi / 180.0
    ry = ay * math.pi / 180.0
    rz = az * math.pi / 180.0

    cx, sx = ti.cos(rx), ti.sin(rx)
    cy, sy = ti.cos(ry), ti.sin(ry)
    cz, sz = ti.cos(rz), ti.sin(rz)

    Rx = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx,  cx, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    Ry = ti.Matrix([
        [ cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    Rz = ti.Matrix([
        [cz, -sz, 0.0, 0.0],
        [sz,  cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return Rz @ Ry @ Rx


# =========================
# View
# =========================
@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])


# =========================
# Projection
# =========================
@ti.func
def get_projection_matrix(fov: ti.f32, aspect: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar

    fov_rad = fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect * t
    l = -r

    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    M_scale = ti.Matrix([
        [2/(r-l), 0.0, 0.0, 0.0],
        [0.0, 2/(t-b), 0.0, 0.0],
        [0.0, 0.0, 2/(n-f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r+l)/2],
        [0.0, 1.0, 0.0, -(t+b)/2],
        [0.0, 0.0, 1.0, -(n+f)/2],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return M_scale @ M_trans @ M_p2o


# =========================
# 核心计算
# =========================
@ti.kernel
def compute(ax: ti.f32, ay: ti.f32, az: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])

    model = get_model_matrix(ax, ay, az)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)

    mvp = proj @ view @ model

    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])

        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]

        screen_coords[i][0] = (v_ndc[0] + 1.0) * 0.5
        screen_coords[i][1] = (v_ndc[1] + 1.0) * 0.5


# =========================
# 主程序
# =========================
def main():

    # 初始化立方体
    cube = [
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]
    ]

    for i in range(8):
        vertices[i] = cube[i]

    gui = ti.GUI("3D Cube", (700, 700))

    ax = ay = az = 0.0

    while gui.running:

        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                gui.running = False

            elif e.key == 'w':
                ax += 5
            elif e.key == 's':
                ax -= 5

            elif e.key == 'a':
                ay += 5
            elif e.key == 'd':
                ay -= 5

            elif e.key == 'q':
                az += 5
            elif e.key == 'e':
                az -= 5

        compute(ax, ay, az)

        # 画边
        for edge in edges:
            a = screen_coords[edge[0]]
            b = screen_coords[edge[1]]
            gui.line(a, b, radius=2, color=0xFFFFFF)

        gui.show()


if __name__ == '__main__':
    main()