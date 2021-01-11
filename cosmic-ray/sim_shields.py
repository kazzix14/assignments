from sim_prelude import Box

# north y+
#  east x+
# refs blueprint

# ドームは遠いし何かよくわからんからとりあえず無視

# 本体
CEILING_MAIN_X_ORIGIN = -35.150
CEILING_MAIN_Y_ORIGIN = -2.900
CEILING_MAIN_Z_1F_ORIGIN = 4.200
CEILING_MAIN_Z_2F_ORIGIN = CEILING_MAIN_Z_1F_ORIGIN + 3.600
CEILING_MAIN_Z_3F_ORIGIN = CEILING_MAIN_Z_2F_ORIGIN + 3.600
CEILING_MAIN_Z_4F_ORIGIN = CEILING_MAIN_Z_3F_ORIGIN + 3.600
CEILING_MAIN_Z_5F_ORIGIN = CEILING_MAIN_Z_4F_ORIGIN + 3.600
CEILING_MAIN_Z_6F_ORIGIN = CEILING_MAIN_Z_5F_ORIGIN + 3.600
CEILING_MAIN_WIDTH = 53.9
CEILING_MAIN_DEPTH = 11.6

# 屋上の出口みたいな小屋
CEILING_X_7F_ORIGIN = -3.75
CEILING_Y_7F_ORIGIN = 2.9
CEILING_Z_7F_ORIGIN = CEILING_MAIN_Z_6F_ORIGIN + 3.6
CEILING_WIDTH_7F = 7.5
CEILING_DEPTH_7F = 7.6
CEILING_HEIGHT_7F = 0.1


# 南に飛び出してるとこ
CEILING_SUB_WIDTH = 11.8
CEILING_SUB_DEPTH = 7.6
CEILING_SUB_X_ORIGIN = CEILING_MAIN_X_ORIGIN
CEILING_SUB_Y_ORIGIN = CEILING_MAIN_Y_ORIGIN - \
    CEILING_SUB_DEPTH
CEILING_SUB_Z_1F_ORIGIN = CEILING_MAIN_Z_1F_ORIGIN
CEILING_SUB_Z_2F_ORIGIN = CEILING_MAIN_Z_2F_ORIGIN
CEILING_SUB_Z_3F_ORIGIN = CEILING_MAIN_Z_3F_ORIGIN
CEILING_SUB_Z_4F_ORIGIN = CEILING_MAIN_Z_4F_ORIGIN

# 壁
# よくわからんので適当に薄く
# 壁 2階からになってるので注意
WALL_Y_0_ORIGIN_X = CEILING_SUB_X_ORIGIN
WALL_Y_0_ORIGIN_Y = CEILING_SUB_Y_ORIGIN
WALL_Y_0_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_Y_0_WIDTH = CEILING_SUB_WIDTH
WALL_Y_0_DEPTH = 0.15
WALL_Y_0_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_Y_0_ORIGIN_Z


WALL_Y_2_ORIGIN_X = CEILING_MAIN_X_ORIGIN
WALL_Y_2_ORIGIN_Y = -2.900
WALL_Y_2_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_Y_2_WIDTH = CEILING_MAIN_WIDTH
WALL_Y_2_DEPTH = 0.1
WALL_Y_2_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_Y_2_ORIGIN_Z

WALL_Y_3_ORIGIN_X = CEILING_MAIN_X_ORIGIN
WALL_Y_3_ORIGIN_Y = 2.900
WALL_Y_3_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_Y_3_WIDTH = CEILING_MAIN_WIDTH
WALL_Y_3_DEPTH = 0.1
WALL_Y_3_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_Y_3_ORIGIN_Z

WALL_Y_3d_ORIGIN_X = CEILING_MAIN_X_ORIGIN
WALL_Y_3d_ORIGIN_Y = 2.900 + 2.000
WALL_Y_3d_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_Y_3d_WIDTH = CEILING_MAIN_WIDTH
WALL_Y_3d_DEPTH = 0.1
WALL_Y_3d_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_Y_3d_ORIGIN_Z

WALL_Y_5_ORIGIN_X = CEILING_MAIN_X_ORIGIN
WALL_Y_5_ORIGIN_Y = -2.900 + 11.600
WALL_Y_5_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_Y_5_WIDTH = CEILING_MAIN_WIDTH
WALL_Y_5_DEPTH = 0.15
WALL_Y_5_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_Y_5_ORIGIN_Z


WALL_X_0_ORIGIN_X = CEILING_SUB_X_ORIGIN
WALL_X_0_ORIGIN_Y = CEILING_SUB_Y_ORIGIN
WALL_X_0_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_X_0_WIDTH = 0.1
WALL_X_0_DEPTH = CEILING_MAIN_DEPTH + CEILING_SUB_DEPTH
WALL_X_0_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_0_ORIGIN_Z

WALL_X_2_ORIGIN_X = CEILING_MAIN_X_ORIGIN
WALL_X_2_ORIGIN_Y = CEILING_SUB_Y_ORIGIN
WALL_X_2_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_X_2_WIDTH = 0.1
WALL_X_2_DEPTH = CEILING_MAIN_DEPTH + CEILING_SUB_DEPTH
WALL_X_2_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_2_ORIGIN_Z

WALL_X_3d_ORIGIN_X = -3.750 - 7.500
WALL_X_3d_ORIGIN_Y = -2.900
WALL_X_3d_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_X_3d_WIDTH = 0.16
WALL_X_3d_DEPTH = 5.800
WALL_X_3d_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_3d_ORIGIN_Z

WALL_X_4_ORIGIN_X = -3.750
WALL_X_4_ORIGIN_Y = -2.900
WALL_X_4_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_X_4_WIDTH = 0.1
WALL_X_4_DEPTH = 5.800
WALL_X_4_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_4_ORIGIN_Z

WALL_X_5_ORIGIN_X = -3.750 + 7.500
WALL_X_5_ORIGIN_Y = -2.900
WALL_X_5_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_X_5_WIDTH = 0.1
WALL_X_5_DEPTH = 5.800
WALL_X_5_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_5_ORIGIN_Z

WALL_X_6_ORIGIN_X = -3.750 + 7.500*2
WALL_X_6_ORIGIN_Y = -2.900
WALL_X_6_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_X_6_WIDTH = 0.1
WALL_X_6_DEPTH = 5.800
WALL_X_6_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_6_ORIGIN_Z

WALL_X_7_ORIGIN_X = -3.750 + 7.500*3
WALL_X_7_ORIGIN_Y = -2.900
WALL_X_7_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_X_7_WIDTH = 0.1
WALL_X_7_DEPTH = 5.800
WALL_X_7_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_7_ORIGIN_Z

WALL_X_8_ORIGIN_X = -3.750 + 7.500*4
WALL_X_8_ORIGIN_Y = -2.900
WALL_X_8_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_X_8_WIDTH = 0.20
WALL_X_8_DEPTH = 5.800
WALL_X_8_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_8_ORIGIN_Z


WALL_XU_0_ORIGIN_X = CEILING_SUB_X_ORIGIN
WALL_XU_0_ORIGIN_Y = CEILING_SUB_Y_ORIGIN
WALL_XU_0_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_XU_0_WIDTH = 0.1
WALL_XU_0_DEPTH = CEILING_MAIN_DEPTH + CEILING_SUB_DEPTH
WALL_XU_0_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_0_ORIGIN_Z

WALL_XU_2_ORIGIN_X = CEILING_MAIN_X_ORIGIN
WALL_XU_2_ORIGIN_Y = CEILING_SUB_Y_ORIGIN
WALL_XU_2_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_XU_2_WIDTH = 0.1
WALL_XU_2_DEPTH = CEILING_MAIN_DEPTH + CEILING_SUB_DEPTH
WALL_XU_2_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_2_ORIGIN_Z

WALL_XU_3d_ORIGIN_X = -3.750 - 7.500
WALL_XU_3d_ORIGIN_Y = +2.900 + 2.000
WALL_XU_3d_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_XU_3d_WIDTH = 0.16
WALL_XU_3d_DEPTH = 5.800
WALL_XU_3d_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_XU_3d_ORIGIN_Z

WALL_XU_4_ORIGIN_X = -3.750
WALL_XU_4_ORIGIN_Y = +2.900 + 2.000
WALL_XU_4_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_XU_4_WIDTH = 0.1
WALL_XU_4_DEPTH = 5.800
WALL_XU_4_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_4_ORIGIN_Z

WALL_XU_5_ORIGIN_X = -3.750 + 7.500
WALL_XU_5_ORIGIN_Y = +2.900 + 2.000
WALL_XU_5_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_XU_5_WIDTH = 0.1
WALL_XU_5_DEPTH = 5.800
WALL_XU_5_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_5_ORIGIN_Z

WALL_XU_6_ORIGIN_X = -3.750 + 7.500*2
WALL_XU_6_ORIGIN_Y = +2.900 + 2.000
WALL_XU_6_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_XU_6_WIDTH = 0.1
WALL_XU_6_DEPTH = 5.800
WALL_XU_6_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_6_ORIGIN_Z

WALL_XU_7_ORIGIN_X = -3.750 + 7.500*3
WALL_XU_7_ORIGIN_Y = +2.900 + 2.000
WALL_XU_7_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_XU_7_WIDTH = 0.1
WALL_XU_7_DEPTH = 5.800
WALL_XU_7_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_7_ORIGIN_Z

WALL_XU_8_ORIGIN_X = -3.750 + 7.500*4
WALL_XU_8_ORIGIN_Y = +2.900 + 2.000
WALL_XU_8_ORIGIN_Z = CEILING_MAIN_Z_1F_ORIGIN
WALL_XU_8_WIDTH = 0.20
WALL_XU_8_DEPTH = 5.800
WALL_XU_8_HEIGHT = CEILING_MAIN_Z_6F_ORIGIN - WALL_X_8_ORIGIN_Z


def build_shields(ceiling_thickness):
    shields = []

    # floors
    # ceiling 1f
    shields.append(Box(CEILING_MAIN_X_ORIGIN,
                       CEILING_MAIN_Y_ORIGIN,
                       CEILING_MAIN_Z_1F_ORIGIN,
                       CEILING_MAIN_WIDTH,
                       CEILING_MAIN_DEPTH,
                       ceiling_thickness))
    shields.append(Box(CEILING_SUB_X_ORIGIN,
                       CEILING_SUB_Y_ORIGIN,
                       CEILING_SUB_Z_1F_ORIGIN,
                       CEILING_SUB_WIDTH,
                       CEILING_SUB_DEPTH,
                       ceiling_thickness))
    # ceiling 2f
    shields.append(Box(CEILING_MAIN_X_ORIGIN,
                       CEILING_MAIN_Y_ORIGIN,
                       CEILING_MAIN_Z_2F_ORIGIN,
                       CEILING_MAIN_WIDTH,
                       CEILING_MAIN_DEPTH,
                       ceiling_thickness))
    shields.append(Box(CEILING_SUB_X_ORIGIN,
                       CEILING_SUB_Y_ORIGIN,
                       CEILING_SUB_Z_2F_ORIGIN,
                       CEILING_SUB_WIDTH,
                       CEILING_SUB_DEPTH,
                       ceiling_thickness))
    # ceiling 3f
    shields.append(Box(CEILING_MAIN_X_ORIGIN,
                       CEILING_MAIN_Y_ORIGIN,
                       CEILING_MAIN_Z_3F_ORIGIN,
                       CEILING_MAIN_WIDTH,
                       CEILING_MAIN_DEPTH,
                       ceiling_thickness))
    shields.append(Box(CEILING_SUB_X_ORIGIN,
                       CEILING_SUB_Y_ORIGIN,
                       CEILING_SUB_Z_3F_ORIGIN,
                       CEILING_SUB_WIDTH,
                       CEILING_SUB_DEPTH,
                       ceiling_thickness))
    # ceiling 4f
    shields.append(Box(CEILING_MAIN_X_ORIGIN,
                       CEILING_MAIN_Y_ORIGIN,
                       CEILING_MAIN_Z_4F_ORIGIN,
                       CEILING_MAIN_WIDTH,
                       CEILING_MAIN_DEPTH,
                       ceiling_thickness))
    shields.append(Box(CEILING_SUB_X_ORIGIN,
                       CEILING_SUB_Y_ORIGIN,
                       CEILING_SUB_Z_4F_ORIGIN,
                       CEILING_SUB_WIDTH,
                       CEILING_SUB_DEPTH,
                       ceiling_thickness))
    # ceiling 5f
    shields.append(Box(CEILING_MAIN_X_ORIGIN,
                       CEILING_MAIN_Y_ORIGIN,
                       CEILING_MAIN_Z_5F_ORIGIN,
                       CEILING_MAIN_WIDTH,
                       CEILING_MAIN_DEPTH,
                       ceiling_thickness))
    # ceiling 6f
    shields.append(Box(CEILING_MAIN_X_ORIGIN,
                       CEILING_MAIN_Y_ORIGIN,
                       CEILING_MAIN_Z_6F_ORIGIN,
                       CEILING_MAIN_WIDTH,
                       CEILING_MAIN_DEPTH,
                       ceiling_thickness))  # 防水やシンダー分増える

    # ceiling 7f
    shields.append(Box(CEILING_X_7F_ORIGIN,
                       CEILING_Y_7F_ORIGIN,
                       CEILING_Z_7F_ORIGIN,
                       CEILING_WIDTH_7F,
                       CEILING_DEPTH_7F,
                       CEILING_HEIGHT_7F))

    # walls
    # コミュニティースペース北
    shields.append(Box(WALL_X_0_ORIGIN_X,
                       WALL_X_0_ORIGIN_Y,
                       WALL_X_0_ORIGIN_Z,
                       WALL_X_0_WIDTH,
                       WALL_X_0_DEPTH,
                       WALL_X_0_HEIGHT))

    shields.append(Box(WALL_X_2_ORIGIN_X,
                       WALL_X_2_ORIGIN_Y,
                       WALL_X_2_ORIGIN_Z,
                       WALL_X_2_WIDTH,
                       WALL_X_2_DEPTH,
                       WALL_X_2_HEIGHT))

    shields.append(Box(WALL_Y_2_ORIGIN_X,
                       WALL_Y_2_ORIGIN_Y,
                       WALL_Y_2_ORIGIN_Z,
                       WALL_Y_2_WIDTH,
                       WALL_Y_2_DEPTH,
                       WALL_Y_2_HEIGHT))

    shields.append(Box(WALL_Y_3_ORIGIN_X,
                       WALL_Y_3_ORIGIN_Y,
                       WALL_Y_3_ORIGIN_Z,
                       WALL_Y_3_WIDTH,
                       WALL_Y_3_DEPTH,
                       WALL_Y_3_HEIGHT))

    shields.append(Box(WALL_Y_3d_ORIGIN_X,
                       WALL_Y_3d_ORIGIN_Y,
                       WALL_Y_3d_ORIGIN_Z,
                       WALL_Y_3d_WIDTH,
                       WALL_Y_3d_DEPTH,
                       WALL_Y_3d_HEIGHT))

    shields.append(Box(WALL_Y_5_ORIGIN_X,
                       WALL_Y_5_ORIGIN_Y,
                       WALL_Y_5_ORIGIN_Z,
                       WALL_Y_5_WIDTH,
                       WALL_Y_5_DEPTH,
                       WALL_Y_5_HEIGHT))

    shields.append(Box(WALL_X_3d_ORIGIN_X,
                       WALL_X_3d_ORIGIN_Y,
                       WALL_X_3d_ORIGIN_Z,
                       WALL_X_3d_WIDTH,
                       WALL_X_3d_DEPTH,
                       WALL_X_3d_HEIGHT))

    shields.append(Box(WALL_X_4_ORIGIN_X,
                       WALL_X_4_ORIGIN_Y,
                       WALL_X_4_ORIGIN_Z,
                       WALL_X_4_WIDTH,
                       WALL_X_4_DEPTH,
                       WALL_X_4_HEIGHT))

    shields.append(Box(WALL_X_5_ORIGIN_X,
                       WALL_X_5_ORIGIN_Y,
                       WALL_X_5_ORIGIN_Z,
                       WALL_X_5_WIDTH,
                       WALL_X_5_DEPTH,
                       WALL_X_5_HEIGHT))

    shields.append(Box(WALL_X_6_ORIGIN_X,
                       WALL_X_6_ORIGIN_Y,
                       WALL_X_6_ORIGIN_Z,
                       WALL_X_6_WIDTH,
                       WALL_X_6_DEPTH,
                       WALL_X_6_HEIGHT))

    shields.append(Box(WALL_X_7_ORIGIN_X,
                       WALL_X_7_ORIGIN_Y,
                       WALL_X_7_ORIGIN_Z,
                       WALL_X_7_WIDTH,
                       WALL_X_7_DEPTH,
                       WALL_X_7_HEIGHT))

    shields.append(Box(WALL_X_8_ORIGIN_X,
                       WALL_X_8_ORIGIN_Y,
                       WALL_X_8_ORIGIN_Z,
                       WALL_X_8_WIDTH,
                       WALL_X_8_DEPTH,
                       WALL_X_8_HEIGHT))

    shields.append(Box(WALL_XU_0_ORIGIN_X,
                       WALL_XU_0_ORIGIN_Y,
                       WALL_XU_0_ORIGIN_Z,
                       WALL_XU_0_WIDTH,
                       WALL_XU_0_DEPTH,
                       WALL_XU_0_HEIGHT))

    shields.append(Box(WALL_XU_2_ORIGIN_X,
                       WALL_XU_2_ORIGIN_Y,
                       WALL_XU_2_ORIGIN_Z,
                       WALL_XU_2_WIDTH,
                       WALL_XU_2_DEPTH,
                       WALL_XU_2_HEIGHT))

    shields.append(Box(WALL_XU_3d_ORIGIN_X,
                       WALL_XU_3d_ORIGIN_Y,
                       WALL_XU_3d_ORIGIN_Z,
                       WALL_XU_3d_WIDTH,
                       WALL_XU_3d_DEPTH,
                       WALL_XU_3d_HEIGHT))

    shields.append(Box(WALL_XU_4_ORIGIN_X,
                       WALL_XU_4_ORIGIN_Y,
                       WALL_XU_4_ORIGIN_Z,
                       WALL_XU_4_WIDTH,
                       WALL_XU_4_DEPTH,
                       WALL_XU_4_HEIGHT))

    shields.append(Box(WALL_XU_5_ORIGIN_X,
                       WALL_XU_5_ORIGIN_Y,
                       WALL_XU_5_ORIGIN_Z,
                       WALL_XU_5_WIDTH,
                       WALL_XU_5_DEPTH,
                       WALL_XU_5_HEIGHT))

    shields.append(Box(WALL_XU_6_ORIGIN_X,
                       WALL_XU_6_ORIGIN_Y,
                       WALL_XU_6_ORIGIN_Z,
                       WALL_XU_6_WIDTH,
                       WALL_XU_6_DEPTH,
                       WALL_XU_6_HEIGHT))

    shields.append(Box(WALL_XU_7_ORIGIN_X,
                       WALL_XU_7_ORIGIN_Y,
                       WALL_XU_7_ORIGIN_Z,
                       WALL_XU_7_WIDTH,
                       WALL_XU_7_DEPTH,
                       WALL_XU_7_HEIGHT))

    shields.append(Box(WALL_XU_8_ORIGIN_X,
                       WALL_XU_8_ORIGIN_Y,
                       WALL_XU_8_ORIGIN_Z,
                       WALL_XU_8_WIDTH,
                       WALL_XU_8_DEPTH,
                       WALL_XU_8_HEIGHT))

    return shields
