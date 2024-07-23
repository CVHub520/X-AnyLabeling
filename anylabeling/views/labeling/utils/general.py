import re


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return bool(re.search('[\u4e00-\u9fff]', str(s)))

def is_possible_rectangle(points):
    if len(points) != 4:
        return False

    # Check if four points form a rectangle
    # The points are expected to be in the format:
    # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    dists = [square_dist(points[i], points[(i + 1) % 4]) for i in range(4)]
    dists.sort()

    # For a rectangle, the two smallest distances 
    # should be equal and the two largest should be equal
    return dists[0] == dists[1] and dists[2] == dists[3]

def square_dist(p, q):
    # Calculate the square distance between two points
    return (p[0] - q[0])**2 + (p[1] - q[1])**2
