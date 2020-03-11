import numpy as np


def parse_inp_file(path: str):
    """Parses .inp file (ABAQUS input) into format suitable for further usage by FEM model
        Args:
            path (str): path to .inp file
    """
    with open(path, 'r') as ifile:
        content = ifile.readlines()
    # reading nodes
    nodes_section_start = next((idx + 1 for idx, s in enumerate(content) if s.startswith("*NODE")))
    nodes_section_end = next((idx for idx, s in enumerate(content) if s.startswith("******* E L E M E N T S *************")))
    points = np.genfromtxt(content[nodes_section_start:nodes_section_end], \
            delimiter=',', usecols=[1,2], dtype=np.float64)

    triangles = []
    lines = []

    first_triangle_id = 1
    for linenum, s in enumerate(content[nodes_section_end + 1:]):
        if s.startswith("*"):
            if not s.startswith("*ELEMENT"):
                break
        else:
            nodes = [int(i) - 1 for i in s.split(',')[1:]]
            if len(nodes) == 2:
                lines.append(nodes)
                first_triangle_id += 1
            elif len(nodes) == 3:
                triangles.append(nodes)
            else:
                raise ValueError(f"Line {nodes_section_end + linenum + 2:d}: only 2 or 3 node elements supported. Got {s}")

    groups_section_start = linenum + nodes_section_end + 1
    prefix_len = len("*ELSET,ELSET=")
    node_groups = dict()
    element_groups = dict()

    current_group_name = None
    current_group_dim = None
    current_group = None

    content.append("*ELSET,ELSET=")

    for linenum, s in enumerate(content[groups_section_start:]):
        if s.startswith("*ELSET,ELSET="):
            if current_group_name is not None:
                # saving previous group
                if current_group_dim == 2:
                    node_groups[current_group_name] = list(current_group) # nodes
                elif current_group_dim == 3:
                    element_groups[current_group_name] = current_group 
            # reading new group
            current_group_name = s[prefix_len : -1]
            current_group = set()
            current_group_dim = None
        else:
            ids = [int(i) for i in s.split(', ')[:-1] if len(i) > 0]
            tri_ids = [i - first_triangle_id for i in ids if i >= first_triangle_id]
            line_ids = [i - 1 for i in ids if i < first_triangle_id]

            if current_group_dim is None:
                if len(tri_ids) > 0:
                    current_group_dim = 3
                    current_group = []
                else:
                    current_group_dim = 2
                    current_group = set()

            if current_group_dim == 3:
                current_group += tri_ids
            elif current_group_dim == 2:
                for i in line_ids:
                    for j in lines[i]:
                        current_group.add(j)

    return points, triangles, node_groups, element_groups


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        _, _, node_groups, element_groups = parse_inp_file(sys.argv[1])
        print(element_groups)
    else:
        print(f'Usage: {sys.argv[0]} mesh_file.inp')
