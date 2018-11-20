import code.membrane as mb

g = mb.Grid(3)
e = mb.Element((0, 1, 2), 1e6, 0.3, 0.01, 4000)
g.add_elem(e)
print(g.elements[0].to_string())