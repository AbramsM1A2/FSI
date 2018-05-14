# Search methods

import search

ab = search.GPSProblem('A', 'B', search.romania)
fr = search.GPSProblem('F', 'R', search.romania)
co = search.GPSProblem('C', 'O', search.romania)

print("||||||||||||||||||||-De B a A-||||||||||||||||||||")

print("-------Busqueda no informada por anchura---------")
print search.breadth_first_graph_search(ab).path()
print("-------Busqueda no informada por profundidad---------")
print search.depth_first_graph_search(ab).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()
print("-------Busqueda no informada por coste de camino---------")
print search.search_fring(ab).path()
print("-------Busqueda informada---------")
print search.search_fring_h(ab).path()

print("||||||||||||||||||||-De R a F-||||||||||||||||||||")

print("-------Busqueda no informada por anchura---------")
print search.breadth_first_graph_search(fr).path()
print("-------Busqueda no informada por profundidad---------")
print search.depth_first_graph_search(fr).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()
print("-------Busqueda no informada por coste de camino---------")
print search.search_fring(fr).path()
print("-------Busqueda informada---------")
print search.search_fring_h(fr).path()

print("||||||||||||||||||||-De O a C-|||||||||||||||||||")

print("-------Busqueda no informada por anchura---------")
print search.breadth_first_graph_search(co).path()
print("-------Busqueda no informada por profundidad---------")
print search.depth_first_graph_search(co).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()
print("-------Busqueda no informada por coste de camino---------")
print search.search_fring(co).path()
print("-------Busqueda informada---------")
print search.search_fring_h(co).path()
#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
