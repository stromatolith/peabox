#!python
"""
just checking whether local neighbourhood of degree N is determined correctly
"""

from PSO_defs import PSO_Topo_standard2D as PSOTop


pt=PSOTop([4,5],1)
i,j=pt.get_indices_of(17)
print pt.map
print i,j
#l=pt.get_neighbourhood_upto(17,2)
l=pt.get_neighbourhood(17,2)
for el in l:
    i,j=pt.imap[el]
    pt.map[i,j]=-1

print pt.map