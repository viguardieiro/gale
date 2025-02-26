from gudhi.cover_complex import MapperComplex
from gale import create_pd

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import networkx as nx


def plot_mapper(mapper: MapperComplex, 
                add_text=True, 
                ax=None,
                node_scale=1.0,
                node_position='kamada_kawai'):
    """Plots the Mapper output as a networkx graph

    Args:
        mapper (MapperComplex): Mapper output from `create_mapper`
        add_text (bool, optional): Add text with number of components and nodes. Defaults to True.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes. Defaults to creating a new axis.
        node_scale (float, optional): Scale the node size. Defaults to 1.0.
        node_position (str, optional): Networkx layout. Defaults to 'kamada_kawai'.
    """
    G = mapper.get_networkx(set_attributes_from_colors=True)

    nodes = list(G.nodes)
    node_info = mapper.node_info_
    node_color = [node_info[k]["colors"][0] for k in nodes]
    node_size = [node_scale*node_info[k]["size"] for k in nodes]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    if node_position == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif node_position == 'spring':
        pos = nx.spring_layout(G)

    nx.draw(G, with_labels=False, ax=ax, 
            pos=pos,
            node_color=node_color, node_size=node_size, 
            edge_color='grey', width=0.5, cmap='coolwarm')

    if add_text:
        cc = nx.number_connected_components(G)
        n_nodes = len(nodes)
        at = AnchoredText(f"components = {cc}\nnodes = {n_nodes}", loc='lower left', prop=dict(size=10), frameon=False)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

    ax.set_title('Mapper representation')
    ax.axis('on')

    return fig, ax

def plot_ext_persistance_diagram(mapper: MapperComplex, 
                                 ax=None):
    """Plots the extended persistence diagram of the Mapper output

    Args:
        mapper (MapperComplex): Mapper output from `create_mapper`
        ax (matplotlib.axes.Axes, optional): Matplotlib axes. Defaults to creating a new axis.
    """
    
    types = ["Down branch","Upper branch","Trunk","Holes"]

    dgms, pdgms = create_pd(mapper, return_d=True)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    for type_i in range(4):
            type_s = types[type_i]
            ax.scatter([dgms[type_i][j][1][0] for j in range(len(dgms[type_i]))], 
                                [dgms[type_i][j][1][1] for j in range(len(dgms[type_i]))],
                                label=type_s)

    ax.set_title('Extended persistence diagram')
    ax.plot([-1, 1], [-1, 1], color='black', linewidth=0.5)
    ax.legend()

    return fig, ax
