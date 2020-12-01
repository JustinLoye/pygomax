import numpy as np
import networkx as nx
from . import pagerankcpp

class GoogleMatrix(nx.DiGraph):
    """
    Google matrix class. Mostly inherit from nx.Digraph.
    """

    def __init__(self, data=None, weighted=False, ALPHA = 0.85):
        super(GoogleMatrix, self).__init__(incoming_graph_data=data)

        if weighted == False:
            nx.set_edge_attributes(self, 1.0, "weight")

        self.mapping2py = list(self.nodes())
        Relabeled = nx.convert_node_labels_to_integers(nx.DiGraph(self),label_attribute="pylabel")
        self.mapping2cpp = dict((v,k) for k,v in  nx.get_node_attributes(Relabeled, "pylabel").items())

        nfrom = [0]*self.number_of_edges()
        nto = [0]*self.number_of_edges()
        weight = [0]*self.number_of_edges()
        idx = 0
        for u, v, d in Relabeled.edges(data=True):
            nfrom[idx] = u 
            nto[idx] = v
            weight[idx] = d["weight"]
            idx+=1

        self.N = self.number_of_nodes()
        self.ALPHA = ALPHA
        self.cpp = pagerankcpp.Gmatrix(nfrom = nfrom, 
                                        to = nto,
                                        weight = weight,
                                        alpha = self.ALPHA) 

def pagerank_cpp(Gmatrix, maxiter=150):
    """Compute the PageRank of a Google Matrix.

    Parameters
    ----------
    Gmatrix : class Google Matrix
    maxiter : int, optional(defaut=150)

    Returns
    -------
    dict PageRank probability keyed by node label
    """
    
    PageRank = Gmatrix.cpp.pagerank(MAXITER = maxiter)
    
    return {Gmatrix.mapping2py[i] : PageRank[i] for i in range(Gmatrix.N)}

def regomax(Gmatrix, rnodes, maxiter=150):
    """Regomax algorithm.

    Parameters
    ----------
    Gmatrix : class Google Matrix
    rnodes : list of the node labels used to compute regomax with

    Returns
    -------
    list of numpy arrays(len(rnodes)*len(rnodes)) containing Grr, Gpr, Gqr
    GR[i,j] is the probability to jump from node rnodes[j] to rnodes[i]
    """

    rnodes_cpp = [Gmatrix.mapping2cpp[i] for i in rnodes]

    Grr, Gpr, Gqr = Gmatrix.cpp.regomax(rnodes=rnodes_cpp, MAXITER = maxiter)

    print("Weight of Grr is ", np.sum(Grr)/len(rnodes))
    print("Weight of Gpr is ", np.sum(Gpr)/len(rnodes))
    print("Weight of Gqr is ", np.sum(Gqr)/len(rnodes))
    print("Weight of GR is ", np.sum(Grr+Gpr+Gqr)/len(rnodes))

    return Grr, Gpr, Gqr