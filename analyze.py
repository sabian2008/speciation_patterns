import numpy as np
import matplotlib.pyplot as plt


def gen_dist(genes):
    """Computes the genetic distance between individuals.

    Inputs:
      - genes: NxB matrix with the genome of each individual.

    Output:
      - out:   NxN symmetric matrix with (out_ij) the genetic
               distance from individual N_i to individual N_j.
    """

    # First generate an NxNxB matrix that has False where
    # i and j individuals have the same kth gene and True
    # otherwise (XOR operation). Then sum along
    # the genome axis to get distance
    return np.sum(genes[:,None,:] ^ genes, axis=-1)


def classify(genes, G):
    """Classifies a series of individuals in isolated groups,
    using transitive genetic distance as the metric. If individual
    1 is related to invidual 2, and 2 with 3, then 1 is related to
    3 (independently of their genetic distance). Based on:
    https://stackoverflow.com/questions/41332988  and
    https://stackoverflow.com/questions/53886120

    Inputs:
      - genes: NxB matrix with the genome of each individual.
      - G:     Integer denoting the maximum genetic distance.

    Output:
      - out:   list of lists. The parent lists contains as many
               elements as species. The child lists contains the
               indices of the individuals corresponding to that
               species.
    """

    import networkx as nx

    # Calculate distance
    gendist = gen_dist(genes)
    N = gendist.shape[0]

    # Make the distance with comparison "upper triangular" by
    # setting lower diagonal of matrix extremely large
    gendist[np.arange(N)[:,None] >= np.arange(N)] = 999999

    # Get a list of pairs of indices (i,j) satisfying distance(i,j) < G
    indices = np.where(gendist <= G)
    indices = list(zip(indices[0], indices[1]))

    # Now is the tricky part. I want to combine all the (i,j) indices
    # that share at least either i or j. This non-trivial problem can be
    # modeled as a graph problem (stackoverflow link). The solution is
    G = nx.Graph()
    G.add_edges_from(indices)
    return list(nx.connected_components(G))


# Load simulation
target = "S6-G20.npz"

data = np.load(target)

genes, pos, params = data['genes'], data['pos'], data['params'][()]

G = params['G']  # Max. genetic distance


#t = 2050//25  # Target time (-1 is the last one)

species = []

fig, ax = plt.subplots(1,1)

plt.show()

for t in range(500, genes.shape[0]):
    # Classify genomes into species
    species.append(classify(genes[t], G))

    # Plot geographical distribution of species
    ax.clear()
    for s in species[t-500]:
        ax.scatter(pos[t,list(s),0], pos[t,list(s),1], s=5)

    ax.set_xlim([1, 128])
    ax.set_ylim([1, 128])

    plt.pause(0.1)

    fig.tight_layout()

