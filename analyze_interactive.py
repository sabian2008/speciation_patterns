import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
S = 5
G = 20
target = "S{}-G{}.npz".format(S,G)

data = np.load(target, allow_pickle=True)
genes, pos, params = data['genes'], data['pos'], data['params'][()]

G = params['G']          # Max. genetic distance
tsave = params["tsave"]  # Saves cadency

# Load classification. If missing, perform it
try:
    with open("S{}-G{}.class".format(S,G), 'rb') as fp:
        species = pickle.load(fp)
except:
    print("Classification not found. Classyfing...")
    species = []
    for t in range(0, genes.shape[0]):
        print("{} of {}".format(t, genes.shape[0]), end="\r")
        species.append(classify(genes[t], G))
    with open("S{}-G{}.class".format(S,G), 'wb') as fp:
        pickle.dump(species, fp)

# Create figure, axes and slider
fig, ax = plt.subplots(1,1)
axtime = plt.axes([0.25, 0.01, 0.5, 0.03], facecolor="lightblue")
stime = Slider(axtime, 'Generation', 0, len(species), valfmt='%0.0f', valinit=0)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Initial condition
for s in species[0]:
    ax.scatter(pos[0,list(s),0], pos[0,list(s),1], s=5)

# Function to update plots
def update(val):
    i = int(round(val))

    # Update plots
    ax.clear()
    for s in species[i]:
        ax.scatter(pos[i,list(s),0], pos[i,list(s),1], s=5)

    stime.valtext.set_text(tsave*i)  # Update label

    # Draw
    fig.canvas.draw_idle()

# Link them
stime.on_changed(update)


# Function to change slider using arrows
def arrow_key_control(event):
    curr = stime.val

    if event.key == 'left':
        stime.set_val(curr-1)

    if event.key == 'right':
        stime.set_val(curr+1)

fig.canvas.mpl_connect('key_release_event', arrow_key_control)

plt.show()
