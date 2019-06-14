import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


def man_dist(pos, Nx, Ny):
    """Computes the periodic Manhattan distance between individuals.

    Inputs:
      - pos: Nx2 matrix with the (x,y) coordinates of the
             N individuals.
      - Nx:  Length of the lattice in X direction.
      - Ny:  Length of the lattice in Y direction.

    Output:
      -     NxN symmetric matrix with (Out_ij) the periodic
            Manhattan distance from individual N_i to individual N_j.
    """

    # Compute x distance and y distance between individuals
    dx = np.abs(pos[:,0][:,None] - pos[:,0])
    dy = np.abs(pos[:,1][:,None] - pos[:,1])

    # Periodic distance can't be greater than Nx/2 (or Ny/2)
    # in that case distance is N - dist
    dx[dx > Nx/2] = Nx - dx[dx > Nx/2]
    dy[dy > Ny/2] = Ny - dy[dy > Ny/2]

    return dx + dy


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
    # otherwise (exclusive OR operation). Then sum along
    # the genome axis to get distance
    return np.sum(genes[:,None,:] ^ genes)


def reproduce(genes, dist, gendist, S, G, P):
    """ Choose a suitable pairing mate and generate offspring.

    Inputs:
      - genes:   NxB matrix with the genes for each individual.
      - dist:    NxN matrix with the geometric distances between
                 individuals.
      - gendist: NxN matrix with the genetic distances between
                 individuals.
      - S:       Max. mating distance for reproductive viability.
      - G:       Max. genetic distance for reproductive viability.
      - P:       Minimum number of viable partners required.

    Output:
      - out:     Bx1 vector with the genome for the offspring.
    """


# Parameters
Nx = 128       # Lattice size in X
Ny = Nx        # Lattice size in Y

N  = Nx*Ny//8  # Number of individuals

S  = 6         # Max. mating distance
G  = 6         # Min. mating distance
P  = 8         # Min. number of possible mates

B  = 125       # Number of genes per individual

D  = 0.01      # Probability of offspring dispersion
Q  = 0.3       # Probability of forever alone
mu = 0.001     # Probability of mutation


# Initialize genes and position of individuals
genes = np.array([ True for i in range(0,N*B) ]).reshape(N, B)
pos = random.randint(1, high=Nx+1, size=N)
pos = np.stack( [pos, random.randint(1, high=Ny+1, size=N)], axis=-1)

# Temporary arrays to store offspring data until iteration completes
tgenes = np.empty((N, B), dtype=np.bool)
tpos = np.empty((N,2), dtype=np.int)


# PSEUDOCODE
# Iterate
# for t in range(1, tend):
#     # Precompute distances
#     dist = man_distance(pos)
#     gendist = gen_distance(genes)

#     # Precompute chances of skipping, dispersing and mutating
#     skip = random.random(size=N) < Q
#     mutate = random.random(size=N) < mu
#     disp = random.random(size=N) < D

      # By default position stays the same
#     tpos[:] = pos[:]

#     # Populate tgenes and tpos
#     for i in range(0, N):
#         # No offspring
#         if skip[i]:
#             # Reproduce a neighbour in it's place
#         # Offspring
#         else:
#             tgenes[i] = reproduce(genes, dist[i], gendist[i], S, G, P)

#         # Mutations
#         if mutate[i]:
#             # Change genome

#         # Dispersion
#         if disp[i]:
#             # Change position slightly

#     pos = tpos
#     genes = tgenes


# fig, ax = plt.subplots(1,1)
# ax.scatter(pos[:,0], pos[:,1], s=2)

# fig.tight_layout()
# plt.show()
