import numpy as np
import numpy.random as random


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
    # otherwise (XOR operation). Then sum along
    # the genome axis to get distance
    return np.sum(genes[:,None,:] ^ genes, axis=-1)


def reproduce(genes, dist, gendist, S, G, P):
    """ Choose a suitable pairing mate and generate offspring.

    Inputs:
      - genes:   NxB matrix with the genes for each individual.
      - dist:    N vector with the geometric distances between
                 individuals.
      - gendist: N vector with the genetic distances between
                 individuals.
      - S:       Max. mating distance for reproductive viability.
      - G:       Max. genetic distance for reproductive viability.
      - P:       Min. number of viable partners required.

    Output:
      - out:     Bx1 vector with the genome for the offspring.
    """

    myG = G+1
    myS = S+1

    # Find viable partners
    viable = (dist <= myS) & (gendist <= myG)
    num_viable = np.sum(viable)

    # Relax criteria until P parters are viable
    while num_viable <= P:
        myG = myG+1
        myS = myS+1

        viable = (dist <= myS) & (gendist <= myG)
        num_viable = np.sum(viable)

    # Randomly choose index of partner from viable individuals
    partner = random.choice(np.flatnonzero(viable))

    # How much from each parent the offspring inherits
    cross_distance = random.randint(0, high=B)

    # Id of current parent
    my_id = np.where(dist == 0)[0][0]

    return np.hstack( [genes[my_id,:cross_distance],
        genes[partner,cross_distance:]] )


# ------------
# Parameters
# ------------
Nx    = 128       # Lattice size in X
Ny    = Nx        # Lattice size in Y

N     = Nx*Ny//8  # Number of individuals

S     = 4         # Max. geometric mating distance
G     = 20        # Min. genetic mating distance
P     = 8         # Min. number of possible mates

B     = 125       # Number of genes per individual

D     = 0.01      # Probability of offspring dispersion
Q     = 0.3       # Probability of forever alone
mu    = 0.001     # Probability of mutation
pmu   = 0.65      # p value of the geometric distribution
                  # that regulates number of genes that mutate

tend  = 2500      # Number of generations to evolve
tsave = 25        # Save every tsave generations

prev  = True      # True for loading a previous run. All the parameters
                  # must be the same (this is not checked)

# ------------
# Simulation
# -----------

# Initial condition
if prev:
    # Previous run
    data = np.load("S{}-G{}.npz".format(S,G))
    genes = data["genes"][-1]
    pos = data["pos"][-1]
else:
    # Uniform genes and random placement
    genes = np.array([ True for i in range(0,N*B) ]).reshape(N, B)
    pos = random.randint(1, high=Nx+1, size=N)
    pos = np.stack( [pos, random.randint(1, high=Ny+1, size=N)], axis=-1)

# Temporary arrays to store offspring data until iteration completes
tgenes = np.empty((N, B), dtype=np.bool)
tpos = np.empty((N,2), dtype=np.int)

# Arrays to store cumulative results
sgenes = np.empty((tend//tsave, N, B), dtype=np.bool)
spos = np.empty((tend//tsave, N, 2), dtype=np.int)

# Iterate
for t in range(0, tend+1):
    print("Iteration {} of {}".format(t, tend), end="\r")

    # Precompute distances
    dist = man_dist(pos, Nx, Ny)
    gendist = gen_dist(genes)

    # Precompute chances of skipping, dispersing and mutating
    skip = random.random(size=N) < Q
    mutate = random.random(size=N) < mu
    disp = random.random(size=N) < D

    # By default position stays the same
    tpos[:] = pos[:]

    # Populate tgenes and tpos
    for i in range(0, N):
        # No offspring, reproduce a neighbour in it's place
        if skip[i]:
            # Search spatial neighbourhood, and choose random sample
            j = random.choice( np.flatnonzero(dist[i] <= S ) )

            # Reproduce neighbour
            tgenes[i] = reproduce(genes, dist[j], gendist[j], S, G, P)

        # Offspring
        else:
            tgenes[i] = reproduce(genes, dist[i], gendist[i], S, G, P)

        # Mutations
        if mutate[i]:
            # Number of genes to mutate from geometric distribution
            # which ends at B-1
            num_mutations = (random.geometric(pmu) - 1) % B

            # Determine positions to flip (replace=False guarantees
            # num_mutations different positions). Then flip genome
            flipped = random.choice(B, size=num_mutations, replace=False)
            tgenes[flipped] = ~ tgenes[flipped]

        # Dispersion
        if disp[i]:
            # Move to one of the 20 neighbouring locations
            dx, dy = random.randint(-2, high=3, size=2)
            while np.abs(dx) + np.abs(dy) in [0, 4]:
                dx, dy = random.randint(-2, high=3, size=2)

            # Apply translation (module is for periodicity)
            tpos[i, 0] = (tpos[i, 0] + dx -1) % Nx + 1
            tpos[i, 1] = (tpos[i, 1] + dy - 1) % Ny + 1

    # Offsprings are now parents
    genes = tgenes
    pos = tpos

    if t % tsave == 0:
        sgenes[t//tsave-1] = genes
        spos[t//tsave-1] = pos

# Write simulation to disk
params = {"Nx":Nx, "Ny":Ny, "N":N, "S":S, "G":G, "P":P, "B":B, "D":D,
          "Q":Q, "mu":mu, "pmu":pmu, "tsave":tsave}

if prev:
    genes = np.concatenate((data["genes"], sgenes), axis=0)
    pos = np.concatenate((data["pos"], spos), axis=0)
    np.savez("S{}-G{}".format(S,G), genes=genes, pos=pos, params=params)
else:
    np.savez("S{}-G{}".format(S,G), genes=sgenes, pos=spos, params=params)
