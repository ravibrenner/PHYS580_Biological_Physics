# PHYS580-Project
Code from my final project. The code was written to accompany the paper
I would recommend running each simulation separately (they each have a number at the end of the name indicating the order in which they were intended to be run).
If you prefer to view the code altogether, the files titled "PHYS580_Project.py" is the one for you

Feel free to vary some of the initial values for dim, i, and r. Be careful, as some values won't work if they
are out of range.

# The Moran Process on a Lattice with Spatial Dependence
The Moran Process is a model of nite population biology that can be used to
model natural selection, genetic drift, and spontaneous mutation. In each time
step, one random individual is chosen for reproduction, and another is chosen
for death or replacement, causing the total population size to remain constant.
In this paper, I will describe the basics of the typical Moran Process, which
uses a well mixed population. Then, I will explore the application of this same
process to a 1D and then a 2D lattice, where there is spatial dependence on
reproduction and replacement. I start with simpler models and build up to
more complicated setups, using python to create simulations of some dierent
possible outcomes. Most of the work of this project was in actually encoding
the Moran process on a lattice. Use this link for a github of the code.
