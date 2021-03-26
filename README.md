# LangevinPy

Python Function adapting the implementation of the Langevin Thermostat from Allen and Tildesley's _Computer Simulations of Liquids_ p. 263. This is adapted into a dual thermostat system with one providing a thermal bath and the other providing stronger, less frequent kicks to coarse grain the impact of random activity on a tracer particle. 

### Usage:
python VariableLangevin.py -N "Number of particles" -x "filepath"/"outputname_prefix" -S "Seed for RNG" -n "How many timesteps between strong kicks" -T "k_B T"
### Example Usage:
python VariableLangevin.py -N 1 -x outputdir/Sim_N1T0.01 -T 0.01 -S 105693 <br />

### Outputs:
.xyz file containing the boxed coordinates of the simulation at every timestep <br />
.dat file containing unboxed coordinates and velocities of particles

### Dependencies
numpy
pandas
argparse
