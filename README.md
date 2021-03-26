# LangevinPy

Python Function adapting the implementation of the Langevin Thermostat from Allen and Tildesley's \emph{Computer Simulations of Liquids} p. 263. This is adapted into a dual thermostat system with one providing a thermal bath and the other providing stronger, less frequent kicks to coarse grain the impact of random activity on a tracer particle. 

### Usage:
python VariableLangevin.py -N "Number of particles" -x "filepath"/"outputname_prefix" -S "Seed for RNG" -n "How many timesteps between strong kicks" -T "k_B T"
### Example Usage:
python VariableLangevin.py -N 1 <br />

### Dependencies
numpy
pandas
argparse
