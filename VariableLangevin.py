import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Langevin Simulation of Spheres with Variable number of steps between random forces')
parser.add_argument('-N', '--N', nargs='?', help='Number of spheres', required=True, type=int)
parser.add_argument('-x', '--fname', nargs='?',default='Lang', help='Output path and filename prefix', type=str)
parser.add_argument('-n', '--Nevery', nargs='?', default=1, help='Number of steps betweeen random force implementations', type=int)
parser.add_argument('-t', '--Nsteps', nargs='?', default=1e7, help='Number of Steps for Simulation', type=int)
parser.add_argument('-P', '--Nprint', nargs='?', default=100, help='Number of Steps Between Prints', type=int)
parser.add_argument('-d', '--dt', nargs='?', default=0.01, help='timestep', type=float)
parser.add_argument('-f', '--phi', nargs='?', default=0.0001, help='Initial Packing Fraction', type=float)
parser.add_argument('-T', '--T', nargs='?', default=1e-2, help='Temperature', type=float)
parser.add_argument('-S', '--seedval', nargs='?', default=1, help='Seed for pseudorandom number generator', type=int)

args = parser.parse_args()


def run_sim(N, phi, Nsteps, Nevery, Nprint, dt, KbT, fname, seedval):
    # Define system parameters
    #N =10 # Number of atoms
    #phi = 0.1 # Packing Fraction
    sigmas = np.ones(N) # Diameters of spheres
    ms = np.ones(N) # Masses
    #Nprint = 1000
    #Nsteps = 1000000
    #Nevery = 1
    #fname = 'Desktop/LangevinTest'
    # Seed random number generator
    np.random.seed(seedval)

    # Initialize rs array and set 
    rs = np.zeros((N,3)) # Positions
    Vi = np.sum(np.pi*sigmas/6.) # Total volumes of spheres
    L = (Vi/phi)**(1./3.) # Define box 
    Ls = np.ones(3)*L
    bounds = ['p','p','p']

    # Initialize Box
    simbox = box(Ls,bounds)

    # Open output files
    xyzname = fname + '.xyz'
    datname = fname + '.dat'
    xyz = open(xyzname,'w')
    dat = open(datname,'w')

    # Construct Simulation Object
    sim = langevin_sim(N, rs, ms, sigmas, simbox, xyz=xyz,dat=dat, Nevery=Nevery, KbT=KbT, dt = dt,damp = 1e-1)

    # Place atoms randomly in box
    sim.MC_init()

    # Run simulation
    KEarr = []
    for i in np.arange(0,Nsteps):
        if (i+1)%Nprint==0:
            #print(sim.Gr)
            #print(sim.Gv)
            print('%1.1f%% (%d/%d) Completed, KE: %e' %(100*(i+1)/Nsteps,i+1,Nsteps, sim.KE()))
            KEarr.append(sim.KE())
            sim.print_coords()
        sim.timestep()
        
    # Close Files
    xyz.close()
    dat.close()

    # Process Data and Save Files
    np.save(fname+'_KE.npy', KEarr)

    df = pd.read_csv(fname+'.dat',delimiter=' ')
    rs = df[['x','y','z']]
    times = np.unique(df['Time'].array)
    N = len(np.where(df['Time']==times[0])[0])
    Nt = len(times)
    rs_arr = np.zeros((N,3,Nt))*np.nan
    i_t = -1
    for t in times:
        i_t += 1
        rs_arr[:,:,i_t] = rs.iloc[np.where(df['Time']==t)]
    np.save(fname+'_rs.npy', rs_arr)

    # Get MSD
    dts, MSD, MFD = make_MSDMFD(times,rs_arr,N,500)    
    np.save(fname+'_msd.npy', MSD)
    np.save(fname+'_dts.npy', dts)

    # Get Displacement Correlation
    Ndt = 50
    dr_bins = np.arange(0,500,0.1)
    dr_binmids, corr_all, inds = dr_corr_all(rs_arr,dr_bins, Ndt)
    np.save(fname+'_dr_binmids.npy', dr_binmids)
    np.save(fname+'_dr_corr.npy', corr_all)
    np.save(fname+'_dt_corr.npy',inds*dt)

def dr_corr_all(rs, dr_bins, Ndt):
    Nt = np.shape(rs)[-1]
    inds=np.unique(np.round(np.logspace(np.log10(1),np.log10(Nt-1),Ndt))).astype(int)
    corr_all = np.zeros((len(dr_bins)-1,len(inds)))*np.nan
    for i in np.arange(0,len(inds)):
        ind=inds[i]
        drs=rs[:,:,np.arange(ind,Nt,ind)]-rs[:,:,np.arange(0,Nt-ind,ind)]
        if np.shape(drs)[-1]<10:
            break
        dr_binmids, corr_bin = binned_corrs(drs[0,0,:],dr_bins)
        corr_all[:,i] = corr_bin
    return dr_binmids, corr_all, inds

def bin_inds(bins,values):
    inds = np.zeros(len(values))*np.nan
    bin_lows = bins[:-1]
    bin_highs = bins[1:]
    i=-1
    for value in values:
        i+=1
        ind = np.where((value>bin_lows)*(value<bin_highs))[0]
        try:
            inds[i] = ind
        except:
            print(ind)
    #print(np.min(inds))
    return inds

def corr_means(bins, drs, corrs):
    inds = bin_inds(bins,drs)
    corr_bin = np.zeros(len(bins)-1)*np.nan
    ind_unique = np.unique(inds.astype(int))
    for ind in ind_unique:
        try:
            corr_bin[ind] = np.mean(corrs[np.where(inds==ind)])
        except:
            print('In corr_means, found meaningless ind: %d' %ind)
    return corr_bin

def binned_corrs(drs, dr_bins):
    #dr_bins = np.arange(0,np.max(np.abs(drs))+delta,delta)
    dr_bin_mids = 0.5*(dr_bins[1:]+dr_bins[:-1])
    corr = drs[1:]*drs[:-1]
    corr = corr/np.abs(corr)*np.abs(drs[1:])
    corr_bin = corr_means(dr_bins, np.abs(drs[:-1]), corr)
    return dr_bin_mids, corr_bin

def make_MSDMFD(t, rs, Natoms, Ndt):
    inds=np.unique(np.round(np.logspace(np.log10(1),np.log10(len(t)-2),Ndt)))
    dts=inds*float(t[1]-t[0])
    MSD=np.zeros((Natoms,len(inds)))
    MFD=np.zeros((Natoms,len(inds)))
    for i in np.arange(0,len(inds)):
        ind=int(inds[i])
        drs=rs[:,:,0:-ind]-rs[:,:,ind:]
        MSD[:,i]=np.mean(np.sum(drs**2,axis=1),axis=1)
        MFD[:,i]=np.mean(np.sum(drs**4,axis=1),axis=1)
    return dts, MSD, MFD

class box:
    ndim = 3
    Ls = np.zeros(ndim)
    bounds = ['p','p','p']
    
    def __init__(self, Ls, bounds, ndim=3):
        self.Ls = Ls
        self.bounds = bounds
    
    def box_vec(self, r):
        r_box = r.copy()
        for d in np.arange(0,self.ndim):
            if self.bounds[d] == 'p':
                r_box[d] -= self.Ls[d]*np.round(r[d]/self.Ls[d])
        return r_box
    
class langevin_sim:
    dt = 0
    damp = 1.
    c0 = 0
    c1 = 0
    c2 = 0
    sig_r = 1.
    sig_v = 1.
    N = 1
    ndim = 3
    rs = np.zeros((N,ndim))
    vs = np.zeros((N,ndim))

    def __init__(self, N, rs, ms, sigmas, box, xyz=open('test.xyz','w'), dat=open('test.dat','w'), Nevery=1, dt=0.01, damp=0.01, ndim=3, KbT = 0.01, eps=1.):
        
        # Constructor for simulation
        
        self.box = box
        
        # Assign system quantities and initialize velocities and accelerations
        self.dt = dt
        self.N = int(N)
        self.rs = rs
        self.sigmas = sigmas
        self.sig_ij = 0.5*np.add.outer(sigmas,sigmas)
        self.ndim = int(ndim)
        self.KbT = KbT
        self.ms = ms
        self.vs = np.zeros((N,ndim))
        self.init_vs()
        self.a = np.zeros((N,ndim))
        
        # Define Langevin Thermostat constants and initialize drG and drV
        
        if damp != np.ndarray:
            damp = damp*np.ones(N)
        self.damp = damp
        c = np.tile(self.damp*self.dt, (ndim,1)).T
        
        self.c0 = np.exp(-c)
        self.c1 = -np.expm1(-c)/c
        self.c2 = (1.0-self.c1)/c
        
        self.sig_r = np.sqrt(2.*c/3. - 0.5*c*c+7.*c**3/30.)
        self.sig_v = np.sqrt(-np.expm1(-2.*c))
        exdpdt = - np.expm1(-c)
        self.c_rv = exdpdt*exdpdt/c/self.sig_r/self.sig_v
        g1 = np.random.randn(self.N,self.ndim)
        g2 = np.random.randn(self.N,self.ndim)
        self.Gr = self.sig_r*g1
        self.Gv = self.c_rv*self.sig_v*g1 + self.sig_v*np.sqrt(1.-self.c_rv**2)*g2
        
        
        # Define constants for repulsive LJ interaction
        self.u0 = -12.*eps*self.sig_ij**6
        self.u1 = 12.*eps*self.sig_ij**12

        # Counting of steps for frequency of applied random force
        self.step=0
        self.Nevery = Nevery
        self.Gpre = np.tile(np.sqrt(KbT/ms)*np.sqrt(Nevery), (self.ndim,1)).T
        #self.Gpre = np.tile(np.sqrt(KbT/ms), (self.ndim,1)).T
        
        #print('Gpre:')
        #print(self.Gpre)
        
        # Initialize files for printing
        self.xyz = xyz
        self.dat = dat
        self.dat.write('Time ind diameter x y z vx vy vz\n')
        
    def MC_init(self):
        # put each atom at a different place in box
        self.rs[0,:] = (np.random.rand(3)-0.5)*self.box.Ls
        for i in np.arange(1,self.N):
            print('Placed %d Atoms' %i)
            bad_coords = True
            while bad_coords:
                self.rs[i,:] = (np.random.rand(3)-0.5)*self.box.Ls
                # Check if overlaps
                for j in np.arange(0,i):
                    dist = np.linalg.norm(self.rs[i,:] - self.rs[j,:])
                    bad_coords = dist<self.sig_ij[i,j]
                    if bad_coords:
                        break
        print('Placed %d Atoms' %self.N)

    def init_vs(self):
        '''
        Initialize velocities to the specified KbT
        '''
        # Give velocities random direction
        self.vs = np.random.randn(self.N,self.ndim)
        
        # Rescale velocities to specified KbT
        self.vs *= np.sqrt(float(self.ndim)*self.KbT/np.sum(np.mean(np.tile(self.ms,(self.ndim,1)).T*self.vs**2,axis=0)))
        
    def KE(self):
        '''
        Calculate kinetic energy of system
        '''
        return 0.5*np.sum(np.mean(np.tile(self.ms,(self.ndim,1)).T*self.vs**2,axis=0))
    
    def update_GrGv(self):
        '''
        Function to update the random variables for the Langevin thermostate
        '''
        g1 = np.random.randn(self.N,self.ndim)
        g2 = np.random.randn(self.N,self.ndim)
        self.Gr = self.sig_r*g1
        self.Gv = self.c_rv*self.sig_v*g1 + self.sig_v*np.sqrt(1.-self.c_rv**2)*g2


    def timestep(self):
        '''
        Make velocity-verlet langevin step
        '''
        self.rs += self.c1*self.dt*self.vs + self.c2*self.dt**2*self.a
         
        # First update to vs:
        self.vs = self.c0*self.vs + (self.c1 - self.c2) * self.dt * self.a
        
        # Add random force
        if (self.step % self.Nevery) == 0:
            #print(self.step)
            self.update_GrGv()
            self.rs += self.dt*self.Gpre*self.Gr
            self.vs += self.Gpre*self.Gv
        
        # Get new forces due to new locations
        self.update_interactions()
        
        # 
        self.vs += self.c2*self.dt*self.a
        
        self.step += 1
    
    def drs(self):
        '''
        Calculate displacement vectors
        '''
        drs = np.zeros((self.N,self.N,self.ndim))*np.nan
        distmat = np.zeros((self.N,self.N))
        for i in np.arange(0,self.N):
            for j in np.arange(i+1,self.N):
                dr = self.rs[i,:] - self.rs[j,:]
                dr = self.box.box_vec(dr)
                dr_norm = np.linalg.norm(dr)
                drs[i,j,:] = dr
                drs[j,i,:] = dr
                distmat[i,j] = dr_norm
                distmat[j,i] = dr_norm
        return drs, distmat
    
    def update_interactions(self):
        self.a = np.zeros((self.N,self.ndim))
        '''
        Implement repulsive Lennard-Jones interaction
        '''
        drs, distmat = self.drs()
        for i in np.arange(0,self.N):
            for j in np.arange(i+1,self.N):
                dist = distmat[i,j]
                if dist < self.sig_ij[i,j]:
                    f = (self.u0[i,j]*dist**(-7) + self.u1[i,j] * dist**(-13))*drs[i,j,:]/dist
                    #print('(%d,%d): dist: %f, fi: (%f, %f, %f)' %(i,j,dist,f[0],f[1],f[2]))
                    self.a[i,:] += f/self.ms[i]
                    self.a[j,:] += -f/self.ms[j]
                    
    def print_coords(self):
        time = self.step*self.dt
        Lx = self.box.Ls[0]
        Ly = self.box.Ls[1]
        Lz = self.box.Ls[2]
        self.xyz.write(str(self.N)+'\n')
        box_str = 'Lattice="'+str(Lx) + ' 0. 0. 0. '+str(Ly)+' 0. 0. 0. '+str(Lz)+'" Origin ="'+str(-0.5*Lx)+' '+\
            str(-0.5*Ly)+' '+str(-0.5*Lz)+'" Time='+str(time)+'\n'
        self.xyz.write(box_str)
        #xyz.write('Lattice="%f 0. 0. 0. %f 0. 0. 0. %f" Origin="-35.3785 -20.4683 -50" Time=%f\n' %(Lx,Ly,Lz,-0.5*Lx,-0.5*Ly,-0.5*Lz,time))
        for i in np.arange(0,self.N):
            sig = self.sigmas[i]
            vx = self.vs[i,0]
            vy = self.vs[i,1]
            vz = self.vs[i,2]
            r = self.rs[i,:].copy()
            x = r[0]
            y = r[1]
            z = r[2]
            self.dat.write('%f %d %f %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e\n' %(time,i,sig,x,y,z,vx,vy,vz))
            r = self.box.box_vec(self.rs[i,:].copy())
            x = r[0]
            y = r[1]
            z = r[2]
            self.xyz.write('X %f %1.8e %1.8e %1.8e\n' %(sig*0.5,x+0.5*Lx,y+0.5*Ly,z+0.5*Lz))
            

run_sim(args.N, args.phi, args.Nsteps, args.Nevery, args.Nprint, args.dt, args.T, args.fname, args.seedval)          