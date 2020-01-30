# README #

MATLAB code for disambiguating control signals from intrinsic dynamics.
The mathematical form is, in matrix notation:
	x' = Ax + Bu
where x is the data (e.g. voltage level of neurons), x' is the next step in time, A is a matrix that implements the intrinsic dynamics, u is the control signal, and B maps the control input to the full phase space.
	
If you use this, please cite: ...

### What is this repository for? ###

* Signal processing when there are two types of processes present:
	* Linear, intrinsic dynamics. See DMD for the assumptions on the dynamics.
	* Sparse control signals. These may be external, e.g. sensory input, or much faster than the intrinsic dynamics.
* These elements are produced in two steps which correspond to separate MATLAB functions:
	* First, learning the control signal
	* Second, learning the dynamics given the control signal
* Three objects will be produced, which can each be analyzed, with some cautionary points discussed in the paper.
	* The intrinsic dynamics matrix, A.
	* The control signal time series, U.
	* The mapping from the control to the original nodes, B.
* Together, these produce a model of the dynamics of the system. This model can be used to reconstruct the original data, or extrapolate into alternative initial conditions or control input.
	

### How do I get set up? ###

Download the custom toolboxes with helper functions as described in the next section.

The script 'setup_Learn_control_signals.m' adds these folders to your MATLAB path.


#### Requirements to run

MATLAB; Tested on MATLAB 2018a.

Data in the proper format as documented in the main function.

The following toolboxes, which need to be downloaded and the folders added to the MATLAB path (as described in their respective README's):
* Toolbox_DMD_git
	* https://github.com/Charles-Fieseler/MATLAB_DMD_toolbox
* Charles_Matlab_toolbox
	* https://github.com/Charles-Fieseler/Charles_Matlab_toolbox


#### Getting started

* Learning control signals
	* Main function: learn_control_signals.m
	* This function returns a sweep of possible control signals with increasing sparsity
	* Several helper functions are included to automatically choose the "best" one; type 'help learn_control_signals' for more information
* Learning intrinsic dynamics
	* The main user interface is provided by the large class, 'SignalLearningObject.m'
	* This will intake the relevant data and solve the optimization problem. 
	* There are many plotting options and algorithm settings available, as documented in the class.


### Contribution guidelines ###

For academic use; I'm not planning on continuing to support this MATLAB version of the code.


### Who do I talk to? ###

* Charles Fieseler (konda at uw dot edu)