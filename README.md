# README #

Requires some of my utility functions and classes, which are not public.

Uses Calcium imaging data, which is not included.

### What is this repository for? ###

* Uses Dynamic Mode Decomposition to analyze C elegans behavior
* Note: For introduction and examples look at ./doc
* ./Zimmer_analysis_functions contains the current work:
	* Zimmer_analysis.m is basically my lab manual (all the scripts I've run)
	* Zimmer_paper_plots.m produces the plots in the paper
		* Note: this will take a long time to run and may need external DMD files
	* CElegansModel.m is a class with most of the analysis and plotting functions
* ./examples contains one script for now:
	* Zimmer_interactive_plots.m produces an interactive plot for data exploration
	

### How do I get set up? ###

#### Without downloading anything else you can:
* View the doc folder, particularly the pdfs
	* These go through the basic DMD analysis and how Robust PCA is used
	* Hopefully will include some more recent results soon!

#### Requirements to run:
* Get data files from an amazing experimentalist
* Praise the experimentalist
* Get my utility functions (DMD_toolbox) 
	* Run their setup script (setup_toolbox_dmd.m)
	* Note: This repository private
* Run the setup for this folder
	* Example:
	* >> filename = "FILENAME OF DATA STRUCT";
	* >> my_model = Zimmer_interactive_plots(filename);

### Contribution guidelines ###

* For personal use!

### Who do I talk to? ###

* Charles Fieseler (konda at uw dot edu)