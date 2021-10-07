# Oracle-Efficient Regret Minimization in Factored MDPs with Unknown Structure

This code implements the experiments from the paper "Oracle-Efficient Regret Minimization in Factored MDPs with Unknown Structure" by Aviv Rosenberg and Yishay Mansour (NeurIPS 2021).

It runs different versions of UCRL, Factored UCRL and SLF-UCRL on the SysAdmin domain (both the circular and the star configurations).

The experiments are defined in RL_FMDP_XXX.py, where XXX is the name of the domain.

The rest of the code is organized as follows:

- MDP.py represents the true model of an MDP, while FactoredMDP.py represents the true model of an FMDP. Both are subclasses of AbstractMDP.

- The domains used in experiments are all subclasses of FactoredMDP.

- AbstractUCRL.py contains the main loop of UCRL, but defers details (updatepolicy and updateparams) to its subclasses.

- UCRL.py implements UCRL for a standard MDP, while FactoredUCRL.py implements UCRL for FMDPs and SLFUCRL.py implements SLF-UCRL for FMDPs with unknown (or partially known) structure.

- Confidence intervals for both rewards and transition probabilities are computed using classes in ConfidenceBounds.py.

- Different versions of EVI are implemented using classes in ExtendedValueIteration.py.
