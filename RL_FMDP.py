
from __future__             import print_function

import sys
import time

from MDP                    import MDP
from FactoredMDP            import FactoredMDP

from UCRL                   import UCRL
from FactoredUCRL           import FactoredUCRL
from SLFUCRL           		import SLFUCRL

from ConfidenceBounds       import *
from ExtendedValueIteration import ElementwiseEVI, OsbandEVI, StandardVI

import matplotlib.pyplot as plt
import pickle

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class RL_FMDP:

	def __init__( self, FMDP, time_horizon, ival, delta, no_expr ):
		self.FMDP         = FMDP
		self.time_horizon = time_horizon
		self.ival         = ival
		self.delta        = delta
		self.no_expr      = no_expr

	def run( self ):
		# Solve the full MDP
		MDP = self.FMDP.fullMDP()
		pi, g_opt = MDP.solve()
		pkl_dict = {}
		pkl_dict["SLF-UCRL"] = [{} for i in range(self.no_expr)]
		pkl_dict["SLF-UCRL2"] = [{} for i in range(self.no_expr)]
		pkl_dict["SLF-UCRL3"] = [{} for i in range(self.no_expr)]
		pkl_dict["SLF-UCRL4"] = [{} for i in range(self.no_expr)]
		pkl_dict["UCRL"] = [{} for i in range(self.no_expr)]
		pkl_dict["UCRL-Fact"] = [{} for i in range(self.no_expr)]

		print( g_opt, flush = True )

		# Main loop
		for k in range( self.no_expr ):
			print( k + 1, flush = True )

			UCRL_FactoredOsband  = FactoredUCRL( self.FMDP,
			                                     [OsbandRI()],
			                                     [OsbandCB()],
			                                     OsbandEVI() )

			SLF_UCRL            	 = SLFUCRL( self.FMDP,
												[OsbandRI()],
			                                     [OsbandCB()],
			                                     OsbandEVI(),
			                                     [0] )

			SLF_UCRL2            	 = SLFUCRL( self.FMDP,
												[OsbandRI()],
			                                     [OsbandCB()],
			                                     OsbandEVI(),
			                                     [0,1] )

			SLF_UCRL3            	 = SLFUCRL( self.FMDP,
												[OsbandRI()],
			                                     [OsbandCB()],
			                                     OsbandEVI(),
			                                     [0,1,2] )

			SLF_UCRL4            	 = SLFUCRL( self.FMDP,
												[OsbandRI()],
			                                     [OsbandCB()],
			                                     OsbandEVI(),
			                                     [0,1,2,3] )

			UCRLR             	=         UCRL( MDP,
			                                     [HoeffdingLaplace()],
			                                     [OsbandCB()],
			                                     ElementwiseEVI() )
			
			start = time.time()

			regret_SLFUCRL             = SLF_UCRL           .runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of SLF-UCRL              : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of SLF_UCRL            : " + str( regret_SLFUCRL            [-1] )  , flush = True )
			pkl_dict["SLF-UCRL"][k]["regret"] = regret_SLFUCRL
			pkl_dict["SLF-UCRL"][k]["elim_times"] = SLF_UCRL.elimination_times

			start = time.time()

			regret_SLFUCRL2             = SLF_UCRL2           .runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of SLF-UCRL2              : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of SLF_UCRL2            : " + str( regret_SLFUCRL2            [-1] )  , flush = True )
			pkl_dict["SLF-UCRL2"][k]["regret"] = regret_SLFUCRL2
			pkl_dict["SLF-UCRL2"][k]["elim_times"] = SLF_UCRL2.elimination_times

			start = time.time()

			regret_SLFUCRL3             = SLF_UCRL3           .runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of SLF-UCRL3              : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of SLF_UCRL3            : " + str( regret_SLFUCRL3            [-1] )  , flush = True )
			pkl_dict["SLF-UCRL3"][k]["regret"] = regret_SLFUCRL3
			pkl_dict["SLF-UCRL3"][k]["elim_times"] = SLF_UCRL3.elimination_times

			start = time.time()

			regret_SLFUCRL4             = SLF_UCRL4           .runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of SLF-UCRL4              : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of SLF_UCRL4            : " + str( regret_SLFUCRL4            [-1] )  , flush = True )
			pkl_dict["SLF-UCRL4"][k]["regret"] = regret_SLFUCRL4
			pkl_dict["SLF-UCRL4"][k]["elim_times"] = SLF_UCRL4.elimination_times

			start = time.time()

			regret_UCRLR                = UCRLR              .runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of UCRLR                 : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of UCRLR               : " + str( regret_UCRLR               [-1] )  , flush = True )
			pkl_dict["UCRL"][k]["regret"] = regret_UCRLR

			start = time.time()

			regret_UCRL_Factored_Osband = UCRL_FactoredOsband.runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of Factored Osband       : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of UCRL_Factored_Osband: " + str( regret_UCRL_Factored_Osband[-1] )  , flush = True )
			pkl_dict["UCRL-Fact"][k]["regret"] = regret_UCRL_Factored_Osband
