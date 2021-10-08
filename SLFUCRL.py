
import numpy as np
import itertools
from AbstractUCRL import AbstractUCRL

import copy

class SLFUCRL( AbstractUCRL ):

	def __init__( self,   MDP, RCB, TCB, EVI, unknown_factors ):
		super().__init__( MDP, RCB, TCB, EVI )

		# list all possible scopes
		self.unknown_factors = unknown_factors
		self.num_factors_unknown = len(self.unknown_factors)
		self.scope_size = len(self.MDP.transitionstruct[1].scope)
		self.num_scopes_of_size_m = 0
		self.possible_scopes = []
		for i in range(self.scope_size-1, self.scope_size*2):
			all_possible_scopes = itertools.combinations_with_replacement(list(range(self.MDP.nstatefactors)), i)
			for scope in all_possible_scopes:
				flag = True
				for j in range(len(scope)-1):
					if scope[j] == scope[j+1]:
						flag = False
						break
				if flag:
					self.possible_scopes.append(list(scope)+[self.MDP.nstatefactors])
			if i == self.scope_size-1:
				self.num_scopes_of_size_m = len(self.possible_scopes)
		self.num_scopes_total = len(self.possible_scopes)

		# counts and sums for reward factors
		self.XR = MDP.rewardsizes();
		self.NR = [None] * MDP.nrewardfactors
		self.SR = [None] * MDP.nrewardfactors
		for i in range( MDP.nrewardfactors ):
			self.NR[i] = np.zeros( self.XR[i] )
			self.SR[i] = np.zeros( self.XR[i] )

		# counts for state factors
		self.XP = MDP.transitionsizes();
		self.NP = [None] * MDP.nstatefactors
		for i in range( MDP.nstatefactors ):
			self.NP[i] = np.zeros( ( self.XP[i], MDP.factordomains[i] ) )

		# counts that determine whether to update policy
		self.N_prev = [None] * ( MDP.nrewardfactors + MDP.nstatefactors )
		self.v      = [None] * ( MDP.nrewardfactors + MDP.nstatefactors )

		# consistent scopes
		self.consistent_scopes = [None] * self.num_factors_unknown
		for i in range( self.num_factors_unknown ):
			self.consistent_scopes[i] = [True] * self.num_scopes_of_size_m

		# counts for unknown state factors
		self.unknown_NP = [None] * self.num_factors_unknown
		self.elimination_times = [None] * self.num_factors_unknown
		for i in range( self.num_factors_unknown ):
			self.unknown_NP[i] = [None] * self.num_scopes_total
			self.elimination_times[i] = [-1] * self.num_scopes_of_size_m
			for Z in range(self.num_scopes_total):
				self.unknown_NP[i][Z] = np.zeros( ( self.MDP.nelements(self.possible_scopes[Z]), MDP.factordomains[self.unknown_factors[i]] ) )

		# counts that determine whether to update policy for unknown factors
		self.unknown_N_prev = [None] * self.num_scopes_total
		self.unknown_v      = [None] * self.num_scopes_total
		for Z in range(self.num_scopes_total):
			self.unknown_N_prev[Z] = np.zeros( self.MDP.nelements(self.possible_scopes[Z]) )
			self.unknown_v[Z] = np.zeros( self.MDP.nelements(self.possible_scopes[Z]) )

	def updatepolicy( self, delta, t ):
		# compute upper bound on full reward
		r_upper = np.zeros( self.MDP.nstates * self.MDP.nactions )

		for i in range( self.MDP.nrewardfactors ):
			# compute local reward factor estimate
			Nplus      = np.maximum( self.NR[i], 1 )
			r_tot_hat  = self.SR[i] / Nplus

			# compute confidence bounds on local reward factor
			# if desired, "self.MDP.rewardstruct[i].params" allows access to true reward
			real_delta = delta / ( self.MDP.nrewardfactors * np.size( Nplus ) )
			r_tilde    = self.rewardbound( 1, r_tot_hat, Nplus, self.MDP.rewardstruct[i].params, real_delta, t )
 
			# update bound on full reward
			r_upper    = r_upper + ( self.MDP.rewardstruct[i].mapping @ r_tilde )

			# update counts
			self.N_prev[i] = np.copy ( self.NR[i] )
			self.v     [i] = np.zeros( self.XR[i] )

		# compute estimate and bounds on full transition kernel
		p_lower = np.ones ( ( self.MDP.nstates * self.MDP.nactions, self.MDP.nstates ) )
		p_hat   = np.ones ( ( self.MDP.nstates * self.MDP.nactions, self.MDP.nstates ) )
		p_upper = np.ones ( ( self.MDP.nstates * self.MDP.nactions, self.MDP.nstates ) )

		for i in range( self.MDP.nstatefactors ):
			# compute local state factor estimate
			NP_sum          = np.sum( self.NP[i], 1 )
			ix              = NP_sum > 0
			p_tot_hat       = np.ones( ( self.XP[i], self.MDP.factordomains[i] ) ) / self.MDP.factordomains[i]
			p_tot_hat[ix,:] = self.NP[i][ix,:] / NP_sum[ix,None]

			# compute confidence bounds on local state factor
			Nplus      = np.maximum( NP_sum, 1 )
			real_delta = delta / ( self.MDP.nstatefactors * np.size( Nplus ) )
			LB, UB     = self.transitionbound( self.MDP.factordomains[i], p_tot_hat, Nplus, self.NP[i], real_delta, t )

			# update bounds on full transition kernel
			p_lower = p_lower * ( self.MDP.transitionstruct[i].mapping @ LB         @ self.MDP.statemappings[i] )
			p_hat   = p_hat   * ( self.MDP.transitionstruct[i].mapping @ p_tot_hat  @ self.MDP.statemappings[i] )
			p_upper = p_upper * ( self.MDP.transitionstruct[i].mapping @ UB         @ self.MDP.statemappings[i] )

			# update counts
			self.N_prev[self.MDP.nrewardfactors + i] = np.copy ( NP_sum     )
			self.v     [self.MDP.nrewardfactors + i] = np.zeros( self.XP[i] )

		# update counts
		for Z in range(self.num_scopes_total):
			self.unknown_N_prev[Z] = self.unknown_N_prev[Z] + self.unknown_v[Z]
			self.unknown_v[Z] *= 0

		# do not continue if all wrong scopes were eliminted 
		if  np.count_nonzero(self.consistent_scopes) == self.num_factors_unknown:
			return self.EVI.computepolicy( self.MDP, r_upper, p_lower, p_hat, p_upper, t )

		# eliminte inconsistent scopes
		for i in range(self.num_factors_unknown):
			for Z in range(self.num_scopes_of_size_m):
				if not self.consistent_scopes[i][Z]:
					continue

				NP_sum = np.sum( self.unknown_NP[i][Z], 1 )
				ix = NP_sum > 0
				p_tot_hat_Z = np.ones( ( self.MDP.nelements(self.possible_scopes[Z]), self.MDP.factordomains[self.unknown_factors[i]] ) ) / self.MDP.factordomains[self.unknown_factors[i]]
				p_tot_hat_Z[ix,:] = self.unknown_NP[i][Z][ix,:] / NP_sum[ix,None]
				
				for Z_tag in range(self.num_scopes_of_size_m,self.num_scopes_total):
					if not set(self.possible_scopes[Z]).issubset(self.possible_scopes[Z_tag]):
						continue

					NP_sum = np.sum( self.unknown_NP[i][Z_tag], 1 )
					ix = NP_sum > 0
					p_tot_hat_Z_tag = np.ones( ( self.MDP.nelements(self.possible_scopes[Z_tag]), self.MDP.factordomains[self.unknown_factors[i]] ) ) / self.MDP.factordomains[self.unknown_factors[i]]
					p_tot_hat_Z_tag[ix,:] = self.unknown_NP[i][Z_tag][ix,:] / NP_sum[ix,None]

					tmp = np.ones( ( self.MDP.nelements(self.possible_scopes[Z_tag]), self.MDP.factordomains[self.unknown_factors[i]] ) )
					tmp[ix,:] = tmp[ix,:] / NP_sum[ix,None]
					tmp = tmp * 18 * np.log(6 * self.MDP.nstatefactors * self.XP[self.unknown_factors[i]] * t / delta)
					eps_Z_tag = tmp + np.sqrt(tmp * p_tot_hat_Z_tag)

					for sa in range( self.MDP.nelements( range( len( self.MDP.factordomains ) ) ) ):
						stateAction = self.MDP.decode( sa, range( len( self.MDP.factordomains ) ) )
						sa_Z = self.MDP.encode( stateAction, list(self.possible_scopes[Z]) )
						stateAction_Z = self.MDP.decode( sa_Z, list(self.possible_scopes[Z]) )
						sa_Z_tag = self.MDP.encode( stateAction, list(self.possible_scopes[Z_tag]) )
						stateAction_Z_tag = self.MDP.decode( sa_Z_tag, list(self.possible_scopes[Z_tag]) )
						if set(stateAction_Z).issubset(set(stateAction_Z_tag)) and self.consistent_scopes[i][Z] and np.max(np.abs(p_tot_hat_Z_tag[sa_Z_tag] - p_tot_hat_Z[sa_Z]) - 2*eps_Z_tag[sa_Z_tag]/10) > 0 and np.count_nonzero(self.consistent_scopes[i]) > 1:
							print(str(t) + " : ELIMINATED " + str(self.unknown_factors[i]) + " : " + str(np.count_nonzero(self.consistent_scopes[i]) - 1))
							self.consistent_scopes[i][Z] = False
							self.elimination_times[i][Z] = t
							if set(self.possible_scopes[Z]) == set(self.MDP.transitionstruct[self.unknown_factors[i]].scope):
								print("WRONG SCOPE")
							break

		# handle each consistent scope of each unknown factor
		best_pi = None
		best_max_V = - 1
		for Z in itertools.product(range(self.num_scopes_of_size_m),repeat=self.num_factors_unknown):
			consistent_flag = True
			for i in range(self.num_factors_unknown):
				if not self.consistent_scopes[i][Z[i]]:
					consistent_flag = False
					break
			if not consistent_flag:
				continue

			p_lower_n = p_lower.copy()
			p_hat_n   = p_hat.copy()
			p_upper_n = p_upper.copy()
			MDP_copy = copy.deepcopy(self.MDP)

			for sa in range( MDP_copy.nelements( range( len( MDP_copy.factordomains ) ) ) ):
				stateAction = MDP_copy.decode( sa, range( len( MDP_copy.factordomains ) ) )
				for i in range( self.num_factors_unknown ):
					MDP_copy.transitionstruct[self.unknown_factors[i]].mapping[sa, MDP_copy.encode( stateAction, MDP_copy.transitionstruct[self.unknown_factors[i]].scope )] = 0
					MDP_copy.transitionstruct[self.unknown_factors[i]].mapping[sa, MDP_copy.encode( stateAction, list(self.possible_scopes[Z[i]]) )] = 1

			for i in range(self.num_factors_unknown):
				NP_sum = np.sum( self.unknown_NP[i][Z[i]], 1 )
				ix = NP_sum > 0
				p_tot_hat_Z = np.ones( ( MDP_copy.nelements(self.possible_scopes[Z[i]]), MDP_copy.factordomains[self.unknown_factors[i]] ) ) / MDP_copy.factordomains[self.unknown_factors[i]]
				p_tot_hat_Z[ix,:] = self.unknown_NP[i][Z[i]][ix,:] / NP_sum[ix,None]
				Nplus      = np.maximum( NP_sum, 1 )
				real_delta = delta / ( MDP_copy.nstatefactors * np.size( Nplus ) )
				LB, UB     = self.transitionbound( MDP_copy.factordomains[self.unknown_factors[i]], p_tot_hat_Z, Nplus, self.unknown_NP[i][Z[i]], real_delta, t )

				p_lower_n = p_lower_n * ( MDP_copy.transitionstruct[self.unknown_factors[i]].mapping @ LB         @ MDP_copy.statemappings[self.unknown_factors[i]] )
				p_hat_n   = p_hat_n   * ( MDP_copy.transitionstruct[self.unknown_factors[i]].mapping @ p_tot_hat_Z  @ MDP_copy.statemappings[self.unknown_factors[i]] )
				p_upper_n = p_upper_n * ( MDP_copy.transitionstruct[self.unknown_factors[i]].mapping @ UB         @ MDP_copy.statemappings[self.unknown_factors[i]] )

			# run Extended Value Iteration
			pi, max_V = self.EVI.computepolicy( MDP_copy, r_upper, p_lower_n, p_hat_n, p_upper_n, t ,True)
			if best_pi is None or (max_V >= best_max_V and np.random.random() <= 0.5):
				best_max_V = max_V
				best_pi = pi
		
		return best_pi


	def updateparams( self, s, a, r, sp ):
		fs  = self.MDP.decode(  s, range( self.MDP.nstatefactors ) )
		fa  = self.MDP.decode(  a, range( self.MDP.nstatefactors, len( self.MDP.factordomains ) ) )
		fsp = self.MDP.decode( sp, range( self.MDP.nstatefactors ) )
		fsa = np.concatenate( ( fs, fa ) )

		compute_pi = False

		# update counts and sums for reward factors
		IR = self.MDP.rewardindices( fsa )
		for i in range( self.MDP.nrewardfactors ):
			self.NR[i][IR[i]] = self.NR[i][IR[i]] + 1
			self.SR[i][IR[i]] = self.SR[i][IR[i]] + r[i]

			self.v [i][IR[i]] = self.v [i][IR[i]] + 1
			if self.v[i][IR[i]] >= self.N_prev[i][IR[i]]:
				compute_pi = True

		# update counts for state factors
		IP = self.MDP.transitionindices( fsa )
		for i in range( self.MDP.nstatefactors ):
			self.NP[i][IP[i],fsp[i]] = self.NP[i][IP[i],fsp[i]] + 1

			self.v[self.MDP.nrewardfactors + i][IP[i]] = self.v[self.MDP.nrewardfactors + i][IP[i]] + 1
			if self.v[self.MDP.nrewardfactors + i][IP[i]] >= self.N_prev[self.MDP.nrewardfactors + i][IP[i]]:
				compute_pi = True

		# update counts for state factors unknown
		for Z in range(self.num_scopes_total):
			scope_val = self.MDP.encode( fsa, list(self.possible_scopes[Z]) )
			self.unknown_v[Z][scope_val] = self.unknown_v[Z][scope_val] + 1
			if self.unknown_v[Z][scope_val] >= self.unknown_N_prev[Z][scope_val] and np.count_nonzero(self.consistent_scopes) > self.num_factors_unknown:
				compute_pi = True
			for i in range( self.num_factors_unknown ):
				self.unknown_NP[i][Z][scope_val,fsp[self.unknown_factors[i]]] = self.unknown_NP[i][Z][scope_val,fsp[self.unknown_factors[i]]] + 1
		
		return compute_pi

