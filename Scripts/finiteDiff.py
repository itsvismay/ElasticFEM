import numpy as np
from version2 import rectangle_mesh, torus_mesh, featherize, get_min_max, NeohookeanElastic, ARAP, Mesh

def FiniteDifferencesARAP():
	eps = 1e-4
	iV, iT, iU = featherize(1,3,.1)
	# iV, iT, iU = torus_mesh(5,4,3,.1)
	its = 100
	to_fix = get_min_max(iV, a=1)
	# print(to_fix)
	mesh = Mesh((iV,iT, iU),ito_fix=to_fix)
	mesh.fixed = mesh.fixed_max_axis(1)
	# print(mesh.fixed)
	
	arap = ARAP(mesh)	
	mesh.getGlobalF()

	E0 = arap.energy(_g=mesh.g, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
	print("Default Energy ", E0)
	
	def check_dEdg():
		dEdg = []
		dg = np.zeros(len(mesh.g)) + mesh.g
		for i in range(len(mesh.g)):
			dg[i] += eps
			mesh.getGlobalF()
			Ei = arap.energy(_g=dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
			dEdg.append((Ei - E0)/eps)
			dg[i] -= eps

		print("Eg ", np.linalg.norm(arap.dEdg() - np.array(dEdg)))

	def check_dEds():
		realdEdS, realdEds = arap.dEds()
		dEds = []
		for i in range(len(mesh.T)):
			mesh.q[3*i+1] += eps
			mesh.getGlobalF()
			Ei = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
			dEds.append((Ei - E0)/eps)
			mesh.q[3*i+1] -= eps

			mesh.q[3*i+2] += eps
			mesh.getGlobalF()
			Ei = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
			dEds.append((Ei - E0)/eps)
			mesh.q[3*i+2] -=eps
		print(realdEds)
		print("Es ", np.sum(np.array(dEds)-realdEds))

	def check_dEdS():
		dEdS_real, dEds_real = arap.dEds()
		F,R,S,U = mesh.getGlobalF()
		
		S[0,0] += eps
		Ei = arap.energy(_g=mesh.g, _R =R0, _S=S, _U=U0)
		print((Ei - E0)/eps - dEdS_real[0,0])

	def check_dEdr():
		realdEdR, realdEdr = arap.dEdr()
		dEdr = []
		for i in range(len(mesh.red_r)):
			mesh.red_r[i] += 0.5*eps
			mesh.getGlobalF()
			Eleft = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
			mesh.red_r[i] -= 0.5*eps

			mesh.red_r[i] -= 0.5*eps
			mesh.getGlobalF()
			Eright = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
			mesh.red_r[i] += 0.5*eps

			dEdr.append((Eleft - Eright)/eps)
		print(realdEdr)
		print("Er ", np.sum(np.array(dEdr) - realdEdr))

	def check_Hessian_dEdgdg():
		real = arap.Hess_Egg()

		Egg = []
		dg = np.zeros(len(mesh.g)) + mesh.g
		for i in range(len(mesh.g)):
			Egg.append([])
			for j in range(len(mesh.g)):
				dg[i] += eps
				dg[j] += eps
				Eij = arap.energy(_g=dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				dg[i] -= eps
				dg[j] -= eps

				dg[i] += eps
				Ei = arap.energy(_g=dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				dg[i] -= eps

				dg[j] += eps
				Ej = arap.energy(_g=dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				dg[j] -= eps

				Egg[i].append((Eij - Ei - Ej + E0)/(eps*eps))
		
		# print("Egg")
		# print(Egg)
		print("Egg ", np.sum(np.array(Egg) - real))

	def check_Hessian_dEdrdg():
		real = arap.Hess_Erg()

		Erg = []
		dg = np.zeros(len(mesh.g)) + mesh.g
		for i in range(len(mesh.g)):
			Erg.append([])
			for j in range(len(mesh.red_r)):
				dg[i] += eps
				mesh.red_r[j] += eps
				mesh.getGlobalF()
				Eij = arap.energy(_g =dg, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[j] -= eps
				dg[i] -= eps

				dg[i] += eps
				mesh.getGlobalF()
				Ei = arap.energy(_g=dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				dg[i] -= eps

				mesh.red_r[j] += eps
				mesh.getGlobalF()
				Ej = arap.energy(_g =dg, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[j] -= eps


				Erg[i].append((Eij - Ei - Ej + E0)/(eps*eps))

		print("Erg ",np.sum(np.array(Erg) - real))

	def check_Hessian_dEdrdr():
		real = arap.Hess_Err()

		Err = []
		for i in range(len(mesh.red_r)):
			Err.append([])
			for j in range(len(mesh.red_r)):
				mesh.red_r[i] += eps
				mesh.red_r[j] += eps
				mesh.getGlobalF()
				Eij = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[i] -= eps
				mesh.red_r[j] -= eps

				mesh.red_r[j] += eps
				mesh.getGlobalF()
				Ej = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[j] -= eps

				mesh.red_r[i] += eps
				mesh.getGlobalF()
				Ei = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[i] -= eps

				Err[i].append((Eij - Ei - Ej + E0)/(eps*eps))
		# print(real)
		print("Err ", np.sum(np.array(Err) - real))
	
	def check_Hessian_dEdrds():
		real = arap.Hess_Ers()

		Ers = []
		for i in range(len(mesh.red_r)):
			Ers.append([])
			for j in range(len(mesh.T)):
				for k in range(1, 3):
					mesh.red_r[i] += eps
					mesh.q[3*j + k] += eps
					mesh.getGlobalF()
					Eij = arap.energy(_g =mesh.g, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
					mesh.q[3*j + k] -= eps
					mesh.red_r[i] -= eps

					mesh.red_r[i] += eps
					mesh.getGlobalF()
					Ei = arap.energy(_g =mesh.g, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
					mesh.red_r[i] -= eps

					mesh.q[3*j + k] += eps
					mesh.getGlobalF()
					Ej = arap.energy(_g =mesh.g, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
					mesh.q[3*j + k] -= eps
					mesh.getGlobalF()

					Ers[i].append((Eij - Ei - Ej + E0)/(eps*eps))
		# print(real)
		# print("")
		# print(np.array(Ers))
		print("Ers ", np.linalg.norm(np.array(Ers) - real))

	def unred_check_Hessian_dEdrds():
		real = arap.Hess_Ers()

		Ers = []
		for i in range(len(mesh.T)):
			Ers.append([])
			for j in range(len(mesh.T)):
				for k in range(1, 3):
					mesh.q[3*i] += eps
					mesh.q[3*j + k] += eps
					mesh.getGlobalF()

					Eij = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
					mesh.q[3*j + k] -= eps
					mesh.q[3*i] -= eps

					mesh.q[3*i] += eps
					mesh.getGlobalF()
					Ei = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
					mesh.q[3*i] -= eps

					mesh.q[3*j + k] += eps
					mesh.getGlobalF()
					Ej = arap.energy(_g =mesh.g, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
					mesh.q[3*j + k] -= eps

					Ers[i].append((Eij - Ei - Ej + E0)/(eps*eps))

		print(real)
		print("")
		print(np.array(Ers))
		print("Ers ", np.linalg.norm(np.array(Ers) - real))

	def check_Hessian_dEdgds():
		real = arap.Hess_Egs()

		Egs = []
		dg = np.zeros(len(mesh.g)) + mesh.g
		for i in range(len(mesh.g)):
			Egs.append([])
			for j in range(len(mesh.T)):
				for k in range(1,3):
					dg[i] += eps
					mesh.q[3*j+k] += eps
					mesh.getGlobalF()
					Eij = arap.energy(_g =dg, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
					mesh.q[3*j+k] -= eps
					dg[i] -= eps

					dg[i] += eps
					mesh.getGlobalF()
					Ei = arap.energy(_g=dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
					dg[i] -= eps

					mesh.q[3*j+k] += eps
					mesh.getGlobalF()
					Ej = arap.energy(_g =dg, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
					mesh.q[3*j+k] -= eps

					Egs[i].append((Eij - Ei - Ej + E0)/(eps*eps))
		# print(real)
		print("Egs ", np.sum(np.array(Egs) - real))

	def check_dgds_drds():
		Jac, real1, real2 = arap.Jacobian()

		dgds = []
		drds = []
		g0 = np.zeros(len(mesh.g)) + mesh.g
		r0 = np.array(mesh.red_r) 
		q0 = np.zeros(len(mesh.q)) + mesh.q
		for i in range(len(mesh.T)):
			for j in range(1,3):
				mesh.g = np.zeros(len(mesh.g)) + g0

				mesh.q[3*i + j] += 0.5*eps
				mesh.getGlobalF()

				arap.iterate()
				drds_left = np.array(mesh.red_r)
				dgds_left =mesh.g + np.zeros(len(mesh.g))

				mesh.q[3*i + j] -= 0.5*eps
				mesh.getGlobalF()
				arap.iterate()

				mesh.q[3*i + j] -= 0.5*eps
				mesh.getGlobalF()
				arap.iterate()
				drds_right = np.array(mesh.red_r)
				dgds_right =mesh.g + np.zeros(len(mesh.g))
				mesh.q[3*i + j] += 0.5*eps
				mesh.getGlobalF()
				arap.iterate()


				dgds.append((dgds_left - dgds_right)/(eps))
				drds.append((drds_left - drds_right)/(eps))
				# exit()
				

		print("FD")
		print(np.array(dgds).T)
		print(np.array(drds).T)
		print("")
		print("real")
		print(real1)
		print(real2)
		print("DIFF")
		# print("T: ", len(mesh.T))
		print("dgds:", np.linalg.norm(real1 - np.array(dgds).T))
		print("drds:", np.linalg.norm(real2 - np.array(drds).T))
		# print("its: ", its)
		# print("Energy: ", arap.Energy())
		# print("grad",  np.linalg.norm(arap.dEdg()), np.linalg.norm(arap.dEdr()[1]))


	# check_dEdg()
	# check_dEds()
	# check_dEdr()

	# check_Hessian_dEdgdg()
	# check_Hessian_dEdrdg()
	# check_Hessian_dEdrdr()
	# check_Hessian_dEdrds()
	# check_Hessian_dEdgds()
	check_dgds_drds()

FiniteDifferencesARAP()

def FiniteDifferencesElasticity():
	mdim = 2
	eps = 1e-5
	iV, iT, iU = rectangle_mesh(mdim, mdim, .1)
	its = 50
	to_fix = get_min_max(iV, a=1)

	mesh = Mesh((iV,iT, iU),ito_fix=to_fix)
	mesh.fixed = mesh.fixed_max_axis(1)
	arap = ARAP(imesh=mesh)
	ne = NeohookeanElastic(imesh = mesh)
	

	def check_PrinStretchForce():
		e0 = ne.PrinStretchEnergy(_q = mesh.q)
		real = ne.PrinStretchForce(_q = mesh.q)
		print("e0", e0)
		dEds = []
		for i in range(len(mesh.T)):
			for j in range(1,3):
				mesh.q[3*i+j] += eps
				left = ne.PrinStretchEnergy(_q=mesh.q)
				mesh.q[3*i+j] -= eps

				mesh.q[3*i+j] -= eps
				right = ne.PrinStretchEnergy(_q=mesh.q)
				mesh.q[3*i+j] += eps

				dEds.append((left - right)/(2*eps))

		print("real", real)
		print("fake", dEds)
		print("Diff", np.sum(real - np.array(dEds)))

	def check_gravityForce():
		e0 = ne.GravityEnergy()
		print("E0", e0)
		arap.iterate()
		
		real = -1*ne.GravityForce(dgds=arap.Jacobian()[1])

		dEgds = []
		for i in range(len(mesh.T)):
			for j in range(1,3):
				mesh.g = np.zeros(len(mesh.g)) + mesh.x0
				mesh.q[3*i+j] += eps
				arap.iterate()
				e1 = ne.GravityEnergy()
				dEgds.append((e1 - e0)/eps)
				mesh.q[3*i+j] -= eps
				arap.iterate()

		print("real", real)
		print("fake", dEgds)
		print("Diff", np.sum(real - np.array(dEgds)))

	check_PrinStretchForce()
	check_gravityForce()
	# test()

# FiniteDifferencesElasticity()
