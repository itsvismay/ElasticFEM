from Helpers import *
import Meshwork
import Arap
import Neo
import Display
import Solvers
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.5f}'.format(x)})

def FiniteDifferencesARAP():
	eps = 1e-1
	its = 100

	VTU = Meshwork.rectangle_mesh(x=2, y=2, step=0.1)
	mw = Meshwork.Preprocessing(_VT = VTU)
	mw.Fix = get_max(mw.V, a=1, eps=1e-2)		
	mw.Mov = get_min(mw.V, a=1, eps=1e-2)
	mesh = mw.getMesh()
	arap = Arap.ARAP(imesh = mesh, filen="crap/")
	E0 = arap.energy(_z=mesh.z, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
	print("Default Energy ", E0)

	def check_dEdg():
		real = arap.dEdg()
		dEdz = []
		z = np.zeros(len(mesh.z)) + mesh.z
		for i in range(len(mesh.z)):
			z[i] += 0.5*eps
			Eleft = arap.energy(_z=z, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
			z[i] -= 0.5*eps
			z[i] -= 0.5*eps
			Eright = arap.energy(_z=z, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
			z[i] += 0.5*eps
			dEdz.append((Eleft - Eright)/eps)

		print(np.array(dEdz))
		print(real)
		print("Eg ", np.linalg.norm(real - np.array(dEdz)))

	def check_dEds():
		realdEdS, realdEds = arap.dEds()
		dEds = []
		for i in range(len(mesh.red_s)):
			mesh.red_s[i] += eps
			arap.updateConstUSUtPAx()
			mesh.getGlobalF()
			Ei = arap.energy(_z =mesh.z, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
			dEds.append((Ei - E0)/eps)
			mesh.red_s[i] -= eps

		print(realdEds)
		print(np.array(dEds))
		print("Es ", np.sum(np.array(dEds)-realdEds))

	def check_dEdr():
		realdEdR, realdEdr = arap.dEdr()
		dEdr = []
		for i in range(len(mesh.red_r)):
			mesh.red_r[i] += 0.5*eps
			mesh.getGlobalF()
			Eleft = arap.energy(_z =mesh.z, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
			mesh.red_r[i] -= 0.5*eps

			mesh.red_r[i] -= 0.5*eps
			mesh.getGlobalF()
			Eright = arap.energy(_z =mesh.z, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
			mesh.red_r[i] += 0.5*eps

			dEdr.append((Eleft - Eright)/eps)

		print(realdEdr)
		print(np.array(dEdr))
		print("Er ", np.sum(np.array(dEdr) - realdEdr))

	def check_Hessian_dEdgdg():
		real = arap.Hess_Egg()

		Egg = []
		dg = np.zeros(len(mesh.z)) + mesh.z
		for i in range(len(mesh.z)):
			Egg.append([])
			for j in range(len(mesh.z)):
				dg[i] += eps
				dg[j] += eps
				Eij = arap.energy(_z=dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				dg[i] -= eps
				dg[j] -= eps

				dg[i] += eps
				Ei = arap.energy(_z=dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				dg[i] -= eps

				dg[j] += eps
				Ej = arap.energy(_z=dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				dg[j] -= eps

				Egg[i].append((Eij - Ei - Ej + E0)/(eps*eps))
		
		# print("Egg")
		print(real)
		print("Egg ", np.sum(np.array(Egg) - real))

	def check_Hessian_dEdrdg():
		real = arap.Hess_Erg()

		Erg = []
		dg = np.zeros(len(mesh.z)) + mesh.z
		for i in range(len(mesh.z)):
			Erg.append([])
			for j in range(len(mesh.red_r)):
				dg[i] += eps
				mesh.red_r[j] += eps
				mesh.getGlobalF()
				Eij = arap.energy(_z =dg, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[j] -= eps
				dg[i] -= eps

				dg[i] += eps
				mesh.getGlobalF()
				Ei = arap.energy(_z =dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				dg[i] -= eps

				mesh.red_r[j] += eps
				mesh.getGlobalF()
				Ej = arap.energy(_z =dg, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[j] -= eps


				Erg[i].append((Eij - Ei - Ej + E0)/(eps*eps))

		print(real)
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
				Eij = arap.energy(_z =mesh.z, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[i] -= eps
				mesh.red_r[j] -= eps

				mesh.red_r[j] += eps
				mesh.getGlobalF()
				Ej = arap.energy(_z =mesh.z, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[j] -= eps

				mesh.red_r[i] += eps
				mesh.getGlobalF()
				Ei = arap.energy(_z =mesh.z, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[i] -= eps

				Err[i].append((Eij - Ei - Ej + E0)/(eps*eps))
		print(real)
		print(np.array(Err))
		print("Err ", np.sum(np.array(Err) - real))
	
	def check_Hessian_dEdrds():
		real = arap.Hess_Ers()
		Ers = []
		for i in range(len(mesh.red_r)):
			Ers.append([])
			for j in range(len(mesh.red_s)):
				mesh.red_r[i] += eps
				mesh.red_s[j] += eps
				mesh.getGlobalF()
				Eij = arap.energy(_z =mesh.z, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_s[j] -= eps
				mesh.red_r[i] -= eps

				mesh.red_r[i] += eps
				mesh.getGlobalF()
				Ei = arap.energy(_z =mesh.z, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_r[i] -= eps

				mesh.red_s[j] += eps
				mesh.getGlobalF()
				Ej = arap.energy(_z =mesh.z, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_s[j] -= eps
				mesh.getGlobalF()

				Ers[i].append((Eij - Ei - Ej + E0)/(eps*eps))
		print(real)
		print("")
		print(np.array(Ers))
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
		dg = np.zeros(len(mesh.z)) + mesh.z
		for i in range(len(mesh.z)):
			Egs.append([])
			for j in range(len(mesh.red_s)):
				dg[i] += eps
				mesh.red_s[j] += eps
				mesh.getGlobalF(updateR = False, updateS = True, updateU = False)
				Eij = arap.energy(_z =dg, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_s[j] -= eps
				dg[i] -= eps

				dg[i] += eps
				mesh.getGlobalF(updateR = False, updateS = True, updateU = False)
				Ei = arap.energy(_z =dg, _R =mesh.GR, _S=mesh.GS, _U=mesh.GU)
				dg[i] -= eps

				mesh.red_s[j] += eps
				mesh.getGlobalF(updateR = False, updateS = True, updateU = False)
				Ej = arap.energy(_z =dg, _R=mesh.GR, _S=mesh.GS, _U=mesh.GU)
				mesh.red_s[j] -= eps

				Egs[i].append((Eij - Ei - Ej + E0)/(eps*eps))
		
		print(real)
		print("\n")
		print(np.array(Egs))
		print("Egs ", np.linalg.norm(np.array(Egs) - real))

	def check_dgds_drds():


		Jac, real1, real2 = arap.Jacobian()
		dgds = []
		drds = []

		z0 = np.zeros(len(mesh.z)) + mesh.z
		g0 = np.zeros(len(mesh.g)) + mesh.g
		r0 = np.array(mesh.red_r) 
		for i in range(0,len(mesh.red_s)):
			mesh.z = np.zeros(len(mesh.z)) + z0

			mesh.red_s[i] += 0.5*eps
			mesh.getGlobalF()
			arap.updateConstUSUtPAx()
			arap.iterate()
			# print(mesh.red_r)
			# print(arap.Energy())
			# exit()
			drds_left = np.array(mesh.red_r)
			dgds_left =mesh.z + np.zeros(len(mesh.z))
	
			mesh.red_s[i] -= 0.5*eps
			mesh.getGlobalF()
			arap.updateConstUSUtPAx()
			arap.iterate()

			mesh.red_s[i] -= 0.5*eps
			mesh.getGlobalF()
			arap.updateConstUSUtPAx()
			arap.iterate()
			drds_right = np.array(mesh.red_r)
			dgds_right =mesh.z + np.zeros(len(mesh.z))
	
			mesh.red_s[i] += 0.5*eps
			mesh.getGlobalF()
			arap.updateConstUSUtPAx()
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
	# check_Hessian_dEdgds()
	# check_Hessian_dEdrds()
	check_dgds_drds()

FiniteDifferencesARAP()

def FiniteDifferencesElasticity():
	eps = 1e-1
	its = 100
	# VTU,tofix = Meshwork.feather_muscle2_test_setup()
	VTU = Meshwork.rectangle_mesh(x=2, y=2, step=0.1)
	mw = Meshwork.Preprocessing(_VT = VTU)
	mw.Fix = get_max(mw.V, a=1, eps=1e-2)		
	mw.Mov = get_min(mw.V, a=1, eps=1e-2)
	mesh = mw.getMesh()
	arap = Arap.ARAP(imesh = mesh, filen="crap/")
	# mesh.red_s[2] = 0.1
	ne = Neo.NeohookeanElastic(imesh = mesh)

	def check_PrinStretchForce():
		e0 = ne.WikipediaEnergy(_rs = mesh.red_s)
		real = -ne.WikipediaForce(_rs = mesh.red_s)
		dEds = []
		for i in range(len(mesh.red_s)):
			mesh.red_s[i] += 0.5*eps
			left = ne.WikipediaEnergy(_rs=mesh.red_s)
			mesh.red_s[i] -= 0.5*eps
			
			mesh.red_s[i] -= 0.5*eps
			right = ne.WikipediaEnergy(_rs=mesh.red_s)
			mesh.red_s[i] += 0.5*eps

			dEds.append((left - right)/(eps))

		print("real", real)
		print("fake", dEds)
		print("Diff", np.linalg.norm(real - np.array(dEds)))

	def check_gravityForce():
		e0 = ne.GravityEnergy()
		print("E0", e0)
		arap.iterate()
		
		real = -1*ne.GravityForce(dzds=arap.Jacobian()[1])

		dEgds = []
		for i in range(len(mesh.red_s)):
			mesh.z = np.zeros(len(mesh.z))	
			mesh.red_s[i] += eps
			mesh.getGlobalF(updateR=False, updateS=True)
			arap.updateConstUSUtPAx()
			arap.iterate()
			e1 = ne.GravityEnergy()
			dEgds.append((e1 - e0)/eps)
			mesh.red_s[i] -= eps
			mesh.getGlobalF(updateR=False, updateS=True)
			arap.updateConstUSUtPAx()
			arap.iterate()

		print("real", real)
		print("fake", dEgds)
		print("Diff", np.sum(real - np.array(dEgds)))

	def check_muscleForce():
		
		e0 = ne.MuscleEnergy(_rs = mesh.red_s)
		real = -ne.MuscleForce(_rs = mesh.red_s)
		print("e0", e0)
		dEds = []
		for i in range(len(mesh.red_s)):
			mesh.red_s[i] += 0.5*eps
			left = ne.MuscleEnergy(_rs=mesh.red_s)
			mesh.red_s[i] -= 0.5*eps
			
			mesh.red_s[i] -= 0.5*eps
			right = ne.MuscleEnergy(_rs=mesh.red_s)
			mesh.red_s[i] += 0.5*eps

			dEds.append((left - right)/(eps))

		print("real", real)
		print("fake", dEds)
		print("Diff", np.sum(real - np.array(dEds)))

	# check_PrinStretchForce()
	check_gravityForce()
	# check_muscleForce()

# FiniteDifferencesElasticity()
