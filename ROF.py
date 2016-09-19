import numpy as np

def nabla(I) :
	(h, w) = I.shape
	k = np.zeros((h, w, 2), I.dtype)
	k[:, :-1, 0] -= I[:, :-1]
	k[:, :-1, 0] += I[:, 1:]
	k[:-1, :, 1] -= I[:-1]
	k[:-1, :, 1] += I[1:]
	return k
	
def nablaT(k) :
	(h, w) = k.shape[:2]
	I = np.zeros((h, w), k.dtype)
	I[:, :-1] -= k[:, :-1, 0]
	I[:, 1:] += k[:, :-1, 0]
	I[:-1] -= k[:-1, :, 1]
	I[1:] += k[:-1, :, 1]
	return I
	
def anorm(x) :
	return(np.sqrt((x * x).sum(-1)))

def energy_ROF(x, obser, clambda) :
	Ereg = anorm(nabla(x)).sum()
	Edata = 0.5 * clambda * ((x - obser) ** 2).sum()
	return(Ereg + Edata)

def project_nd(P, r) :
	nP = np.maximum(1.0, anorm(P) / r)
	return(P / nP[..., np.newaxis])
	
def shrink_1d(x, f, step) :
	return(x + np.clip(f, -x, -step, step))
	
def solve_ROF(img, clambda, iter_n = 201) :
	L2 = 8.0
	tau = 0.02
	sigma = 1.0 / (L2 * tau)
	theta = 1.0
	
	x = img.copy()
	P = nabla(x)
	for i in range(iter_n) :
		P = project_nd(P + sigma * nabla(x), 1.0)
		lt = clambda * tau
		x1 = (x - tau * nablaT(P) + lt * img) / (1.0 + lt)
		x = x1 + theta * (x1 - x)
		if(i % 10 == 0) :
			print(energy_ROF(x, img, clambda))
			
	return x
	