import torch
import pandas as pd
import numpy as np
from src.net.chemtools.PLS import PLS, getknn
import matplotlib.pyplot as plt

class LWPLSR:
    def __init__(self, ncompdis=None, diss="euclidean", h=5, k=None, ncomp=2, cri=3, stor=True, print_out=True) :
        self.ncompdis = ncompdis if ncompdis is not None else 0
        self.diss = diss
        self.h = h if isinstance(h, list) else [h]
        self.k = k
        self.ncomp = ncomp
        self.cri = cri
        self.stor = stor
        self.print_out = print_out

    def fit(self, Xr, Yr, Xu) :
        Xr = torch.tensor(Xr, dtype=torch.float64)
        Xu = torch.tensor(Xu, dtype=torch.float64)
        Yr = torch.tensor(Yr, dtype=torch.float64)
   

        n = Xr.shape[0]
        m = Xu.shape[0]


        self.k = min(self.k, n) if self.k else n
        param_grid = [(ncd, h, k) for ncd in np.unique(self.ncompdis) for h in np.unique(self.h) for k in
                      np.unique(self.k)]
        npar = len(param_grid)

        results_y, results_fit, results_r = [], [], []

        for i, (zncompdis, zh, zk) in enumerate(param_grid):
            if self.print_out:
                print(f"\nparam ({i + 1}/{npar}) \n{zncompdis, zh, zk}\n")

            if zncompdis == 0:
                zresn = getknn(Xr, Xu, k=zk)
            else:
                global_model = PLS(ncomp=zncompdis)
                global_model.fit(Xr, Yr)
                zresn = getknn(global_model.T, global_model.transform(Xu), k=zk, diss=self.diss)

            zlistw = [self.wdist(d, zh,self.cri) for d in zresn['listd']]

            zfm = self.locw(Xr, Yr, Xu, listnn=zresn['listnn'], listw=zlistw, ncomp=self.ncomp)

            nr = zfm['y'].shape[0]
            z = pd.DataFrame({'ncompdis': [zncompdis] * nr, 'h': [zh] * nr})

            results_y.append(pd.concat([z, pd.DataFrame(zfm['y'])], axis=1))
           

        results_y = pd.concat(results_y).reset_index(drop=True)

        self.res_nn = {'listnn': zresn['listnn'], 'listd': zresn['listd'], 'listw': zlistw}
        self.param = pd.DataFrame(param_grid, columns=['ncompdis', 'h', 'k'])

        if self.print_out:
            print("\n\n")

        return {'y': results_y, 'res.nn': self.res_nn,
                'param': self.param}

    def wdist(self, d, h, cri=4, squared=False):
        d = d.flatten()
        if squared:
            d = d ** 2

        zmed = torch.median(d)
        zmad = torch.median(torch.abs(d - zmed))
        condition = d < (zmed + cri * zmad)
        w = torch.where(condition, torch.exp(-d / (h * zmad)), torch.tensor(0.0, dtype=torch.float64))
        w = w / torch.max(w)

        # Remplacer les NaN et Inf par 1
        w = torch.where(torch.isnan(w) | torch.isinf(w), torch.tensor(1.0, dtype=torch.float64), w)

        return w

    def locw(self, Xr, Yr, Xu, listnn, listw, ncomp):
        n = Xr.shape[0]
        m = Xu.shape[0]

        y_pred = torch.zeros((m, Yr.shape[1]), dtype=torch.float64)
        fit = torch.zeros_like(y_pred)
        r = torch.zeros_like(y_pred)

        for i in range(m):
            nn_indices = listnn[i]
            weights = listw[i]

            Xr_nn = Xr[nn_indices]
            Yr_nn = Yr[nn_indices]

            if torch.sum(weights) == 0:
                weights[:] = 1.0

            weights = weights / torch.sum(weights)
            local_pls = PLS(ncomp=ncomp,weights = weights)
            local_pls.fit(Xr_nn, Yr_nn)
            y_pred[i] = local_pls.predict(Xu[i])

        return {'y': y_pred}
    
if __name__ == "__main__":
    
    Xr = np.random.randn(1000,250)
    Xu = np.copy(Xr)
    Yr = np.random.randn(1000,1)
    model = LWPLSR(k = 20,ncomp = 20,ncompdis=0,diss = "euclidean")
    pred = model.fit(Xr,Yr,Xu)["y"][0]
    plt.scatter(pred,Yr[...,0])
    
