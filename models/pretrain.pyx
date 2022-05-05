cimport numpy as np
import numpy as np
from six import iteritems

def pretrain_SVD(SVD, n_epochs=20, verbose=True,
                 lr_all=.005, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_all=.02, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None):

    # user biases
    cdef np.ndarray[np.double_t] bu
    # item biases
    cdef np.ndarray[np.double_t] bi
    # user factors
    cdef np.ndarray[np.double_t, ndim=2] pu
    # item factors
    cdef np.ndarray[np.double_t, ndim=2] qi

    cdef int u, i, f
    cdef double r, err, dot, puf, qif
    cdef double global_mean = SVD.global_mean

    cdef double lr_bu_ = lr_bu if lr_bu is not None else lr_all
    cdef double lr_bi_ = lr_bi if lr_bi is not None else lr_all
    cdef double lr_pu_ = lr_pu if lr_pu is not None else lr_all
    cdef double lr_qi_ = lr_qi if lr_qi is not None else lr_all

    cdef double reg_bu_ = reg_bu if reg_bu is not None else reg_all
    cdef double reg_bi_ = reg_bi if reg_bi is not None else reg_all
    cdef double reg_pu_ = reg_pu if reg_pu is not None else reg_all
    cdef double reg_qi_ = reg_qi if reg_qi is not None else reg_all

    rng = np.random.mtrand._rand

    bu = np.zeros(SVD.n_users, np.double)
    bi = np.zeros(SVD.n_items, np.double)
    pu = rng.normal(SVD.init_mean, SVD.init_std_dev,
                        (SVD.n_users, SVD.n_factors))
    qi = rng.normal(SVD.init_mean, SVD.init_std_dev,
                        (SVD.n_items, SVD.n_factors))

    if not SVD.biased:
        global_mean = 0

    for current_epoch in range(n_epochs):
        if verbose:
            print("Processing epoch {}".format(current_epoch))

        #for u, i, r in trainset.all_ratings():
        # for all (inner) u, i, r
        for u, u_ratings in iteritems(SVD.ur):
            for i, r in u_ratings:
                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(SVD.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if SVD.biased:
                    bu[u] += lr_bu_ * (err - reg_bu_ * bu[u])
                    bi[i] += lr_bi_ * (err - reg_bi_ * bi[i])

                # update factors
                for f in range(SVD.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu_ * (err * qif - reg_pu_ * puf)
                    qi[i, f] += lr_qi_ * (err * puf - reg_qi_ * qif)

    return pu, qi, bu, bi