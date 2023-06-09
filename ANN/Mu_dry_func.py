def Mu_dry_func(Km,Mum,Kdry_cri,Mudry_cri,Phi_cri,Phi,method='Reuss'):
    if method=='Reuss':
        return (1/((1-Phi/Phi_cri)/Mum+Phi/Phi_cri/Mudry_cri))
    if method == 'Hash':
        z = Mudry_cri / 6 * (9 * Kdry_cri + 8 * Mudry_cri) / (Kdry_cri + 2 * Mudry_cri)
        Mudry = ((Phi / Phi_cri) / (Mudry_cri + z) + (1 - Phi / Phi_cri) / (Mum + z)) ** (-1) - z
        return Mudry