def Kdry_func(Km, Mum, Kdry_cri, Mudry_cri, Phi_cri, Phi, method='Reuss'):
    if method == 'Reuss':
        return (1 / ((1 - Phi / Phi_cri) / Km + Phi / Phi_cri / Kdry_cri))
    if method=='Hash':
        Kdry=(Phi/Phi_cri/(Kdry_cri+4/3*Mudry_cri)+(1-Phi/Phi_cri)/(Km+4/3*Mudry_cri))**(-1)-4/3*Mudry_cri
        return Kdry