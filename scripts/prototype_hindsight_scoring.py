import numpy as np

def legt_mats(N, theta):
    A = np.zeros((N,N)); 
    for n in range(N):
        for k in range(N):
            A[n,k] = ((-1.)**(n-k) if n>=k else 1.)*(2*n+1)/theta
    B = ((-1.)**np.arange(N))*(2*np.arange(N)+1)/theta
    return A, B

def legs_mats(N):
    A = np.zeros((N,N))
    for n in range(N):
        for k in range(N):
            if n>k: A[n,k]=np.sqrt((2*n+1)*(2*k+1))
            elif n==k: A[n,k]=n+1
    return A, np.sqrt(2*np.arange(N)+1)

def bilinear(A,B,dt):
    N=A.shape[0]; I=np.eye(N); M=I+dt/2*A
    return np.linalg.solve(M, I-dt/2*A), np.linalg.solve(M, dt*B)

N=48; LEVELS=np.linspace(-1,1,6); P=8; NOISE=0.05
THETAS=[16,32,64,128,256]
rng_global=np.random.default_rng(0)

def run(gap, seed, eta_scale=1.0):
    rng=np.random.default_rng(seed)
    toks=rng.integers(0,6,P); clean=np.concatenate([LEVELS[toks], np.zeros(gap)])
    T=len(clean); y=clean+NOISE*rng.standard_normal(T)
    comps=[]
    for th in THETAS:
        A,B=legt_mats(N,th); Ad,Bd=bilinear(A,B,1.0)
        comps.append(dict(name=f"legt{th}", Ad=Ad, Bd=Bd, legs=False, th=th))
    comps.append(dict(name="legs", legs=True, th=None))
    # states + input maps H (c_t = H @ y_{1:t})
    for c in comps:
        c['H']=np.zeros((N,0))
    logw=np.zeros(len(comps)); losses=np.zeros(len(comps))
    for t in range(T):
        for c in comps:
            if c['legs']:
                k=t+1; A,B=legs_mats(N)
                M=np.eye(N)+A/(2*k)
                Ad=np.linalg.solve(M,np.eye(N)-A/(2*k)); Bd=np.linalg.solve(M,B/k)
            else:
                Ad,Bd=c['Ad'],c['Bd']
            c['H']=np.concatenate([Ad@c['H'], Bd[:,None]],axis=1)
        # hindsight scoring every 5 steps after prefix
        if t>=P and t%5==0:
            ages=np.unique(np.geomspace(1,t,12).astype(int))
            idx=t-ages  # 0-based positions in stream
            for ci,c in enumerate(comps):
                H=c['H']; ct=H@y[:t+1]
                G=H@H.T+1e-6*np.eye(N)
                uhat=H.T@np.linalg.solve(G,ct)  # min-norm decode of full past
                err=np.mean((uhat[idx]-y[idx])**2)
                losses[ci]+=err
    eta=eta_scale/(2*NOISE**2)
    logw=-eta*losses; logw-=logw.max(); w=np.exp(logw); w/=w.sum()
    # recall: decode prefix tokens from each component's final state
    accs={}
    for ci,c in enumerate(comps):
        H=c['H']; ct=H@y
        G=H@H.T+1e-6*np.eye(N)
        uhat=H.T@np.linalg.solve(G,ct)
        pred=np.argmin(np.abs(uhat[:P,None]-LEVELS[None,:]),axis=1)
        accs[c['name']]=float((pred==toks).mean())
    sel=int(np.argmax(w))
    geo=np.exp(sum(w[i]*np.log(THETAS[i]) for i in range(len(THETAS)))/max(w[:len(THETAS)].sum(),1e-12))
    return dict(w=w, sel=comps[sel]['name'], geo_theta=geo, accs=accs,
                acc_selected=accs[comps[sel]['name']], w_legs=w[-1])

names=[f"legt{t}" for t in THETAS]+["legs"]
print(f"{'gap':>4} {'geo_theta':>9} {'selected':>9} {'acc_sel':>7} {'acc_oracle':>10} {'acc_legs':>8} {'acc_biggest':>11}")
for gap in [16, 64, 192]:
    geos=[]; asel=[]; aorc=[]; alegs=[]; abig=[]; sels=[]
    for seed in range(5):
        r=run(gap,seed)
        geos.append(r['geo_theta']); asel.append(r['acc_selected'])
        aorc.append(max(r['accs'][n] for n in names[:-1])); alegs.append(r['accs']['legs'])
        abig.append(r['accs']['legt256']); sels.append(r['sel'])
    print(f"{gap:>4} {np.mean(geos):>9.1f} {max(set(sels),key=sels.count):>9} {np.mean(asel):>7.2f} {np.mean(aorc):>10.2f} {np.mean(alegs):>8.2f} {np.mean(abig):>11.2f}")
