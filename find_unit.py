import jsbsim

ROOT = r"C:\Users\argoccc\Desktop\jsbsim-rl\jsbsim-data"
fdm = jsbsim.FGFDMExec(ROOT, None)
fdm.set_debug_level(0)
fdm.load_model('f16')
fdm['ic/h-sl-ft'] = 10000.0
fdm['ic/vt-fps'] = 500.0
fdm.run_ic()
fdm.run()  # tick once so derived values populate

candidates = [
    'aero/alpha-rad',
    'aero/alpha-deg',
    'aero/beta-rad',
    'aero/qbar-psf',
    'accelerations/Nz',
    'accelerations/Nx',
    'accelerations/Ny',
    'accelerations/n-pilot-x-norm',
    'accelerations/n-pilot-y-norm',
    'accelerations/n-pilot-z-norm',
    'accelerations/a-pilot-x-ft_sec2',
    'accelerations/a-pilot-z-ft_sec2',
    'accelerations/udot-ft_sec2',
    'accelerations/wdot-ft_sec2',
    'propulsion/engine[0]/n1',
    'propulsion/engine[0]/n2',
    'propulsion/engine[0]/thrust-lbs',
]

for prop in candidates:
    try:
        value = fdm[prop]
        print(f"  YES  {prop:50s} = {value}")
    except Exception as e:
        print(f"  NO   {prop:50s} ({type(e).__name__})")