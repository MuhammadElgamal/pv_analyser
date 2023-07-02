import numpy as np
import pv_analyser as pv

panel = pv.panel()
measurement = pv.iv_measurement(current=np.linspace(0,10, 1000))
core = np.zeros((panel.rows, panel.columns))
rs, rp, idark, ideality, temp = core + 10e-3, core + 1e6, core+0.1e-9, core+1.1, core+25
iph1 = 9 + np.zeros((panel.rows, panel.columns//3))
iph2 = 8.5 + np.zeros((panel.rows, panel.columns//3))
iph3 = 8.5 + np.zeros((panel.rows, panel.columns//3))
iph = np.concatenate([iph1, iph2, iph3], axis=1)
measurement.state_matrix = [rs, rp, iph, idark, ideality,temp]
panel.load_measurement(source_type='measurement', source=measurement, type='taken')
panel.characterize()
panel.measurements[0].plot()
