import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

bDotCntrllrGain_xMgntrqr = 10000
bDotCntrllrGain_yMgntrqr = 10000
bDotCntrllrGain_zMgntrqr = 10000
numCoils_xMgntrqr = 12000
numCoils_yMgntrqr = 12000
numCoils_zMgntrqr = 320
sctnArea_xMgntrqr = 2.2698E-4
sctnArea_yMgntrqr = 2.2698E-4
sctnArea_zMgntrqr = 8.7695E-3
maxCrrnt_xMgntrqr = 0.2143
maxCrrnt_yMgntrqr = 0.2143
maxCrrnt_zMgntrqr = 0.2135

# Magnetometer Parameters
lpFltrGain_mgntmtr = 0.4
sampRte_mgntmtr = 10
cnstntBiasVctr_mgntmtr = np.array([0, 0, 0])
maxNoiseAmpltde_mgntmtr = 1E-6

# Gyroscope Parameters
lpFltrGain_gyro = 0.45
sampRte_gyro = 10
cnstntBiasVctr_gyro = np.array([0, 0, 0])
maxNoiseAmpltde_gyro = 0.25

# Cubesat Parameters
mass_spcrft = 2.6
dimnsnsX_spcrft = 0.1
dimnsnsY_spcrft = 0.1
dimnsnsZ_spcrft = 0.2
inrtaTnsr_spcrft = np.array([[0.0108, 0, 0],
                              [0, 0.0108, 0],
                              [0, 0, 0.0043]])
initYawAngl_spcrft = 0
initPtchAngl_spcrft = 0
initRollAngl_spcrft = 0
initBdyAnglrRteX_spcrft = 10
initBdyAnglrRteY_spcrft = 10
initBdyAnglrRteZ_spcrft = 10
dsirdBdyAnglrRteX_spcrft = 0
dsirdBdyAnglrRteY_spcrft = 0
dsirdBdyAnglrRteZ_spcrft = 0

# Orbit Parameters
alt_orbt = 6E5
inc_orbt = 97.787
aop_orbt = 144.8873
raan_orbt = 59.4276
numOrbts = 2

# Simulator UI Parameters
scrnScleFctr_simUI = 0.75
frmeRte_simUI = 10
simSpdMltplr = 10
simCalcltnMdfier = 'Pre-Calculate'
viewAzmth = 135
viewElvtn = 22.5

plt.figure(figsize=(scrnScleFctr_simUI * COMP_SCRN_SIZE[3], scrnScleFctr_simUI * COMP_SCRN_SIZE[4]), facecolor='k')
plt.title("CubeSat Detumbling Simulator")
plt.xlim(0, 1)
plt.ylim(-5, 5)
plt.plot([0, 1, 1, 0, 0], [-0.5, -0.5, 0.5, 0.5, -0.5], color='r')
statusBar = plt.barh(0, 0, 0.9, color='g')
pausePlayBttn_simUI = plt.togglebutton('⏸', (15 / 64) * simUI.Position[3], 0, (1 / 32) * simUI.Position[3], (1 / 32) * simUI.Position[3], foregroundcolor='g', backgroundcolor='b')

# Orbit Plot Setup
semiMajr_orbt = R_EQTR_ERTH + alt_orbt
period_orbt = 2 * np.pi * np.sqrt(semiMajr_orbt ** 3 / MU_ERTH)
draan_dt = np.deg2rad((-3 / 2) * np.cos(np.deg2rad(inc_orbt)) * J2_ERTH * R_EQTR_ERTH ** 2 * np.sqrt(MU_ERTH) / semiMajr_orbt ** (7 / 2))
daop_dt = np.deg2rad((-3 / 2) * ((-5 / 2) * np.sin(np.deg2rad(inc_orbt)) ** 2 - 2) * J2_ERTH * R_EQTR_ERTH ** 2 * np.sqrt(MU_ERTH) / semiMajr_orbt ** (7 / 2))
prfclPostn_spcrft = np.array([semiMajr_orbt, 0, 0])
rotMatrx_orbt = rotMatrx313BdyToInrtl(raan_orbt, inc_orbt, aop_orbt)
inrtlPostn_spcrft = np.dot(rotMatrx_orbt, prfclPostn_spcrft)
ltitde_orbt = np.rad2deg(np.arcsin(inrtlPostn_spcrft[2, 0] / semiMajr_orbt))
lngtde_orbt = np.arctan2(inrtlPostn_spcrft[1, 0], inrtlPostn_spcrft[0, 0])
inrtlMgntcFld = 1E-9 * np.dot(rotMatrx321BdyToInrtl(lngtde_orbt, 270 - ltitde_orbt, 0), wrldmagm(alt_orbt, ltitde_orbt, lngtde_orbt, decyear(2020, 1, 1)))

plt.figure()
plt.xlim(0, R_EQTR_ERTH + 8E5)
plt.ylim(0, R_EQTR_ERTH + 8E5)
plt.plot([0, R_EQTR_ERTH + 8E5], [0, 0], color='w', linewidth=2)
plt.plot([0, 0], [0, R_EQTR_ERTH + 8E5], color='w', linewidth=2)
plt.text(R_EQTR_ERTH + 2E6, 0, 'X_I', color='w')
plt.text(0, R_EQTR_ERTH + 2E6, 'Y_I', color='w')
plt.scatter(inrtlPostn_spcrft[0, 0], inrtlPostn_spcrft[1, 0], color='y')
plt.show()

# Spacecraft Plot Setup
dimnsns_spcrft = np.array([dimnsnsX_spcrft, dimnsnsY_spcrft, dimnsnsZ_spcrft])
truAnmly_spcrft = 0
qtrnion_spcrft = np.array([
    [np.cos(np.radians(initRollAngl_spcrft / 2)) * np.cos(np.radians(initPtchAngl_spcrft / 2)) * np.cos(np.radians(initYawAngl_spcrft / 2)) + np.sin(np.radians(initRollAngl_spcrft / 2)) * np.sin(np.radians(initPtchAngl_spcrft / 2)) * np.sin(np.radians(initYawAngl_spcrft / 2))],
    [np.sin(np.radians(initRollAngl_spcrft / 2)) * np.cos(np.radians(initPtchAngl_spcrft / 2)) * np.cos(np.radians(initYawAngl_spcrft / 2)) - np.cos(np.radians(initRollAngl_spcrft / 2)) * np.sin(np.radians(initPtchAngl_spcrft / 2)) * np.sin(np.radians(initYawAngl_spcrft / 2))],
    [np.cos(np.radians(initRollAngl_spcrft / 2)) * np.sin(np.radians(initPtchAngl_spcrft / 2)) * np.cos(np.radians(initYawAngl_spcrft / 2)) + np.sin(np.radians(initRollAngl_spcrft / 2)) * np.cos(np.radians(initPtchAngl_spcrft / 2)) * np.sin(np.radians(initYawAngl_spcrft / 2))],
    [np.cos(np.radians(initRollAngl_spcrft / 2)) * np.cos(np.radians(initPtchAngl_spcrft / 2)) * np.sin(np.radians(initYawAngl_spcrft / 2)) - np.sin(np.radians(initRollAngl_spcrft / 2)) * np.sin(np.radians(initPtchAngl_spcrft / 2)) * np.cos(np.radians(initYawAngl_spcrft / 2))]
])

bdyAnglrRte_spcrft = np.array([
    [np.deg2rad(initBdyAnglrRteX_spcrft)],
    [np.deg2rad(initBdyAnglrRteY_spcrft)],
    [np.deg2rad(initBdyAnglrRteZ_spcrft)]
])

dsirdBdyAnglrRte_spcrft = np.array([
    [np.deg2rad(dsirdBdyAnglrRteX_spcrft)],
    [np.deg2rad(dsirdBdyAnglrRteY_spcrft)],
    [np.deg2rad(dsirdBdyAnglrRteZ_spcrft)]
])

dbdyAnglrRte_dt = np.linalg.inv(inrtaTnsr_spcrft) @ (-np.cross(bdyAnglrRte_spcrft, inrtaTnsr_spcrft @ bdyAnglrRte_spcrft))

dqtrnion_dt = np.array([
    [0, -bdyAnglrRte_spcrft[0, 0], -bdyAnglrRte_spcrft[1, 0], -bdyAnglrRte_spcrft[2, 0]],
    [bdyAnglrRte_spcrft[0, 0], 0, bdyAnglrRte_spcrft[2, 0], -bdyAnglrRte_spcrft[1, 0]],
    [bdyAnglrRte_spcrft[1, 0], -bdyAnglrRte_spcrft[2, 0], 0, bdyAnglrRte_spcrft[0, 0]],
    [bdyAnglrRte_spcrft[2, 0], bdyAnglrRte_spcrft[1, 0], -bdyAnglrRte_spcrft[0, 0], 0]
]) @ qtrnion_spcrft * 0.5

bdyMgntcFld_spcrft = np.array(quatrotate(qtrnion_spcrft.T, inrtlMgntcFld.T)).T

# Create spacecraft plot
fig_spcrft = plt.figure(figsize=(15, 10))
ax_spcrft = fig_spcrft.add_subplot(2, 3, 1, projection='3d')
ax_spcrft.set_xlim([0, dimnsns_spcrft[0]])
ax_spcrft.set_ylim([0, dimnsns_spcrft[1]])
ax_spcrft.set_zlim([0, dimnsns_spcrft[2]])
scptch, = ax_spcrft.plot([], [], [], color='r')
ax_spcrft.set_title('Spacecraft')
ax_spcrft.set_xlabel('X-axis')
ax_spcrft.set_ylabel('Y-axis')
ax_spcrft.set_zlabel('Z-axis')

# Update spacecraft plot
def update_spcrft_plot(i):
    global qtrnion_spcrft, bdyAnglrRte_spcrft, bdyMgntcFld_spcrft

    qtrnion_spcrft += dqtrnion_dt * simTmstp_spcrft
    qtrnion_spcrft /= np.linalg.norm(qtrnion_spcrft)
    bdyAnglrRte_spcrft += dbdyAnglrRte_dt * simTmstp_spcrft

    scptch.set_data(qtrnion_spcrft[1], qtrnion_spcrft[2])
    scptch.set_3d_properties(qtrnion_spcrft[3])

    return scptch,

# Plot
ani_spcrft = animation.FuncAnimation(fig_spcrft, update_spcrft_plot, interval=20)
plt.show()
