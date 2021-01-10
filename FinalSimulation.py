# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
Modified by Mohammad Sharifzadeh
"""


import sys
import os
os.system('cls')
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.FATAL)



import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
import pynamics.integration


from sympy import sin
import sympy

import tanh

tol = 1e-3

#import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi



def Check_Feasibility_Before(freq,offset,ampl,l0,l1,l2,lw):

    const_length = 0
    const_collision = 0
    const_range = 0
    const_servo = 0
    const_beam = 0

    if (l0 + 2 *(l1+l2+lw) > 0.56):
        const_length = 1

    if (numpy.abs(ampl) > 90.2):
        const_range = 1
    elif(numpy.abs(ampl) < 4.8):
        const_range = 1
    elif(numpy.abs(offset) > 90.2):
        const_range = 1
    elif(numpy.abs(freq) > 1.21):
        const_range = 1
    elif(numpy.abs(freq) < 0.1):
        const_range = 1
    elif(numpy.abs(l0) > 0.186):
        const_range = 1
    elif(numpy.abs(l0) < 0.04):
        const_range = 1
    elif(numpy.abs(l1) > 0.165):
        const_range = 1
    elif(numpy.abs(l1) < 0.03):
        const_range = 1
    elif(numpy.abs(l2) > 0.165):
        const_range = 1
    elif(numpy.abs(l2) < 0.03):
        const_range = 1
    elif(numpy.abs(lw) > 0.165):
        const_range = 1
    elif(numpy.abs(lw) < 0.03):
        const_range = 1
    else:
         const_range = 0


    if (ampl*freq > 75):
        const_servo = 1


    vis = const_length + const_collision + const_range + const_servo + const_beam

    return vis

def Check_Feasibility_After(bmax,power_max,PBRmin,pABRmin):

    const_length = 0
    const_collision = 0
    const_range = 0
    const_servo = 0
    const_beam = 0
    const_power = 0

    if (PBRmin < 0):
        const_collision = 1

    if (pABRmin < 0):
        const_collision = 1

    if (bmax > 0.68):
        const_beam = 1

    if (power_max > (2*1.75*0.9)):
        const_power = 1

    vis = const_length + const_collision + const_range + const_servo + const_beam + const_power

    return vis

system = System()
pynamics.set_system(__name__,system)


# 0.8109   33.4578   57.3236    0.0403    0.1169    0.0313    0.1111

freq = 0.8109
offset =  33 - 90
ampl = 57.3236
l0 = 0.0403
l1 = 0.1169
l2 = 0.0313
lw = 0.1111
f_number = 1

print(freq,offset,ampl,l0,l1,l2,lw)

FR = freq
OF = offset
AM = ampl
LB0 = l0
LB1 = l1
LB2 = l2
LBW = lw
FN = f_number

vis1 = Check_Feasibility_Before(freq,offset,ampl,l0,l1,l2,lw)
print('Feasibility one is: ', vis1)
if vis1 == 0:
    sweep = AM*pi/180
    ll = 1


    BWidth = Constant(LB0*ll,'BWidth',system)

    lAR = Constant(LB1*ll,'lAR',system)
    lAL = Constant(LB1*ll,'lAL',system)
    #lA = Constant(.25,'lA',system)

    lBR = Constant(LB2*ll,'lBR',system)
    lBL = Constant(LB2*ll,'lBL',system)
    #lB = Constant(1,'lB',system)

    lCR = Constant(LBW*ll,'lCR',system)
    lCL = Constant(LBW*ll,'lCL',system)
    #lC = Constant(1,'lC',system)

    MperL = 0.0269/0.20533

    Mbody = 1e0

    mO = Constant(Mbody,'mO',system)
    mAR = Constant(MperL*LB1,'mAR',system)
    mAL = Constant(MperL*LB1,'mAL',system)
    mBR = Constant(MperL*LB2,'mBR',system)
    mBL = Constant(MperL*LB2,'mBL',system)
    mCR = Constant(1e-2,'mCR',system)
    mCL = Constant(1e-2,'mCL',system)

    b = Constant(1e-2,'b',system)

    b1 = Constant(1e-1,'b1',system)
    k1 = Constant(1/sweep,'k1',system)
    rho = Constant(1000,'rho',system)

    Cross_sectional_Area = 0.08 * LBW *ll**2
    S = Cross_sectional_Area

    S_Body_y = BWidth*0.05

    tinitial = 0
    tfinal = 10
    tstep = 1/30
    t = numpy.r_[tinitial:tfinal:tstep]

    preload2 = Constant(OF*pi/180,'preload2',system)

    Ixx_O = Constant(1e-1,'Ixx_O',system)
    Iyy_O = Constant(1e-1,'Iyy_O',system)
    Izz_O = Constant(1e-1,'Izz_O',system)

    x,x_d,x_dd = Differentiable('x',system)
    y,y_d,y_dd = Differentiable('y',system)
    qO,qO_d,qO_dd = Differentiable('qO',system)
    qAR,qAR_d,qAR_dd = Differentiable('qAR',system)
    qAL,qAL_d,qAL_dd = Differentiable('qAL',system)
    qBR,qBR_d,qBR_dd = Differentiable('qBR',system)
    qBL,qBL_d,qBL_dd = Differentiable('qBL',system)

    qCR = Constant(0,'qCR',system)
    qCL = Constant(0,'qCL',system)

    initialvalues = {}
    initialvalues[x]=0
    initialvalues[x_d] = 0.0
    initialvalues[y]=0
    initialvalues[y_d] = 0.0
    initialvalues[qO]=0
    initialvalues[qO_d] = 0
    initialvalues[qAR]=OF*pi/180
    initialvalues[qAR_d]=0*pi/180
    initialvalues[qAL]=-OF*pi/180
    initialvalues[qAL_d]=0*pi/180

    initialvalues[qBR]=0*pi/180
    initialvalues[qBR_d]=0*pi/180
    initialvalues[qBL]=0*pi/180
    initialvalues[qBL_d]=0*pi/180


    statevariables = system.get_state_variables()
    ini = [initialvalues[item] for item in statevariables]

    N = Frame('N')
    O = Frame('O')

    AR = Frame('AR')
    AL = Frame('AL')
    BR = Frame('BR')
    BL = Frame('BL')
    CR = Frame('CR')
    CL = Frame('CL')


    system.set_newtonian(N)
    O.rotate_fixed_axis_directed(N,[0,0,1],qO,system)
    AR.rotate_fixed_axis_directed(O,[0,0,1],qAR,system)
    AL.rotate_fixed_axis_directed(O,[0,0,1],qAL,system)
    BR.rotate_fixed_axis_directed(AR,[0,0,1],qBR,system)
    BL.rotate_fixed_axis_directed(AL,[0,0,1],qBL,system)
    CR.rotate_fixed_axis_directed(BR,[0,0,1],qCR,system)
    CL.rotate_fixed_axis_directed(BL,[0,0,1],qCL,system)



    pN = x*N.x + y*N.y
    pNO=pN

    pBDR = pNO + 0.5 * BWidth * O.x
    pABR =  pBDR +  lAR * AR.x
    pBCR = pABR + lBR * BR.x
    pWR  = pBCR + lCR * CR.x

    pBDL = pNO - 0.5 * BWidth * O.x
    pABL = pBDL + lAL * -AL.x
    pBCL = pABL + lBL * -BL.x
    pWL  = pBCL + lCL * -CL.x

    pOcm=pNO
    pARcm=pNO + lAR/2 * AR.x
    pBRcm=pABR + lBR/2 * BR.x
    pCRcm=pBCR + lCR/2 * CR.x

    pALcm=pNO + lAL/2 * -AL.x
    pBLcm=pABL + lBL/2 * -BL.x
    pCLcm=pBCL + lCL/2 * -CL.x


    wNO = N.getw_(O)


    wOAR = O.getw_(AR)
    wOAL = O.getw_(AL)

    wABR = AR.getw_(BR)
    wABL = AL.getw_(BL)
    
    vCRcm = pCRcm.time_derivative(N,system)
    vCRcm2  = vCRcm.dot(vCRcm)

    vCLcm = pCLcm.time_derivative(N,system)
    vCLcm2  = vCLcm.dot(vCLcm)

    vBody = pNO.time_derivative(N,system)
    vBody2  = vBody.dot(vBody)




    IO = Dyadic.build(O,Ixx_O,Iyy_O,Izz_O)
    BodyO = Body('BodyO',O,pOcm,mO,IO,system)


    BodyAR = Particle(pARcm,mAR,'ParticleAR',system)
    BodyAL = Particle(pALcm,mAL,'ParticleAL',system)

    BodyBR = Particle(pBRcm,mBR,'ParticleBR',system)
    BodyBL = Particle(pBLcm,mBL,'ParticleBL',system)

    BodyCR = Particle(pCRcm,mCR,'ParticleCR',system)
    BodyCL = Particle(pCLcm,mCL,'ParticleCL',system)

    fR = sweep*sin(2* numpy.pi * FR * system.t)
    fL = -sweep*sin(2* numpy.pi * FR * system.t)

    system.addforce(fR*N.z,wOAR)
    system.addforce(fL*N.z,wOAL)

    system.addforce(-b*wABR,wABR)
    system.addforce(-b*wABL,wABL)

    system.addforce(-b1*wOAR,wOAR)
    system.addforce(-b1*wOAL,wOAL)

    vRdx = vCRcm.dot(CR.x)
    vLdx = vCLcm.dot(CL.x)

    vRdy = vCRcm.dot(-CR.y)
    vLdy = vCLcm.dot(-CL.y)

    vBodyy = vBody.dot(-O.y)
    vBodyx = vBody.dot(-O.x)

    nvbody = 1/(vBody2**.5)*vBody

    tol1 = 1e-10
    angle_of_attack_BR = sympy.atan2(vRdy+tol1,vRdx)
    angle_of_attack_BL = sympy.atan2(vLdy+tol1,vLdx)
    angle_of_attack_Body = sympy.atan2(vBodyy+tol1,vBodyx)

    f_aero_BR = rho*vCRcm2*sympy.sin(angle_of_attack_BR)*S*BR.y
    system.addforce(f_aero_BR,vCRcm)

    f_aero_BL = rho*vCLcm2*sympy.sin(angle_of_attack_BL)*S*BL.y
    system.addforce(f_aero_BL,vCLcm)

    f_aero_Body = rho*vBody2*S_Body_y*nvbody
    system.addforce(-f_aero_Body,vBody)

    system.add_spring_force1(k1,(qAR-preload2)*N.z,wOAR)
    system.add_spring_force1(k1,(qAL+preload2)*N.z,wOAL)


    f2R = tanh.gen_spring_force(qBR,100, 0, -0.00866, 1*32.3783, 1*6.68, 0e1)
    f2L = tanh.gen_spring_force(-qBL,100, 0, -0.00866, 1*32.3783, 1*6.68, 0e1)

    system.addforce(-f2R*N.z,wABR)
    system.addforce(f2L*N.z,wABL)

    eq = []
    eq_d=[(system.derivative(item)) for item in eq]
    eq_dd=[(system.derivative(item)) for item in eq_d]


    f,ma = system.getdynamics()
    func1 = system.state_space_post_invert(f,ma,eq_dd)
    states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=tol,atol=tol,args=({'constants':system.constant_values},))

    points = [pWL,pBCL,pABL,pBDL,pNO,pBDR,pABR,pBCR,pWR]
    points_output = PointsOutput(points,system)
    yout = points_output.calc(states)


    beams_out_forces = [f2L,f2R]
    beam_output = Output(beams_out_forces,system)
    beams_out = beam_output.calc(states)
    plt.figure()
    plt.plot(t,beams_out)


    Angle_Arm1= [qAR,qBR]
    Angle_Arm1_output = Output(Angle_Arm1,system)
    Arm1_output = Angle_Arm1_output.calc(states)
    plt.figure()
    plt.plot(t,Arm1_output*180/numpy.pi)



    Servo_speed_par = [qAR_d]
    Servo_speed_output = Output(Servo_speed_par,system)
    Servo_speed = Servo_speed_output.calc(states)

    Servo_forces = [k1*(qAR-preload2)]
    Servo_forces_output = Output(Servo_forces,system)
    Servo_out = Servo_forces_output.calc(states)
    Servo_torque = sweep*numpy.sin(2* numpy.pi * FR * t);
    Servo_effective_torque = Servo_torque-Servo_out


    Servo_power = Servo_effective_torque * Servo_speed

    plt.figure()
    plt.plot(*(yout[::].T))
    plt.axis('equal')

#    force_output = Output([-k2*qO],system)
#    torque = force_output.calc(states)
#    force_output.plot_time()
#
#    plt.figure()
#    plt.plot(states[300:,1],torque[300:])
#
#    plt.figure()
#    plt.plot(states[300:,4],torque[300:])
#
#    plt.figure()
#    plt.plot(states[300:,4])

#    energy_output = Output([KE-PE],system)
#    energy_output.calc(states)
#
#    plt.figure()
#    plt.plot(energy_output.y)

    points_output.make_gif()
    points_output.render_movie()
    points_output.animate(fps = 10,movie_name = 'FinalSimulation.mp4',lw=2,marker='o',color=(1,0,0,1),linestyle='-')


    pWmin = numpy.min(yout[:,-1,0])
    pCmin = numpy.min(yout[:,-2,0])

    beam_max = numpy.max(beams_out)
    servo_power_max = numpy.max(Servo_power)

    final_pos = states[-1,1]

    vis2 = Check_Feasibility_After(beam_max,servo_power_max,pWmin,pCmin)

    print(vis2)
    if vis2 == 0:
        print('The run is feasible!')
