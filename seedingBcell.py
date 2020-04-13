# Seeding B cells dictionary

import csv
import numpy as np

seedingCells = []

seed1 = [ 0.07022514,  0.2204461 ,  0.42665499,  0.48978106, -0.08748824,
        0.4018459 ,  0.46512763,  0.07906842, -0.11144974, -0.10954573,
        0.31138372, -0.16650237,  0.25556942,  0.52913136,  0.30569925,
        0.47569101,  0.21168804,  0.77017196, -0.0619979 ,  0.22410438,
        0.04557759,  0.08502408,  0.37117707,  0.42336269, -0.04063839,
        0.02473644,  0.44942242,  0.05532832,  0.30414634,  0.58170743,
        0.35008031,  0.57011912,  0.45295179,  0.36560962,  0.42266259,
        0.4925983 ,  0.49287121,  0.3367013 ,  0.41304339,  0.39157199,
        0.34381319,  0.55376669,  0.50666613,  0.53071259,  0.52794781,
        0.30872277]
seed2 = [ 0.4763202 , -0.00750242,  0.50981168, -0.1110068 ,  0.08621588,
       -0.15296583, -0.006131  ,  0.58770468,  0.30635156,  0.43274814,
       -0.03840287,  0.10743004,  0.01955905, -0.10572687, -0.01942612,
        0.72768483,  0.35760584,  0.58476254,  0.80232361,  0.23373264,
        0.33905151,  0.035612  , -0.0435892 ,  0.19144424,  0.63740894,
       -0.11873477,  0.31616745, -0.14206192,  0.39569959,  0.54806556,
        0.58523602,  0.49113161,  0.45072783,  0.39239811,  0.54479813,
        0.30021966,  0.31993536,  0.48794128,  0.5863878 ,  0.39503657,
        0.42806341,  0.3535234 ,  0.46992831,  0.53590933,  0.34168892,
        0.46607858]
seed3= [ 0.23420231,  0.48391786,  0.20578948, -0.09593152, -0.09054944,
       -0.06675985,  0.05296894,  0.23969947,  0.17343776,  0.18511504,
        0.30572588,  0.12049027,  0.64303802, -0.07746401,  0.30452652,
        0.70109097,  0.57374545,  0.23031899,  0.27144509,  0.05980147,
        0.40604154,  0.5426054 , -0.01041855,  0.07083773,  0.0287572 ,
        0.26727054,  0.11948966,  0.05715505,  0.59037625,  0.47110111,
        0.51371136,  0.39962547,  0.53878237,  0.33907573,  0.48012851,
        0.4557548 ,  0.39491043,  0.53099775,  0.34034315,  0.42156821,
        0.55484131,  0.38798872,  0.43076496,  0.48706499,  0.34068363,
        0.39591926]
seed4 = [ 0.63852984,  0.05601389, -0.08743775, -0.10890119,  0.58821259,
        0.33414853, -0.01694846,  0.59417283,  0.47111276, -0.15692496,
        0.7811553 ,  0.01153833, -0.04993064,  0.51918584, -0.0403107 ,
        0.22727924,  0.10339588,  0.28129638, -0.0979049 ,  0.22489821,
       -0.16671007, -0.037464  ,  0.75912559,  0.25097189, -0.17928689,
        0.10605613,  0.17914467,  0.74868231,  0.54784581,  0.51721094,
        0.33777189,  0.5678137 ,  0.40362156,  0.57277101,  0.45198082,
        0.42277353,  0.48755026,  0.37343167,  0.31832254,  0.57145633,
        0.49162959,  0.3510342 ,  0.41482123,  0.42488958,  0.38312194,
        0.43484894]
seed5 = [-0.0604083 , -0.07426669,  0.09660248,  0.22752482,  0.08703856,
        0.4770526 ,  0.84263226, -0.17621539,  0.12715806,  0.68758132,
        0.00508835,  0.18482192, -0.09270643, -0.02805335, -0.13491518,
        0.02693912,  0.23819318, -0.05947485,  0.34545408, -0.05850273,
        0.89224708,  0.49528519,  0.10862023,  0.12554359,  0.0518507 ,
        0.60667607,  0.27926041,  0.80369725,  0.47184963,  0.3705416 ,
        0.57381293,  0.49148579,  0.32319175,  0.34508816,  0.38638361,
        0.38010465,  0.40363076,  0.58641764,  0.59398491,  0.33204624,
        0.4472232 ,  0.30768067,  0.53458305,  0.49075589,  0.45950045,
        0.56942926]
seed6 = [ 0.54880374,  0.24259767, -0.14119298, -0.06936358,  0.08012165,
       -0.12839486,  0.60320955,  0.15279515,  0.54713153, -0.17100082,
        0.3693853 ,  0.59127644, -0.00969074,  0.86690806,  0.06344028,
       -0.12826925,  0.00121427,  0.81968462,  0.2827906 , -0.1052818 ,
       -0.06920948,  0.25419502,  0.13577124,  0.05030291, -0.14977253,
        0.35849033,  0.41378722,  0.72812563,  0.48663092,  0.45771724,
        0.3960233 ,  0.46260066,  0.57827784,  0.51988415,  0.5304786 ,
        0.40401849,  0.42832521,  0.5076202 ,  0.31598693,  0.39561952,
        0.37825966,  0.31422621,  0.54524773,  0.4559368 ,  0.38619857,
        0.36919778]
seed7 = [ 0.6363549 ,  0.44507192,  0.29732092,  0.52902552,  0.3040249 ,
        0.34199173,  0.06178911,  0.16052024,  0.00409182,  0.35002191,
        0.05535496,  0.25464141,  0.00238257,  0.0611243 ,  0.15749149,
       -0.06947978, -0.03365104,  0.49511172,  0.13192764,  0.3819879 ,
        0.86020781,  0.3499185 ,  0.03404943, -0.1058592 ,  0.07700861,
       -0.1267724 ,  0.34479756,  0.0929796 ,  0.54211867,  0.42524492,
        0.38825148,  0.41833987,  0.30781871,  0.465497  ,  0.56377509,
        0.42880441,  0.58860895,  0.56586655,  0.41620542,  0.53289162,
        0.4440626 ,  0.33024102,  0.43259584,  0.30062474,  0.42313986,
        0.41998865]
seed8 = [ 0.3413041 ,  0.11077598,  0.12106154,  0.16092818,  0.08959668,
        0.56113798,  0.28411997,  0.52910129,  0.43534789,  0.01677139,
        0.04952665,  0.65389299,  0.01289156,  0.10989395,  0.1711146 ,
        0.46657246, -0.07367154,  0.38128277,  0.19809692, -0.01250918,
       -0.1776713 ,  0.04605845, -0.06893825,  0.47526783,  0.09872034,
        0.11600937,  0.81865514,  0.09112034,  0.55185577,  0.43976328,
        0.34435652,  0.40957983,  0.40312365,  0.41810532,  0.51593349,
        0.31748431,  0.38153103,  0.38402443,  0.54237108,  0.40479187,
        0.58306431,  0.56497181,  0.50178709,  0.58394241,  0.32207986,
        0.39984202]
seed9 = [-0.13180874,  0.08209175,  0.62733607,  0.33831358,  0.5668193 ,
       -0.03108531, -0.08341429,  0.38291063,  0.17188873,  0.3288695 ,
        0.22258337,  0.34636319,  0.08749143,  0.58431269, -0.02350962,
        0.20643247,  0.44503547,  0.14580215,  0.79320902, -0.01336858,
        0.63130587,  0.31124387, -0.17448957, -0.13658289,  0.4555238 ,
        0.01453902, -0.02389564, -0.04021365,  0.56696916,  0.4654883 ,
        0.44694387,  0.57464991,  0.41279946,  0.35937767,  0.34522583,
        0.35470039,  0.37442886,  0.37416945,  0.48868719,  0.38650629,
        0.38505597,  0.37793278,  0.52548514,  0.40755148,  0.58342051,
        0.55950902]
seed10 = [ 0.45908634,  0.83077882, -0.11274575, -0.07273841, -0.05809056,
        0.80201407,  0.4313    ,  0.84512236,  0.13501391,  0.20399548,
       -0.12102912, -0.03088453,  0.08181477,  0.17863727,  0.66764808,
        0.31637055,  0.25700006,  0.03751564,  0.48992036,  0.29132027,
       -0.07344502,  0.40680374,  0.11959428,  0.04656967, -0.08110415,
        0.13573544,  0.11531185, -0.16828209,  0.39310269,  0.43168073,
        0.56814685,  0.43048861,  0.45313119,  0.44215871,  0.54056021,
        0.31866925,  0.32925295,  0.42801658,  0.3358829 ,  0.59342651,
        0.58474218,  0.50178241,  0.50967539,  0.37242006,  0.30539031,
        0.37026415]


for i in seed1, seed2, seed3, seed4, seed5, seed6, seed7, seed8, seed9, seed10:
    seedingCells.append(i)

with open('seedingCells.csv', 'w') as output:
    w = csv.writer(output)
    w.writerow(['res0','res1','res2','res3','res4','res5','res6','res7','res8','res9','res10','res11','res12','res13','res14','res15','res16','res17','res18','res19','res20','res21','res22','res23','res24','res25','res26','res27','res28','res29','res30','res31','res32','res33','res34','res35','res36','res37','res38','res39','res40','res41','res42','res43','res44','res45'])
    w.writerows(seedingCells)
for i in seedingCells: print(np.sum(i))