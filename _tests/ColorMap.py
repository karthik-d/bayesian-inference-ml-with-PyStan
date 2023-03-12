import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv
import os
import statistics

'''
data = [[0.9999892359979918, 0.9999999850830539, 0.980495813447584, 0.9922220908461716, 0.9998774178710641, 0.9999939543236174, 0.9999997006837609],
  [0.9999760671838804, 0.9972856958620023, 0.5870666755068339, 0.9921941477529725, 0.1521533042608815, 7.852064804539515e-05, 0.9863139707991583],
  [0.012014845063213993, 0.0038839915321505487, 0.03858976623092149, 0.9922104492863714, 0.585877873308255, 0.7807895927352777, 0.4942813528143675],
  [0.9999892589452457, 0.9999950220610994, 0.5872452593314647, 0.9922028267211362, 0.1523742362724119, 0.9922349543654311, 0.9999892282523094],
  [0.999999701841041, 0.9999999850983435, 0.9806362261607964, 0.7804976071328551, 0.9998783747233437, 0.992260413197263, 0.9999997022872984],
  [9.343288862156207e-06, 0.00443569445323415, 3.100163758317391e-05, 2.177195665718455e-06, 0.0011094302613573381, 6.058029964216709e-08, 0.4985643867200132],
  [0.9999997008748082, 0.9999994597334935, 0.9805893594531424, 0.992222234312157, 0.9956193292223967, 0.9999939832749489, 0.9999892412988354],
  [0.9996157636594034, 0.9999991003118819, 0.9994475497902563, 0.7815741247395271, 0.15244253270619498, 0.9997828311943834, 0.9996131676582548],
  [0.9862557684565051, 0.9999999792627701, 0.9805898761122775, 0.9997826651831804, 0.9998775864186998, 0.9997829029306213, 0.9999890179148105],
  [0.9999997003686506, 0.9999994632556318, 0.038407217123490794, 0.999782257239702, 0.9998780435204312, 0.9922424621537839, 0.9999892475506765],
  [0.9996136002207443, 0.9999829228043536, 0.5864222526686838, 0.9997826820720339, 0.8632059840492257, 0.7812955832595841, 0.9999994961607487],
  [0.9680476728487798, 0.937900480159896, 0.9805060715521661, 0.002816004120843083, 0.994682536598024, 0.002808794241906389, 0.09400206711288159],
  [0.9999892719752945, 0.9999829453989061, 0.9806102615945844, 0.9922320748183168, 0.8639887705336715, 0.9997821947382256, 0.9999890713244655],
  [0.6604950381556343, 0.9998997602379123, 0.9805709475914349, 0.9922402960671729, 0.15276902332262823, 0.7809780178358005, 0.988110909699014],
  [0.9991419368058291, 0.9650035235121881, 0.587220848498918, 0.9922218887029366, 0.8536288041411508, 0.999782728498243, 0.9859312824551479],
  [0.9999892632374401, 0.9999999696279488, 0.5882339271151481, 0.999993944220598, 0.8637170007152497, 0.9999939560016633, 0.9999892855927215],
  [0.9999997014784345, 0.9999999848934885, 0.5865571722082893, 0.992222268034019, 0.9956172782023243, 0.9997828265220666, 0.9999997020717143],
  [0.9999892126288349, 0.9999999848900171, 0.9806069600804074, 0.9922488969772932, 0.9956239107798279, 0.7811928874994666, 0.9999997011013825],
  [0.9996070543811654, 0.9999610353070835, 0.9805661927537545, 0.9997824803970923, 0.8642214417986905, 0.7810406726815574, 0.9862718326041378],
  [0.9999997016113548, 0.9999999848916205, 0.9994505997250304, 0.9997825596356825, 0.9956045477751249, 0.9999939547464964, 0.9999997016731922],
  [0.9999892624676391, 0.9999999793193883, 0.9805233613217568, 0.9997823432677962, 0.9998777335587672, 0.9999939698724523, 0.9999892770028121],
  [0.9999892211001732, 0.9999728311212985, 0.5875701914423992, 0.9997823529175739, 0.9956173423148688, 0.9999939569026847, 0.9996056526362777]]

'''

data_new = [0.6062642608901654, 0.6715761162313131, 0.6648600288113027, 0.7417710964094536, 0.7005771212759447, 0.45397145241510284, 0.8228540024220771, 0.8236358725033595, 0.8622646615848806, 0.7329050971327568, 0.671468198780474, 0.666117068319676, 0.6639637128564916, 0.5850648279989602, 0.6726378386380121, 0.7830815917689851, 0.7819056475238658, 0.6729496713159977, 0.7007556868162256, 0.672496145972877, 0.6067333987612266, 0.17443246617365785, 0.5782527645981032, 0.6381189613718578, 0.7012547930581136, 0.7040776110515959, 0.7327847229487702, 0.7355911617700269, 0.7378482370968206, 0.14126787563178647, 0.4561635278175408, 0.5477370124620107, 0.6710901948021065, 0.7006935675232464, 0.45255418303845313, 0.7045276734466763, 0.5770635725980858, 0.7038549008762073, 0.6709805734231318, 0.4537868916171812, 0.6715232918832807, 0.7034724808052523, 0.6749447178614703, 0.6053890241768358, 0.5190679721403119, 0.5768287812416357, 0.3879268144124263, 0.5802050096402686]

data_comp = list()
with open(os.path.join(os.path.join(os.path.dirname(os.getcwd()), 'Data'), 'guess_probs.csv')) as f:
    reader = csv.reader(f)
    ctr = 0
    for row in reader:
        if(ctr==0):
            ctr += 1
            continue
        data_comp.append(row[3])
#print(data_comp)

diff = list()
div = list()
for i in range(len(data_new)):
    diff.append(abs(data_new[i]-float(data_comp[i])))
    div.append(data_new[i]/float(data_comp[i]))
print(diff)
print()
print(div)
print()
std_1 = (statistics.pstdev(diff))
mean_1 = (sum(diff)/len(diff))
print(std_1/mean_1)
print()
std_2 = (statistics.pstdev(div))
mean_2 = (sum(div)/len(div))
print("MEAN SCALE: ", mean_2)
print("COEFF. OF VARIATION: ", std_2/mean_2)

'''
rows = len(data)
cols = len(data[0])

fig, ax = plt.subplots(1, 1, tight_layout=True)
for x in range(rows + 1):
    ax.axhline(x, lw=1, color='black', zorder=5)
for x in range(cols+1):
    ax.axvline(x, lw=1, color='black', zorder=5)
ax.imshow(data, interpolation='none', cmap=plt.get_cmap('gray'), extent=[0, cols, 0, rows], zorder=0)
ax.axis('off')
plt.show()
'''
