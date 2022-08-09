f = open('spice.txt','w')
# ###################### cell 例化
# for i in range(1,5):
#     for j in range(1,4):
#         f.write('xcomcell'+ str(i)+str(j)+' vin'+str(i)+' rp'+str(i)+str(j)+' rn'+str(i)+str(j)+' vp'+str(j)+' vn'+str(j)+' ComCell_NoC\n')


###################### rp rn
q = [0,0,1]
for i in range(1,5):
    for j in range(1,4):
        vp = ' gnd pulse(0v vgg 1u 1n 1n 1u 100ms)\n' if q[j-1] else ' gnd pulse(0v vgg 100u 1n 1n 1u 100ms)\n'
        vn = ' gnd pulse(0v vgg 100u 1n 1n 1u 100ms)\n' if q[j-1] else ' gnd pulse(0v vgg 1u 1n 1n 1u 100ms)\n'
        f.write('vrp'+str(i)+str(j)+ ' rp'+str(i)+str(j)+vp+
                'vrn'+str(i)+str(j)+' rn'+str(i)+str(j)+vn)

f.close()
