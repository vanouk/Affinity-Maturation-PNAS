#dictionary for Mixture project
import csv

####### parameters #######
time_Steps = [2, 200, 400, 600, 800]
singleAntigenTest = 0

cocktailFirst = 0
l = 46 #number of residues, 36 variable, 10 conserved

concsingle = 0.85
conccock   = 0.85
concseq0   = 0.747
concseq1   = 0.837
concseq2   = 0.610 #1.6*(4+8+8)/3+24.62/.610=51
GCDur1 = time_Steps[1] # 80
GCDur2 = time_Steps[2] # 130
GCDur3 = time_Steps[4] # 300

####### antigens creation #######
#initializer2
cons = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
Ag0 =[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1]+cons
Ag1 =[ 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1]+cons
Ag2 =[ 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1]+cons
AgC1=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1]+cons
AgC2=[-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1]+cons
AgC3=[-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1]+cons

if singleAntigenTest ==1:
        flag = 1
        dicAgs = {c:[Ag0] for c in list(range(time_Steps[0], time_Steps[4]))}
        dicconc= {c:concsingle for c in list(range(time_Steps[0], time_Steps[4]))}
        dicGCDur = {c:GCDur3 for c in list(range(time_Steps[0],time_Steps[4]))}
        with open('output_singleAntigenDict.csv', 'w') as output:
            w = csv.writer(output)
            w.writerows(dicAgs.items())
else:
        if cocktailFirst==1:
            flag = 0
            dicAgs = {c: [AgC1] for c in list(range(time_Steps[0],time_Steps[4]))}
            #dicAgs.update({c:[Ag1] for c in list(range(time_Steps[1],time_Steps[2]))})
            #dicAgs.update({c:[Ag2] for c in list(range(time_Steps[2],time_Steps[3]))})

            dicconc= {c:conccock for c in list(range(time_Steps[0], time_Steps[4]))}
            #dicconc.update({c:concseq1 for c in list(range(time_Steps[1], time_Steps[2]))})
            #dicconc.update({c:concseq2 for c in list(range(time_Steps[2], time_Steps[3]))})

            #dicGCDur = {c:GCDur3 for c in list(range(time_Steps[0],time_Steps[3]))}
            dicGCDur = {c:GCDur1 for c in list(range(time_Steps[0],time_Steps[4]))}
            #dicGCDur.update({c:GCDur2 for c in list(range(time_Steps[1]+1,time_Steps[2]+1))})
            #dicGCDur.update({c:GCDur3 for c in list(range(time_Steps[2]+1,time_Steps[3]))})

            with open('output_cocktailDict.csv', 'w') as output:
                w = csv.writer(output)
                w.writerows(dicAgs.items())

        else:
            flag = 0
            dicAgs = {c:[Ag0] for c in list(range(time_Steps[0],time_Steps[1]))}
            #dicAgs.update({c:[Ag1] for c in list(range(time_Steps[1],time_Steps[2]))})
            #dicAgs.update({c:[Ag2] for c in list(range(time_Steps[2],time_Steps[3]))})

            dicconc = {c:concseq0 for c in list(range(time_Steps[0], time_Steps[1]))}
            #dicconc.update({c:concseq1 for c in list(range(time_Steps[1], time_Steps[2]))})
            #dicconc.update({c:concseq2 for c in list(range(time_Steps[2], time_Steps[3]))})

            dicGCDur = {c:GCDur1 for c in list(range(time_Steps[0],time_Steps[1]+1))}
            #dicGCDur.update({c:GCDur2 for c in list(range(time_Steps[1]+1,time_Steps[2]+1))})
            #dicGCDur.update({c:GCDur3 for c in list(range(time_Steps[2]+1,time_Steps[3]))})

            with open('output_sequenceDict.csv', 'w') as output:
                w = csv.writer(output)
                w.writerows(dicAgs.items())

with open('output_GC_duration.csv', 'w') as output:
    w = csv.writer(output)
    w.writerows(dicGCDur.items())
